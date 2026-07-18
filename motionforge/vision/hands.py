"""Micro-gesture layer: MediaPipe Gesture Recognizer (21 hand landmarks).

Produces finger-level gestures the pose model can't see:
- states: fist_left/right, open_palm_left/right, pinch_left/right
  (pinch = thumb tip touching index tip — a click/grab with zero wrist
  movement, ideal for cursor games)
- pulses: thumbs_up, thumbs_down, victory, point_up (held ~0.3s to fire)

Hand sides are resolved by matching each detected hand to the nearest pose
wrist, which sidesteps MediaPipe's selfie-view handedness ambiguity.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from motionforge.core.events import GestureEvent, PULSE, START, END
from motionforge.vision.pose import MODELS_DIR

HAND_MODEL = MODELS_DIR / "gesture_recognizer.task"
HAND_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
                  "gesture_recognizer/float16/latest/gesture_recognizer.task")

# thumb tip=4, index tip=8, wrist=0, middle mcp=9
_PINCH_ON = 0.35     # thumb-index distance / hand size
_PINCH_OFF = 0.55


@dataclass
class HandObservation:
    wrist_img: tuple          # (x, y) normalized image coords
    gesture: str              # MediaPipe canned label, e.g. "Closed_Fist"
    pinch: bool
    score: float = 1.0
    side: str = ""            # filled in by the engine via pose-wrist matching


def ensure_hand_model() -> Path:
    if HAND_MODEL.exists() and HAND_MODEL.stat().st_size > 1_000_000:
        return HAND_MODEL
    import requests
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    resp = requests.get(HAND_MODEL_URL, timeout=120)
    resp.raise_for_status()
    HAND_MODEL.write_bytes(resp.content)
    return HAND_MODEL


class HandGestureEstimator:
    """Wraps the MediaPipe GestureRecognizer task (VIDEO mode). Skips frames
    adaptively when the combined pipeline would fall behind."""

    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            GestureRecognizer, GestureRecognizerOptions, RunningMode)
        options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_hand_model())),
            running_mode=RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.4,
        )
        self._rec = GestureRecognizer.create_from_options(options)
        self._mp = mp
        self._ts_ms = 0
        self.infer_ms = 0.0
        self._skip = False        # process every other frame when slow
        self._tick = 0

    def process(self, rgb, ts: float) -> list[HandObservation] | None:
        """Returns None on skipped frames (caller keeps previous states)."""
        self._tick += 1
        if self._skip and self._tick % 2:
            return None
        t0 = time.perf_counter()
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int(ts * 1000)
        if ts_ms <= self._ts_ms:
            ts_ms = self._ts_ms + 1
        self._ts_ms = ts_ms
        result = self._rec.recognize_for_video(mp_image, ts_ms)
        ms = (time.perf_counter() - t0) * 1000.0
        self.infer_ms = 0.9 * self.infer_ms + 0.1 * ms if self.infer_ms else ms
        self._skip = self.infer_ms > 18.0     # stay light on slow CPUs

        out: list[HandObservation] = []
        if not result.hand_landmarks:
            return out
        for i, lm in enumerate(result.hand_landmarks):
            pts = np.array([[p.x, p.y] for p in lm], dtype=np.float32)
            hand_size = float(np.linalg.norm(pts[9] - pts[0]))
            pinch_ratio = (float(np.linalg.norm(pts[4] - pts[8])) / hand_size
                           if hand_size > 1e-4 else 1.0)
            gesture, score = "None", 0.0
            if result.gestures and i < len(result.gestures) and result.gestures[i]:
                top = result.gestures[i][0]
                gesture, score = top.category_name, float(top.score)
            out.append(HandObservation(
                wrist_img=(float(pts[0][0]), float(pts[0][1])),
                gesture=gesture, pinch=pinch_ratio < _PINCH_ON, score=score))
        return out

    def close(self) -> None:
        self._rec.close()


# canned label -> (our gesture id prefix, is_state)
_LABELS = {
    "Closed_Fist": ("fist", True),
    "Open_Palm": ("open_palm", True),
    "Thumb_Up": ("thumbs_up", False),
    "Thumb_Down": ("thumbs_down", False),
    "Victory": ("victory", False),
    "Pointing_Up": ("point_up", False),
}


class _SideState:
    def __init__(self):
        self.active_state: str | None = None   # current held micro-gesture id
        self.cand: str | None = None
        self.cand_since = 0.0
        self.pulse_cand: str | None = None
        self.pulse_since = 0.0
        self.pulse_armed = True
        self.last_pulse = -1e9
        self.pinch_active = False
        self.pinch_since: float | None = None
        self.last_seen = 0.0


class HandTracker:
    """Turns per-frame hand observations into debounced gesture events."""

    STATE_DWELL = 0.12
    PULSE_DWELL = 0.30
    PULSE_COOLDOWN = 1.5
    LOST_AFTER = 0.5

    def __init__(self):
        self._sides = {"left": _SideState(), "right": _SideState()}

    def update(self, observations: list[HandObservation] | None, t: float) -> list[GestureEvent]:
        events: list[GestureEvent] = []
        if observations is None:            # skipped frame: hold states
            return events
        seen = {}
        for ob in observations:
            if ob.side in ("left", "right"):
                seen[ob.side] = ob
        for side, st in self._sides.items():
            ob = seen.get(side)
            if ob is None:
                if st.last_seen and t - st.last_seen > self.LOST_AFTER:
                    events += self._release(side, st, t)
                continue
            st.last_seen = t
            events += self._feed(side, st, ob, t)
        return events

    def _feed(self, side: str, st: _SideState, ob: HandObservation, t: float) -> list[GestureEvent]:
        events: list[GestureEvent] = []

        # pinch is landmark-derived and independent of the canned label
        if ob.pinch and not st.pinch_active:
            if st.pinch_since is None:
                st.pinch_since = t
            elif t - st.pinch_since >= self.STATE_DWELL:
                st.pinch_active = True
                events.append(GestureEvent(START, f"pinch_{side}", t, 1.0, t))
        elif not ob.pinch:
            st.pinch_since = None
            if st.pinch_active:
                st.pinch_active = False
                events.append(GestureEvent(END, f"pinch_{side}", t, 1.0, t))

        label = _LABELS.get(ob.gesture)
        state_id = f"{label[0]}_{side}" if label and label[1] else None

        # held micro-gestures (fist / open palm) with dwell debounce
        if state_id != st.cand:
            st.cand, st.cand_since = state_id, t
        if st.active_state != state_id and st.cand == state_id \
                and t - st.cand_since >= self.STATE_DWELL:
            if st.active_state:
                events.append(GestureEvent(END, st.active_state, t, 1.0, t))
            st.active_state = state_id
            if state_id:
                events.append(GestureEvent(START, state_id, t, ob.score, t))

        # momentary micro-gestures (thumbs up/down, victory, point up)
        pulse_id = label[0] if label and not label[1] else None
        if pulse_id != st.pulse_cand:
            st.pulse_cand, st.pulse_since = pulse_id, t
            if pulse_id is None:
                st.pulse_armed = True
        elif (pulse_id and st.pulse_armed
                and t - st.pulse_since >= self.PULSE_DWELL
                and t - st.last_pulse >= self.PULSE_COOLDOWN):
            st.pulse_armed = False
            st.last_pulse = t
            events.append(GestureEvent(PULSE, pulse_id, t, ob.score, t))
        return events

    def _release(self, side: str, st: _SideState, t: float) -> list[GestureEvent]:
        events = []
        if st.pinch_active:
            st.pinch_active = False
            events.append(GestureEvent(END, f"pinch_{side}", t, 1.0, t))
        if st.active_state:
            events.append(GestureEvent(END, st.active_state, t, 1.0, t))
            st.active_state = None
        st.cand = st.pulse_cand = None
        st.pinch_since = None
        st.last_seen = 0.0
        return events

    def release_all(self, t: float) -> list[GestureEvent]:
        out = []
        for side, st in self._sides.items():
            out += self._release(side, st, t)
        return out

    def active_states(self) -> list[str]:
        names = []
        for side, st in self._sides.items():
            if st.pinch_active:
                names.append(f"pinch_{side}")
            if st.active_state:
                names.append(st.active_state)
        return names
