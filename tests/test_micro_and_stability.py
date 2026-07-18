"""Regressions for field reports: phantom detections while standing still,
punches dragging the cursor, and the finger-level micro-gesture layer."""
import numpy as np

from motionforge.core.events import PULSE, START, END
from motionforge.core.selftest_data import _Builder, FPS
from motionforge.gestures.continuous import LookController
from motionforge.gestures.primitives import FeatureExtractor
from motionforge.gestures.recognizer import GestureRecognizer
from motionforge.vision import pose as P
from motionforge.vision.hands import HandObservation, HandTracker


# ---- phantom motion: depth noise with a still body -------------------------

def test_depth_noise_fires_nothing():
    """MediaPipe's z estimate jitters hard even when the player is a statue.
    World-space noise with no on-camera movement must fire zero gestures."""
    b = _Builder()
    b.hold(30)
    rng = np.random.default_rng(11)
    for _ in range(120):   # 4 seconds of violent depth noise, static image
        for lm in (P.L_WRIST, P.R_WRIST, P.L_ELBOW, P.R_ELBOW):
            b.world0[lm][2] = rng.uniform(-0.45, 0.15)
        b.emit()
    ex, rec = FeatureExtractor(), GestureRecognizer()
    pulses = []
    for pf in b.frames:
        pulses += [e.name for e in rec.update(ex.update(pf)) if e.kind == PULSE]
    assert pulses == [], f"depth noise fired: {pulses}"


def test_full_noise_storm_is_quiet():
    """Small 3D noise on every landmark (realistic idle webcam wobble) must
    produce no pulse gestures."""
    b = _Builder()
    b.hold(30)
    rng = np.random.default_rng(5)
    base_world = b.world0.copy()
    base_img = b.img0.copy()
    for _ in range(150):
        b.world0 = base_world + rng.normal(0, 0.012, base_world.shape).astype(np.float32)
        b.img0 = base_img + rng.normal(0, 0.0035, base_img.shape).astype(np.float32)
        b.emit()
    ex, rec = FeatureExtractor(), GestureRecognizer()
    pulses = []
    for pf in b.frames:
        pulses += [e.name for e in rec.update(ex.update(pf)) if e.kind == PULSE]
    assert pulses == [], f"idle wobble fired: {pulses}"


# ---- punch must not drag the cursor ----------------------------------------

def test_ballistic_hand_freezes_cursor():
    look = LookController("cursor_hand")
    look.set_dwell(True)
    b = _Builder()
    b.world0[P.R_WRIST] = (-0.10, 0.30, -0.30)   # aiming pose
    b.img0[P.R_WRIST] = (0.42, 0.42)
    b.hold(20)
    # punch: fast wrist travel
    b.interpolate(4, world_targets={P.R_WRIST: (-0.12, 0.42, -0.62)}, keep=True)
    ex = FeatureExtractor()
    frames = [ex.update(pf) for pf in b.frames]
    active_during_punch = [look.update(f).active for f in frames[-4:]]
    assert not any(active_during_punch), "cursor still steering during a punch"


# ---- micro gestures ---------------------------------------------------------

def obs(side, gesture="None", pinch=False):
    o = HandObservation(wrist_img=(0.4, 0.4), gesture=gesture, pinch=pinch)
    o.side = side
    return o


def run_tracker(script):
    """script: list of (list_of_observations | None) per frame at 30fps."""
    tr = HandTracker()
    events = []
    for i, frame_obs in enumerate(script):
        events += tr.update(frame_obs, i / FPS)
    return tr, events


def test_pinch_start_end():
    frames = [[obs("right", pinch=True)]] * 10 + [[obs("right", pinch=False)]] * 5
    tr, events = run_tracker(frames)
    kinds = [(e.kind, e.name) for e in events]
    assert (START, "pinch_right") in kinds
    assert (END, "pinch_right") in kinds


def test_fist_state_and_release_on_loss():
    frames = [[obs("left", gesture="Closed_Fist")]] * 12
    tr, events = run_tracker(frames)
    assert (START, "fist_left") in [(e.kind, e.name) for e in events]
    # hand disappears -> release after grace period
    more = []
    for i in range(30):
        more += tr.update([], (12 + i) / FPS)
    assert (END, "fist_left") in [(e.kind, e.name) for e in more]


def test_thumbs_up_pulse_requires_dwell_and_rearm():
    # 3 frames (0.1s) is below the 0.3s dwell -> no fire
    _, events = run_tracker([[obs("right", gesture="Thumb_Up")]] * 3)
    assert not any(e.name == "thumbs_up" for e in events)
    # held 15 frames (0.5s) -> exactly one pulse even if held longer
    _, events = run_tracker([[obs("right", gesture="Thumb_Up")]] * 45)
    fires = [e for e in events if e.name == "thumbs_up" and e.kind == PULSE]
    assert len(fires) == 1


def test_skipped_frames_hold_state():
    frames = [[obs("right", pinch=True)]] * 10 + [None] * 20   # estimator skipping
    tr, events = run_tracker(frames)
    kinds = [(e.kind, e.name) for e in events]
    assert (START, "pinch_right") in kinds
    assert (END, "pinch_right") not in kinds   # skips must not drop the pinch


def test_micro_gestures_are_mappable():
    from motionforge.gestures.library import GESTURE_DESCRIPTIONS, STATE_GESTURES
    for g in ("pinch_left", "pinch_right", "fist_left", "fist_right",
              "open_palm_left", "open_palm_right", "thumbs_up", "thumbs_down",
              "victory", "point_up"):
        assert g in GESTURE_DESCRIPTIONS, g
    assert {"pinch_left", "pinch_right", "fist_left", "fist_right"} <= STATE_GESTURES
