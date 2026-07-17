"""Temporal gesture recognizer.

Runs all detectors per frame, then disambiguates pulses in two stages so one
physical motion produces exactly one action:

1. Decision buffer: pulse candidates are collected for a short window
   (~one frame) before the highest-priority candidate is emitted. A punch
   whose early frames also look like a swing resolves to the punch.
2. Cross-gesture suppression: after a pulse is emitted, *different* pulse
   gestures are suppressed for SUPPRESS_AFTER seconds (the tail of a throw
   must not fire a swing). Repeats of the SAME gesture stay allowed so
   rapid actions (repeated mining chops) keep working, limited only by the
   detector's own cooldown.
"""
from __future__ import annotations

from motionforge.core.events import GestureEvent, PULSE
from motionforge.gestures import library
from motionforge.gestures.primitives import Features

DECISION_WINDOW = 0.04   # s to collect competing pulse candidates (~1 frame)
SUPPRESS_AFTER = 0.35    # s other pulse gestures stay muted after an emit


class GestureRecognizer:
    def __init__(self, sensitivity: float = 1.0, accessibility: str = "standing"):
        self.sensitivity = sensitivity
        self.accessibility = accessibility
        self.detectors = library.build_detectors(sensitivity, accessibility)
        self._pending: list[tuple[int, GestureEvent]] = []
        self._pending_since: float | None = None
        self._suppress_until = -1e9
        self._suppress_except = ""      # gesture name allowed through suppression

    def configure(self, sensitivity: float | None = None, accessibility: str | None = None) -> None:
        if sensitivity is not None:
            self.sensitivity = sensitivity
        if accessibility is not None:
            self.accessibility = accessibility
        self.detectors = library.build_detectors(self.sensitivity, self.accessibility)
        self._pending.clear()
        self._pending_since = None

    def update(self, f: Features | None) -> list[GestureEvent]:
        if f is None:
            return self.release_all_states()

        out: list[GestureEvent] = []
        for d in self.detectors:
            try:
                for ev in d.update(f):
                    if ev.kind == PULSE:
                        if f.t < self._suppress_until and ev.name != self._suppress_except:
                            continue
                        self._pending.append((d.priority, ev))
                        if self._pending_since is None:
                            self._pending_since = f.t
                    else:
                        out.append(ev)
            except Exception:
                import traceback
                traceback.print_exc()

        # resolve the decision buffer once its window has elapsed
        if self._pending_since is not None and f.t - self._pending_since >= DECISION_WINDOW:
            self._pending.sort(key=lambda pe: -pe[0])
            best = self._pending[0][1]
            self._pending.clear()
            self._pending_since = None
            self._suppress_until = f.t + SUPPRESS_AFTER
            self._suppress_except = best.name
            out.append(best)
        return out

    def release_all_states(self) -> list[GestureEvent]:
        """Person left the frame / pipeline pausing: end all held states."""
        import time
        out: list[GestureEvent] = []
        now = time.perf_counter()
        self._pending.clear()
        self._pending_since = None
        for d in self.detectors:
            if isinstance(d, library.StateDetector) and d.active:
                d.reset()
                out.append(GestureEvent("end", d.name, now, 1.0, now))
            elif isinstance(d, library.WalkInPlaceDetector):
                if d.sprinting:
                    out.append(GestureEvent("end", "sprint", now, 1.0, now))
                if d.walking:
                    out.append(GestureEvent("end", "walk", now, 1.0, now))
                d.reset()
            elif isinstance(d, library.ClimbDetector):
                if d.climbing:
                    out.append(GestureEvent("end", "climb", now, 1.0, now))
                d.reset()
        return out

    def active_states(self) -> list[str]:
        names = [d.name for d in self.detectors
                 if isinstance(d, library.StateDetector) and d.active]
        for d in self.detectors:
            if isinstance(d, library.WalkInPlaceDetector):
                if d.walking:
                    names.append("walk")
                if d.sprinting:
                    names.append("sprint")
            elif isinstance(d, library.ClimbDetector) and d.climbing:
                names.append("climb")
        return names
