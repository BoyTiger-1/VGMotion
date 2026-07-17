"""Action execution semantics on top of the raw injector.

A profile binds each semantic action to an input with a mode:
- tap:        press+release (~35ms) on a pulse gesture
- double:     two quick taps on a pulse gesture
- toggle:     pulse flips the input between held and released
- hold:       held while a state gesture is active (crouch -> hold shift)
- hold_pulse: pulse presses and holds for hold_ms, repeat pulses extend the
              hold (repeated chops keep the mine button held in Minecraft)
"""
from __future__ import annotations

import heapq
import itertools
import threading
import time
from dataclasses import dataclass, field

from motionforge.core.events import PULSE, START, END
from motionforge.inputs.injector import InputInjector


@dataclass
class ActionBinding:
    input: str = "none"          # "key:w" | "mouse:left" | "wheel:up" | "none"
    mode: str = "tap"            # tap | double | toggle | hold | hold_pulse
    hold_ms: int = 600

    @classmethod
    def from_dict(cls, d: dict) -> "ActionBinding":
        return cls(input=d.get("input", "none"), mode=d.get("mode", "tap"),
                   hold_ms=int(d.get("hold_ms", 600)))

    def to_dict(self) -> dict:
        return {"input": self.input, "mode": self.mode, "hold_ms": self.hold_ms}


class _Scheduler(threading.Thread):
    """Tiny monotonic-time scheduler for delayed releases/taps."""

    def __init__(self):
        super().__init__(name="mf-scheduler", daemon=True)
        self._heap: list = []
        self._cv = threading.Condition()
        self._seq = itertools.count()
        self._running = True
        self.start()

    def at(self, when: float, fn) -> None:
        with self._cv:
            heapq.heappush(self._heap, (when, next(self._seq), fn))
            self._cv.notify()

    def run(self) -> None:
        while self._running:
            with self._cv:
                if not self._heap:
                    self._cv.wait(0.25)
                    continue
                when, _, fn = self._heap[0]
                now = time.perf_counter()
                if when > now:
                    self._cv.wait(min(when - now, 0.25))
                    continue
                heapq.heappop(self._heap)
            try:
                fn()
            except Exception:
                import traceback
                traceback.print_exc()

    def stop(self) -> None:
        self._running = False
        with self._cv:
            self._cv.notify()


class ActionExecutor:
    """Executes semantic actions; owns hold state and pending releases."""

    TAP_MS = 35

    def __init__(self, injector: InputInjector):
        self.injector = injector
        self._sched = _Scheduler()
        self._lock = threading.Lock()
        self._held: set[str] = set()             # binding strings currently held by us
        self._pulse_deadline: dict[str, float] = {}
        self.on_inject = None                     # callback(capture_ts) for latency stats

    def _mark(self, capture_ts: float) -> None:
        if self.on_inject and capture_ts:
            try:
                self.on_inject(capture_ts)
            except Exception:
                pass

    def handle(self, binding: ActionBinding, kind: str, capture_ts: float = 0.0) -> None:
        b = binding.input
        if b == "none" or not b:
            return
        mode = binding.mode
        if mode == "tap" and kind == PULSE:
            self.injector.binding_down(b)
            self._mark(capture_ts)
            self._sched.at(time.perf_counter() + self.TAP_MS / 1000, lambda: self.injector.binding_up(b))
        elif mode == "double" and kind == PULSE:
            self.injector.binding_down(b)
            self._mark(capture_ts)
            t0 = time.perf_counter()
            self._sched.at(t0 + 0.035, lambda: self.injector.binding_up(b))
            self._sched.at(t0 + 0.100, lambda: self.injector.binding_down(b))
            self._sched.at(t0 + 0.135, lambda: self.injector.binding_up(b))
        elif mode == "toggle" and kind == PULSE:
            with self._lock:
                held = b in self._held
                if held:
                    self._held.discard(b)
                else:
                    self._held.add(b)
            if held:
                self.injector.binding_up(b)
            else:
                self.injector.binding_down(b)
            self._mark(capture_ts)
        elif mode == "hold":
            if kind == START:
                with self._lock:
                    self._held.add(b)
                self.injector.binding_down(b)
                self._mark(capture_ts)
            elif kind == END:
                with self._lock:
                    self._held.discard(b)
                self.injector.binding_up(b)
            elif kind == PULSE:  # pulse gesture bound to a hold action: brief hold
                self._hold_pulse(b, 300, capture_ts)
        elif mode == "hold_pulse" and kind == PULSE:
            self._hold_pulse(b, binding.hold_ms, capture_ts)

    def _hold_pulse(self, b: str, hold_ms: int, capture_ts: float) -> None:
        deadline = time.perf_counter() + hold_ms / 1000
        with self._lock:
            fresh = b not in self._pulse_deadline
            self._pulse_deadline[b] = deadline
        if fresh:
            self.injector.binding_down(b)
            self._mark(capture_ts)
            self._sched.at(deadline, lambda: self._maybe_release(b))
        else:
            self._mark(capture_ts)

    def _maybe_release(self, b: str) -> None:
        with self._lock:
            deadline = self._pulse_deadline.get(b)
            if deadline is None:
                return
            now = time.perf_counter()
            if now < deadline - 0.005:            # extended by a repeat pulse
                self._sched.at(deadline, lambda: self._maybe_release(b))
                return
            del self._pulse_deadline[b]
        self.injector.binding_up(b)

    def release_all(self) -> None:
        with self._lock:
            self._held.clear()
            self._pulse_deadline.clear()
        self.injector.release_all()

    def stop(self) -> None:
        self.release_all()
        self._sched.stop()


class LookThread(threading.Thread):
    """Applies the continuous look signal as smooth mouse movement at 120Hz.

    vel mode: joystick units (-1..1) -> pixels/second, fractional remainder
    carried between ticks so slow aim is still smooth.
    abs mode: eased absolute cursor positioning for pointer-style games."""

    RATE_HZ = 120
    MAX_SPEED = 1500.0   # px/s at full deflection, scaled by sensitivity

    def __init__(self, injector: InputInjector):
        super().__init__(name="mf-look", daemon=True)
        self.injector = injector
        self.sensitivity = 1.0
        self._lock = threading.Lock()
        self._mode = "vel"
        self._x = 0.0
        self._y = 0.0
        self._active = False
        self._enabled = False
        self._acc_x = 0.0
        self._acc_y = 0.0
        self._abs_pos: tuple[float, float] | None = None
        self._running = True
        self.start()

    def set_signal(self, mode: str, x: float, y: float, active: bool) -> None:
        with self._lock:
            self._mode, self._x, self._y, self._active = mode, x, y, active

    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._enabled = enabled
            if not enabled:
                self._acc_x = self._acc_y = 0.0

    def run(self) -> None:
        dt = 1.0 / self.RATE_HZ
        while self._running:
            time.sleep(dt)
            with self._lock:
                mode, x, y = self._mode, self._x, self._y
                active = self._active and self._enabled
                sens = self.sensitivity
            if not active:
                continue
            if mode == "vel":
                if x == 0.0 and y == 0.0:
                    continue
                self._acc_x += x * self.MAX_SPEED * sens * dt
                self._acc_y += y * self.MAX_SPEED * sens * dt
                dx, dy = int(self._acc_x), int(self._acc_y)
                if dx or dy:
                    self._acc_x -= dx
                    self._acc_y -= dy
                    self.injector.move_rel(dx, dy)
            else:  # abs cursor: ease toward target to avoid jitter
                if self._abs_pos is None:
                    self._abs_pos = (x, y)
                px, py = self._abs_pos
                # gentle easing + deadband keeps the cursor rock-steady for
                # dwell-clicking while still tracking deliberate movement
                if abs(x - px) + abs(y - py) < 0.0015:
                    continue
                nx, ny = px + 0.18 * (x - px), py + 0.18 * (y - py)
                self._abs_pos = (nx, ny)
                self.injector.move_abs_norm(nx, ny)

    def stop(self) -> None:
        self._running = False
