"""Continuous control channels: camera-look (mouse aim) and hand-cursor.

Discrete gestures map to key/button events; these map body pose to an analog
signal every frame:
- head:        turn/tilt your head to aim
- lean:        lean torso left/right to yaw, forward/back to pitch
- right_hand / left_hand: raised hand acts as a virtual joystick
- cursor_hand: hand position maps to an absolute screen cursor (pointer games
               like Chess.com; pair with a 'push' or 'clap' gesture to click)
"""
from __future__ import annotations

from dataclasses import dataclass

from motionforge.gestures.primitives import Features
from motionforge.vision import pose as P


@dataclass
class LookOutput:
    mode: str = "vel"   # "vel" -> vx/vy in -1..1 joystick units; "abs" -> nx/ny 0..1
    x: float = 0.0
    y: float = 0.0
    active: bool = False
    click: bool = False  # dwell-click fired this frame (cursor_hand mode)


def _deadzone_scale(value: float, deadzone: float, full: float) -> float:
    mag = abs(value)
    if mag < deadzone:
        return 0.0
    out = min(1.0, (mag - deadzone) / max(full - deadzone, 1e-6))
    out = out * out  # expo curve: precise near center, fast at edges
    return out if value > 0 else -out


class LookController:
    """Converts per-frame Features into a look/cursor signal."""

    DWELL_TIME = 0.9        # s the cursor must stay still to dwell-click
    DWELL_RADIUS = 0.022    # img units counting as "still"
    DWELL_REARM = 0.045     # img units of movement required before next dwell

    def __init__(self, mode: str = "head"):
        self.mode = mode
        self.cursor_scale = 1.0                     # higher = less hand travel
        self.dwell_click = False
        self._neutral_nose_dy: float | None = None  # nose-above-shoulder baseline (head pitch)
        self._dwell_anchor: tuple[float, float] | None = None
        self._dwell_since: float | None = None
        self._dwell_armed = True

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self._neutral_nose_dy = None
        self._dwell_anchor = None
        self._dwell_since = None
        self._dwell_armed = True

    def set_dwell(self, enabled: bool) -> None:
        self.dwell_click = enabled
        self._dwell_anchor = None
        self._dwell_since = None
        self._dwell_armed = True

    def update(self, f: Features | None) -> LookOutput:
        if f is None or self.mode == "off":
            return LookOutput(active=False)
        try:
            return getattr(self, f"_{self.mode}")(f)
        except AttributeError:
            return LookOutput(active=False)

    # -- modes ------------------------------------------------------------

    def _head(self, f: Features) -> LookOutput:
        nose = f.w(P.NOSE)
        sho = f.shoulder_mid
        dx = float(nose[0] - sho[0])           # head turn shifts nose sideways
        dy = float(nose[1] - sho[1])           # nod shifts nose down, tilt up raises it
        if self._neutral_nose_dy is None:
            self._neutral_nose_dy = dy
        else:
            self._neutral_nose_dy += 0.002 * (dy - self._neutral_nose_dy)  # slow re-center
        # image x+ is the player's left; looking to their left should aim left,
        # and games treat negative mouse dx as left
        vx = -_deadzone_scale(dx, 0.030, 0.11)
        vy = -_deadzone_scale(dy - self._neutral_nose_dy, 0.022, 0.075)  # mouse y+ = look down
        return LookOutput("vel", vx, vy, active=True)

    def _lean(self, f: Features) -> LookOutput:
        yaw = -_deadzone_scale(f.lean_units, 0.06, 0.30)
        pitch_src = float(f.shoulder_mid[2] - f.hip_mid[2])   # forward lean -> negative z
        vy = _deadzone_scale(-pitch_src - 0.02, 0.05, 0.20)   # lean in -> look down
        return LookOutput("vel", yaw, vy, active=True)

    def _hand(self, f: Features, side_wrist: int, side_shoulder: int) -> LookOutput:
        if f.vis[side_wrist] < 0.4:
            return LookOutput(active=False)
        wrist, shoulder = f.w(side_wrist), f.w(side_shoulder)
        if wrist[2] > shoulder[2] - 0.08:      # only steer while arm is forward
            return LookOutput(active=False)
        dx = float(wrist[0] - shoulder[0])
        dy = float(wrist[1] - shoulder[1])
        vx = -_deadzone_scale(dx, 0.05, 0.30)  # image x+ = player's left = aim left
        vy = -_deadzone_scale(dy, 0.05, 0.25)
        return LookOutput("vel", vx, vy, active=True)

    def _right_hand(self, f: Features) -> LookOutput:
        return self._hand(f, P.R_WRIST, P.R_SHOULDER)

    def _left_hand(self, f: Features) -> LookOutput:
        return self._hand(f, P.L_WRIST, P.L_SHOULDER)

    def _cursor_hand(self, f: Features) -> LookOutput:
        """Right wrist position in a box in front of the shoulders maps to an
        absolute screen position. Scaled by shoulder width (not torso) so it
        works seated at a desk with hips out of frame. Mirrored so moving your
        hand left moves the cursor left from the player's point of view.
        Holding the cursor still dwell-clicks (when enabled)."""
        if f.vis[P.R_WRIST] < 0.4:
            self._dwell_since = None
            return LookOutput(active=False)
        # engage only while the hand reaches toward the screen (pointing
        # intent); a hand hanging at your side must not own the cursor
        if float(f.w(P.R_WRIST)[2]) > float(f.w(P.R_SHOULDER)[2]) - 0.10:
            self._dwell_anchor = None
            self._dwell_since = None
            return LookOutput(active=False)
        wrist = f.img[P.R_WRIST]
        sho = (f.img[P.L_SHOULDER] + f.img[P.R_SHOULDER]) / 2
        sw = abs(float(f.img[P.L_SHOULDER][0] - f.img[P.R_SHOULDER][0]))
        half_w = min(0.45, max(0.15, 2.0 * sw)) / max(self.cursor_scale, 0.25)
        half_h = min(0.35, max(0.12, 1.5 * sw)) / max(self.cursor_scale, 0.25)
        cx, cy = float(sho[0]), float(sho[1]) + 0.35 * half_h   # box sits slightly low
        nx = 1.0 - (float(wrist[0]) - cx + half_w) / (2 * half_w)  # mirror x
        ny = (float(wrist[1]) - cy + half_h) / (2 * half_h)
        # a hand hanging at your side (well outside the box) releases the
        # cursor instead of pinning it to a screen edge / dwell-clicking there
        if not (-0.15 <= nx <= 1.15 and -0.15 <= ny <= 1.15):
            self._dwell_anchor = None
            self._dwell_since = None
            return LookOutput(active=False)
        out = LookOutput("abs", min(1.0, max(0.0, nx)), min(1.0, max(0.0, ny)), active=True)

        if self.dwell_click:
            pos = (float(wrist[0]), float(wrist[1]))
            if self._dwell_anchor is None:
                self._dwell_anchor, self._dwell_since = pos, f.t
            else:
                dist = ((pos[0] - self._dwell_anchor[0]) ** 2
                        + (pos[1] - self._dwell_anchor[1]) ** 2) ** 0.5
                if dist > self.DWELL_RADIUS:
                    self._dwell_anchor, self._dwell_since = pos, f.t
                    if dist > self.DWELL_REARM:
                        self._dwell_armed = True
                elif (self._dwell_armed and self._dwell_since is not None
                        and f.t - self._dwell_since >= self.DWELL_TIME):
                    self._dwell_armed = False
                    self._dwell_since = None
                    out.click = True
        return out
