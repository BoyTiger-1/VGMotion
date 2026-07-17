"""Built-in gesture detectors.

Two kinds:
- PulseDetector: one-shot events (punch, clap, throw) with cooldowns
- StateDetector: sustained poses (crouching, blocking, walking) that emit
  START when entered and END when released, with dwell + hysteresis to
  reject flicker.

All thresholds are expressed in body-relative units (meters in hip-origin
world space, or torso-lengths for whole-body motion), so they transfer
across players and camera placements. `sens` scales velocity thresholds:
sensitivity 1.5 means gestures trigger with ~2/3 the speed.
"""
from __future__ import annotations

import math

from motionforge.core.events import GestureEvent, PULSE, START, END
from motionforge.gestures.primitives import Features
from motionforge.vision import pose as P

LEFT, RIGHT = "left", "right"
_SIDE = {
    LEFT: dict(wrist=P.L_WRIST, elbow=P.L_ELBOW, shoulder=P.L_SHOULDER,
               knee=P.L_KNEE, ankle=P.L_ANKLE, hip=P.L_HIP),
    RIGHT: dict(wrist=P.R_WRIST, elbow=P.R_ELBOW, shoulder=P.R_SHOULDER,
                knee=P.R_KNEE, ankle=P.R_ANKLE, hip=P.R_HIP),
}


class Detector:
    name: str = ""
    priority: int = 50
    uses_arm: str | None = None      # left/right, for one-handed accessibility filtering
    body_required: str = "any"       # "standing" if it needs hip baseline (jump/crouch)

    def __init__(self, sens: float = 1.0):
        self.sens = max(0.25, sens)

    def update(self, f: Features) -> list[GestureEvent]:
        raise NotImplementedError

    def reset(self) -> None:
        pass


class PulseDetector(Detector):
    cooldown = 0.5

    def __init__(self, sens: float = 1.0):
        super().__init__(sens)
        self._last_fire = -1e9

    def _fire(self, f: Features, confidence: float = 1.0) -> list[GestureEvent]:
        if f.t - self._last_fire < self.cooldown:
            return []
        self._last_fire = f.t
        return [GestureEvent(PULSE, self.name, f.t, confidence, f.capture_ts)]

    def reset(self) -> None:
        self._last_fire = -1e9


class StateDetector(Detector):
    enter_dwell = 0.10   # condition must hold this long before START

    def __init__(self, sens: float = 1.0):
        super().__init__(sens)
        self.active = False
        self._cond_since: float | None = None

    def _enter_condition(self, f: Features) -> bool:
        raise NotImplementedError

    def _exit_condition(self, f: Features) -> bool:
        """Default: exit when enter condition fails. Override for hysteresis."""
        return not self._enter_condition(f)

    def update(self, f: Features) -> list[GestureEvent]:
        if not self.active:
            if self._enter_condition(f):
                if self._cond_since is None:
                    self._cond_since = f.t
                if f.t - self._cond_since >= self.enter_dwell:
                    self.active = True
                    self._cond_since = None
                    return [GestureEvent(START, self.name, f.t, 1.0, f.capture_ts)]
            else:
                self._cond_since = None
        else:
            if self._exit_condition(f):
                self.active = False
                self._cond_since = None
                return [GestureEvent(END, self.name, f.t, 1.0, f.capture_ts)]
        return []

    def reset(self) -> None:
        self.active = False
        self._cond_since = None


# --------------------------------------------------------------------------
# Whole-body (image-baseline) gestures
# --------------------------------------------------------------------------

class JumpDetector(PulseDetector):
    name = "jump_in_place"
    priority = 85
    cooldown = 0.7
    body_required = "standing"

    def update(self, f: Features) -> list[GestureEvent]:
        if not f.visible(P.L_HIP, P.R_HIP):   # hips guessed off-frame = no jump
            return []
        if f.hip_units > 0.22 and f.hip_units_vel > 1.2 / self.sens:
            return self._fire(f)
        return []


class CrouchDetector(StateDetector):
    name = "crouch"
    body_required = "standing"
    enter_dwell = 0.08

    def _enter_condition(self, f: Features) -> bool:
        return f.visible(P.L_HIP, P.R_HIP) and f.hip_units < -0.30

    def _exit_condition(self, f: Features) -> bool:
        return f.hip_units > -0.18 or not f.visible(P.L_HIP, P.R_HIP)


class LeanSideDetector(StateDetector):
    """side='left' means the PLAYER leans to their left."""
    enter_dwell = 0.08

    def __init__(self, side: str, sens: float = 1.0):
        super().__init__(sens)
        self.side = side
        self.name = f"lean_{side}"
        # player's left = image x+ (un-mirrored webcam view)
        self._sign = 1.0 if side == LEFT else -1.0

    def _enter_condition(self, f: Features) -> bool:
        return self._sign * f.lean_units > 0.16

    def _exit_condition(self, f: Features) -> bool:
        return self._sign * f.lean_units < 0.10


class LeanForwardDetector(StateDetector):
    name = "lean_forward"
    enter_dwell = 0.10

    def _enter_condition(self, f: Features) -> bool:
        return (f.visible(P.L_HIP, P.R_HIP)
                and float(f.shoulder_mid[2] - f.hip_mid[2]) < -0.14)  # shoulders toward camera

    def _exit_condition(self, f: Features) -> bool:
        return (float(f.shoulder_mid[2] - f.hip_mid[2]) > -0.09
                or not f.visible(P.L_HIP, P.R_HIP))


class LeanBackDetector(StateDetector):
    name = "lean_back"
    enter_dwell = 0.10

    def _enter_condition(self, f: Features) -> bool:
        return (f.visible(P.L_HIP, P.R_HIP)
                and float(f.shoulder_mid[2] - f.hip_mid[2]) > 0.12)

    def _exit_condition(self, f: Features) -> bool:
        return (float(f.shoulder_mid[2] - f.hip_mid[2]) < 0.07
                or not f.visible(P.L_HIP, P.R_HIP))


class WalkInPlaceDetector(Detector):
    """Marching in place -> 'walk' state; fast cadence adds 'sprint'.
    Works standing or seated (knee raises)."""
    name = "walk"
    RAISE_AT = -0.30      # knee y (m, rel hips) counts as raised above this
    LOWER_AT = -0.38
    WINDOW = 1.15         # steps must be at least this frequent to stay walking
    SPRINT_INTERVAL = 0.40

    def __init__(self, sens: float = 1.0):
        super().__init__(sens)
        self._knee_up = {LEFT: False, RIGHT: False}
        self._steps: list[float] = []
        self.walking = False
        self.sprinting = False

    def update(self, f: Features) -> list[GestureEvent]:
        events: list[GestureEvent] = []
        if not f.visible(P.L_KNEE, P.R_KNEE):
            # legs out of frame: never phantom-walk; release if currently walking
            if self.walking:
                out = []
                if self.sprinting:
                    self.sprinting = False
                    out.append(GestureEvent(END, "sprint", f.t, 1.0, f.capture_ts))
                self.walking = False
                out.append(GestureEvent(END, "walk", f.t, 1.0, f.capture_ts))
                self._steps.clear()
                return out
            return events
        for side in (LEFT, RIGHT):
            knee_y = float(f.w(_SIDE[side]["knee"])[1])
            if not self._knee_up[side] and knee_y > self.RAISE_AT:
                self._knee_up[side] = True
                self._steps.append(f.t)
            elif self._knee_up[side] and knee_y < self.LOWER_AT:
                self._knee_up[side] = False
        self._steps = [t for t in self._steps if f.t - t < 2.5]

        recent = [t for t in self._steps if f.t - t < self.WINDOW]
        should_walk = len(recent) >= 2
        if should_walk and not self.walking:
            self.walking = True
            events.append(GestureEvent(START, "walk", f.t, 1.0, f.capture_ts))
        elif self.walking and (not self._steps or f.t - self._steps[-1] > 0.7):
            self.walking = False
            if self.sprinting:
                self.sprinting = False
                events.append(GestureEvent(END, "sprint", f.t, 1.0, f.capture_ts))
            events.append(GestureEvent(END, "walk", f.t, 1.0, f.capture_ts))

        if self.walking and len(recent) >= 3:
            intervals = [b - a for a, b in zip(recent, recent[1:])]
            fast = sum(intervals) / len(intervals) < self.SPRINT_INTERVAL / max(self.sens, 0.5)
            if fast and not self.sprinting:
                self.sprinting = True
                events.append(GestureEvent(START, "sprint", f.t, 1.0, f.capture_ts))
            elif not fast and self.sprinting:
                self.sprinting = False
                events.append(GestureEvent(END, "sprint", f.t, 1.0, f.capture_ts))
        return events

    def reset(self) -> None:
        self._knee_up = {LEFT: False, RIGHT: False}
        self._steps.clear()
        self.walking = False
        self.sprinting = False


# --------------------------------------------------------------------------
# Arm gestures (metric world space)
# --------------------------------------------------------------------------

class PunchDetector(PulseDetector):
    priority = 80
    cooldown = 0.45

    def __init__(self, side: str, sens: float = 1.0):
        super().__init__(sens)
        self.side = side
        self.uses_arm = side
        self.name = f"punch_{side}"
        self.s = _SIDE[side]

    def update(self, f: Features) -> list[GestureEvent]:
        if not f.visible(self.s["wrist"], self.s["shoulder"]):
            return []
        wrist, shoulder = f.w(self.s["wrist"]), f.w(self.s["shoulder"])
        vx, _, vz = (float(v) for v in f.v(self.s["wrist"]))
        elbow_angle = f.elbow_angle_l if self.side == LEFT else f.elbow_angle_r
        # depth-dominant: a horizontal swing that drifts toward the camera
        # must not read as a punch
        if (vz < -1.8 / self.sens and -vz >= 1.2 * abs(vx)
                and wrist[2] < shoulder[2] - 0.20 and elbow_angle > 120):
            return self._fire(f, min(1.0, -vz / 3.0))
        return []


class SwingDetector(PulseDetector):
    """Horizontal arm swing at chest height (melee attack)."""
    priority = 50
    cooldown = 0.5

    def __init__(self, side: str, sens: float = 1.0):
        super().__init__(sens)
        self.side = side
        self.uses_arm = side
        self.name = f"swing_{side}_arm"
        self.s = _SIDE[side]

    def update(self, f: Features) -> list[GestureEvent]:
        if not f.visible(self.s["wrist"]):
            return []
        wrist = f.w(self.s["wrist"])
        vx, vy, vz = (float(v) for v in f.v(self.s["wrist"]))
        elbow_angle = f.elbow_angle_l if self.side == LEFT else f.elbow_angle_r
        chest_band = 0.00 < wrist[1] < 0.60
        # horizontally dominant: punches (depth) and chops (vertical) must not
        # read as swings
        if (abs(vx) > 2.2 / self.sens and abs(vx) >= 1.2 * abs(vz)
                and abs(vx) >= abs(vy) and chest_band and elbow_angle > 100):
            return self._fire(f, min(1.0, abs(vx) / 3.5))
        return []


class ChopDetector(PulseDetector):
    """Overhead downward swing (axe chop / hammer / mining)."""
    priority = 65
    cooldown = 0.5

    def __init__(self, side: str, sens: float = 1.0):
        super().__init__(sens)
        self.side = side
        self.uses_arm = side
        self.name = f"chop_{side}"
        self.s = _SIDE[side]
        self._high_until = -1e9

    def update(self, f: Features) -> list[GestureEvent]:
        if not f.visible(self.s["wrist"], self.s["shoulder"]):
            return []
        wrist, shoulder = f.w(self.s["wrist"]), f.w(self.s["shoulder"])
        if wrist[1] > shoulder[1] + 0.08:
            self._high_until = f.t + 0.35
        vx, vy, _ = (float(v) for v in f.v(self.s["wrist"]))
        # downward-dominant AND in front of the body: relaxing a raised arm
        # drops it at your side and must not read as a chop
        if (f.t < self._high_until and vy < -2.4 / self.sens
                and -vy >= abs(vx) and wrist[2] < shoulder[2] - 0.05):
            return self._fire(f, min(1.0, -vy / 4.0))
        return []


class ThrowDetector(PulseDetector):
    """Wind up above/behind the shoulder, then snap forward."""
    priority = 70
    cooldown = 0.8

    def __init__(self, side: str, sens: float = 1.0):
        super().__init__(sens)
        self.side = side
        self.uses_arm = side
        self.name = f"throw_{side}"
        self.s = _SIDE[side]
        self._wound_until = -1e9

    def update(self, f: Features) -> list[GestureEvent]:
        if not f.visible(self.s["wrist"], self.s["shoulder"]):
            return []
        wrist, shoulder = f.w(self.s["wrist"]), f.w(self.s["shoulder"])
        behind = wrist[2] > shoulder[2] + 0.05 and wrist[1] > shoulder[1] - 0.05
        if behind:
            self._wound_until = f.t + 0.45
        vz = float(f.v(self.s["wrist"])[2])
        if f.t < self._wound_until and vz < -2.2 / self.sens:
            return self._fire(f, min(1.0, -vz / 3.5))
        return []


class PushDetector(PulseDetector):
    """Both palms shove forward together."""
    name = "push"
    priority = 90
    cooldown = 0.8

    def update(self, f: Features) -> list[GestureEvent]:
        if not f.visible(P.L_WRIST, P.R_WRIST):
            return []
        lv, rv = float(f.v(P.L_WRIST)[2]), float(f.v(P.R_WRIST)[2])
        lw, rw = f.w(P.L_WRIST), f.w(P.R_WRIST)
        sho_z = float(f.shoulder_mid[2])
        thr = -1.5 / self.sens
        if lv < thr and rv < thr and lw[2] < sho_z - 0.10 and rw[2] < sho_z - 0.10:
            return self._fire(f)
        return []


class ClapDetector(PulseDetector):
    name = "clap"
    priority = 60
    cooldown = 0.6

    def __init__(self, sens: float = 1.0):
        super().__init__(sens)
        self._prev_dist: float | None = None

    def update(self, f: Features) -> list[GestureEvent]:
        if not f.visible(P.L_WRIST, P.R_WRIST):
            self._prev_dist = None
            return []
        d = f.dist(P.L_WRIST, P.R_WRIST)
        closing = 0.0
        if self._prev_dist is not None and f.dt > 0:
            closing = (d - self._prev_dist) / f.dt
        self._prev_dist = d
        if d < 0.14 and closing < -0.9 / self.sens:
            return self._fire(f)
        return []


class WaveDetector(PulseDetector):
    """Hand above shoulder waving side to side."""
    priority = 40
    cooldown = 2.0

    def __init__(self, side: str, sens: float = 1.0):
        super().__init__(sens)
        self.side = side
        self.uses_arm = side
        self.name = "wave"
        self.s = _SIDE[side]
        self._crossings: list[float] = []
        self._last_sign = 0

    def update(self, f: Features) -> list[GestureEvent]:
        wrist, shoulder = f.w(self.s["wrist"]), f.w(self.s["shoulder"])
        if wrist[1] < shoulder[1]:
            self._crossings.clear()
            self._last_sign = 0
            return []
        vx = float(f.v(self.s["wrist"])[0])
        if abs(vx) > 0.8 / self.sens:
            sign = 1 if vx > 0 else -1
            if sign != self._last_sign and self._last_sign != 0:
                self._crossings.append(f.t)
            self._last_sign = sign
        self._crossings = [t for t in self._crossings if f.t - t < 1.2]
        if len(self._crossings) >= 3:
            self._crossings.clear()
            return self._fire(f)
        return []


class HandToMouthDetector(PulseDetector):
    """Bring either hand to the mouth (eat / drink / heal)."""
    name = "hand_to_mouth"
    priority = 55
    cooldown = 1.2
    DWELL = 0.16

    def __init__(self, sens: float = 1.0):
        super().__init__(sens)
        self._near_since: float | None = None
        self._armed = True          # hand must leave the zone before re-firing

    def update(self, f: Features) -> list[GestureEvent]:
        import numpy as np
        mouth = f.mouth
        near = any(
            float(np.linalg.norm(f.w(w) - mouth)) < 0.17 and f.vis[w] > 0.4
            for w in (P.L_WRIST, P.R_WRIST)
        )
        if near:
            if not self._armed:
                return []
            if self._near_since is None:
                self._near_since = f.t
            elif f.t - self._near_since >= self.DWELL:
                self._near_since = None
                self._armed = False
                return self._fire(f)
        else:
            self._near_since = None
            self._armed = True
        return []


class HandToChestDetector(PulseDetector):
    """Tap the chest (reload / inventory / ability)."""
    name = "hand_to_chest"
    priority = 55
    cooldown = 1.2
    DWELL = 0.16

    def __init__(self, sens: float = 1.0):
        super().__init__(sens)
        self._near_since: float | None = None
        self._armed = True          # hand must leave the zone before re-firing

    def update(self, f: Features) -> list[GestureEvent]:
        import numpy as np
        chest, mouth = f.chest, f.mouth
        near = False
        for w in (P.L_WRIST, P.R_WRIST):
            if f.vis[w] < 0.4:
                continue
            wp = f.w(w)
            if (float(np.linalg.norm(wp - chest)) < 0.16
                    and float(np.linalg.norm(wp - mouth)) > 0.20):
                near = True
                break
        if near:
            if not self._armed:
                return []
            if self._near_since is None:
                self._near_since = f.t
            elif f.t - self._near_since >= self.DWELL:
                self._near_since = None
                self._armed = False
                return self._fire(f)
        else:
            self._near_since = None
            self._armed = True
        return []


class RaiseArmDetector(StateDetector):
    enter_dwell = 0.22

    def __init__(self, side: str, sens: float = 1.0):
        super().__init__(sens)
        self.side = side
        self.uses_arm = side
        self.name = f"raise_arm_{side}"
        self.s = _SIDE[side]

    def _enter_condition(self, f: Features) -> bool:
        wrist, shoulder = f.w(self.s["wrist"]), f.w(self.s["shoulder"])
        return f.vis[self.s["wrist"]] > 0.4 and wrist[1] > shoulder[1] + 0.12

    def _exit_condition(self, f: Features) -> bool:
        wrist, shoulder = f.w(self.s["wrist"]), f.w(self.s["shoulder"])
        return wrist[1] < shoulder[1] + 0.05


class ArmsUpDetector(StateDetector):
    """Both arms extended well above the head (clearly higher than hands
    resting ON the head, which is its own gesture)."""
    name = "arms_up"
    enter_dwell = 0.20

    def _enter_condition(self, f: Features) -> bool:
        nose_y = float(f.w(P.NOSE)[1])
        return (f.visible(P.L_WRIST, P.R_WRIST)
                and float(f.w(P.L_WRIST)[1]) > nose_y + 0.20
                and float(f.w(P.R_WRIST)[1]) > nose_y + 0.20)

    def _exit_condition(self, f: Features) -> bool:
        nose_y = float(f.w(P.NOSE)[1])
        return (float(f.w(P.L_WRIST)[1]) < nose_y + 0.10
                or float(f.w(P.R_WRIST)[1]) < nose_y + 0.10)


class BlockDetector(StateDetector):
    """Both forearms raised in front of chest/face."""
    name = "block"
    enter_dwell = 0.12

    def _cond(self, f: Features, y_lo: float, z_off: float, max_elbow: float) -> bool:
        if not f.visible(P.L_WRIST, P.R_WRIST):
            return False
        # crossed wrists are the separate "arms_crossed" gesture, not a block
        if float(f.w(P.L_WRIST)[0]) < float(f.w(P.R_WRIST)[0]):
            return False
        sho_z = float(f.shoulder_mid[2])
        ok = True
        for w, ang in ((P.L_WRIST, f.elbow_angle_l), (P.R_WRIST, f.elbow_angle_r)):
            wp = f.w(w)
            ok &= (wp[1] > y_lo) and (wp[2] < sho_z - z_off) and (ang < max_elbow)
        return bool(ok)

    def _enter_condition(self, f: Features) -> bool:
        return self._cond(f, y_lo=0.22, z_off=0.08, max_elbow=115)

    def _exit_condition(self, f: Features) -> bool:
        return not self._cond(f, y_lo=0.15, z_off=0.04, max_elbow=130)


class BowDrawDetector(StateDetector):
    """One arm extended sideways/forward, other hand pulled to that shoulder."""
    name = "bow_draw"
    enter_dwell = 0.18

    def _pair_ok(self, f: Features, ext: str, pull: str, slack: float) -> bool:
        import numpy as np
        e, p = _SIDE[ext], _SIDE[pull]
        ext_angle = f.elbow_angle_l if ext == LEFT else f.elbow_angle_r
        wrist, shoulder = f.w(e["wrist"]), f.w(e["shoulder"])
        extended = ext_angle > 150 - slack * 20 and abs(float(wrist[1] - shoulder[1])) < 0.20 + slack * 0.05
        pulled = float(np.linalg.norm(f.w(p["wrist"]) - f.w(e["shoulder"]))) < 0.32 + slack * 0.08
        return extended and pulled

    def _enter_condition(self, f: Features) -> bool:
        if not f.visible(P.L_WRIST, P.R_WRIST):
            return False
        return self._pair_ok(f, LEFT, RIGHT, 0.0) or self._pair_ok(f, RIGHT, LEFT, 0.0)

    def _exit_condition(self, f: Features) -> bool:
        return not (self._pair_ok(f, LEFT, RIGHT, 1.0) or self._pair_ok(f, RIGHT, LEFT, 1.0))


class KickDetector(PulseDetector):
    priority = 45
    cooldown = 0.8

    def __init__(self, side: str, sens: float = 1.0):
        super().__init__(sens)
        self.side = side
        self.name = f"kick_{side}"
        self.s = _SIDE[side]
        self.other = _SIDE[RIGHT if side == LEFT else LEFT]

    def update(self, f: Features) -> list[GestureEvent]:
        if not f.visible(self.s["ankle"], self.other["ankle"]):
            return []
        ankle, other_ankle = f.w(self.s["ankle"]), f.w(self.other["ankle"])
        _, vy, vz = (float(v) for v in f.v(self.s["ankle"]))
        raised = float(ankle[1] - other_ankle[1]) > 0.12
        # forward-dominant: a downward stomp must not read as a kick
        if raised and vz < -1.6 / self.sens and -vz >= abs(vy):
            return self._fire(f)
        return []


class TPoseDetector(PulseDetector):
    """Arms straight out to the sides, held ~0.8s. Reserved as the built-in
    pause/resume gesture so the player can always regain control."""
    name = "t_pose"
    priority = 95
    cooldown = 2.5
    DWELL = 0.8

    def __init__(self, sens: float = 1.0):
        super().__init__(sens)
        self._since: float | None = None

    def update(self, f: Features) -> list[GestureEvent]:
        if not f.visible(P.L_WRIST, P.R_WRIST):
            self._since = None
            return []
        lw, rw = f.w(P.L_WRIST), f.w(P.R_WRIST)
        ls, rs = f.w(P.L_SHOULDER), f.w(P.R_SHOULDER)
        level = abs(float(lw[1] - ls[1])) < 0.15 and abs(float(rw[1] - rs[1])) < 0.15
        straight = f.elbow_angle_l > 150 and f.elbow_angle_r > 150
        wide = abs(float(lw[0] - rw[0])) > 0.9
        if level and straight and wide:
            if self._since is None:
                self._since = f.t
            elif f.t - self._since >= self.DWELL:
                self._since = None
                return self._fire(f)
        else:
            self._since = None
        return []


class UppercutDetector(PulseDetector):
    """Upward punch: fist rockets up through the chest with a bent arm."""
    priority = 75
    cooldown = 0.6

    def __init__(self, side: str, sens: float = 1.0):
        super().__init__(sens)
        self.side = side
        self.uses_arm = side
        self.name = f"uppercut_{side}"
        self.s = _SIDE[side]

    def update(self, f: Features) -> list[GestureEvent]:
        if not f.visible(self.s["wrist"], self.s["shoulder"]):
            return []
        wrist = f.w(self.s["wrist"])
        vx, vy, vz = (float(v) for v in f.v(self.s["wrist"]))
        elbow_angle = f.elbow_angle_l if self.side == LEFT else f.elbow_angle_r
        # explosive, upward-dominant, bent arm: casually raising a hand (to
        # the mouth, chest, or head) is far slower than a real uppercut and
        # must never trigger it
        if (vy > 3.0 / self.sens and vy >= 1.3 * abs(vx) and vy >= 1.3 * abs(vz)
                and -0.10 < wrist[1] < 0.55 and elbow_angle < 150):
            return self._fire(f, min(1.0, vy / 4.5))
        return []


class StompDetector(PulseDetector):
    """Raise a foot, then stamp it straight down."""
    priority = 44
    cooldown = 0.7

    def __init__(self, side: str, sens: float = 1.0):
        super().__init__(sens)
        self.side = side
        self.name = f"stomp_{side}"
        self.s = _SIDE[side]
        self.other = _SIDE[RIGHT if side == LEFT else LEFT]
        self._raised_until = -1e9

    def update(self, f: Features) -> list[GestureEvent]:
        if not f.visible(self.s["ankle"], self.other["ankle"]):
            return []
        ankle, other_ankle = f.w(self.s["ankle"]), f.w(self.other["ankle"])
        if float(ankle[1] - other_ankle[1]) > 0.10:
            self._raised_until = f.t + 0.45
        _, vy, vz = (float(v) for v in f.v(self.s["ankle"]))
        # downward-dominant: a forward kick must not read as a stomp
        if f.t < self._raised_until and vy < -2.0 / self.sens and -vy >= 1.2 * abs(vz):
            return self._fire(f, min(1.0, -vy / 3.5))
        return []


class HeadShakeDetector(PulseDetector):
    """Shake (axis='x' -> head_shake, 'no') or nod (axis='y' -> head_nod,
    'yes'): the nose oscillates relative to the shoulders. Avoid mapping
    these while using head-look aim — turning to aim would trigger them."""
    priority = 30
    cooldown = 1.5

    def __init__(self, axis: str, sens: float = 1.0):
        super().__init__(sens)
        self.axis = 0 if axis == "x" else 1
        self.name = "head_shake" if axis == "x" else "head_nod"
        self._threshold = (0.35 if axis == "x" else 0.30)
        self._crossings: list[float] = []
        self._last_sign = 0

    def update(self, f: Features) -> list[GestureEvent]:
        if not f.visible(P.NOSE, P.L_SHOULDER, P.R_SHOULDER):
            return []
        v_rel = float(f.v(P.NOSE)[self.axis]
                      - (f.v(P.L_SHOULDER)[self.axis] + f.v(P.R_SHOULDER)[self.axis]) / 2)
        if abs(v_rel) > self._threshold / self.sens:
            sign = 1 if v_rel > 0 else -1
            if sign != self._last_sign and self._last_sign != 0:
                self._crossings.append(f.t)
            self._last_sign = sign
        self._crossings = [t for t in self._crossings if f.t - t < 1.0]
        if len(self._crossings) >= 3:
            self._crossings.clear()
            self._last_sign = 0
            return self._fire(f)
        return []


class TwoHandSwingDetector(PulseDetector):
    """Both hands together swinging horizontally (bat / axe / golf club)."""
    name = "two_hand_swing"
    priority = 55          # beats single-arm swings when both hands move
    cooldown = 0.6

    def update(self, f: Features) -> list[GestureEvent]:
        if not f.visible(P.L_WRIST, P.R_WRIST):
            return []
        if f.dist(P.L_WRIST, P.R_WRIST) > 0.35:
            return []
        lv, rv = f.v(P.L_WRIST), f.v(P.R_WRIST)
        lvx, rvx = float(lv[0]), float(rv[0])
        same_dir = (lvx > 0) == (rvx > 0)
        fast = min(abs(lvx), abs(rvx)) > 1.8 / self.sens
        dominant = (abs(lvx) >= 1.2 * abs(float(lv[2]))
                    and abs(rvx) >= 1.2 * abs(float(rv[2])))
        mid_y = (float(f.w(P.L_WRIST)[1]) + float(f.w(P.R_WRIST)[1])) / 2
        if same_dir and fast and dominant and 0.0 < mid_y < 0.6:
            return self._fire(f, min(1.0, abs(lvx) / 3.0))
        return []


class ArmsCrossedDetector(StateDetector):
    """Forearms crossed into an X in front of the chest/face."""
    name = "arms_crossed"
    enter_dwell = 0.15

    def _cond(self, f: Features, margin: float) -> bool:
        if not f.visible(P.L_WRIST, P.R_WRIST):
            return False
        lw, rw = f.w(P.L_WRIST), f.w(P.R_WRIST)
        sho_z = float(f.shoulder_mid[2])
        crossed = float(lw[0]) < float(rw[0]) - margin   # left wrist on the right side
        band = 0.10 < float(lw[1]) < 0.65 and 0.10 < float(rw[1]) < 0.65
        front = lw[2] < sho_z - 0.03 and rw[2] < sho_z - 0.03
        return crossed and band and front

    def _enter_condition(self, f: Features) -> bool:
        return self._cond(f, 0.05)

    def _exit_condition(self, f: Features) -> bool:
        return not self._cond(f, 0.0)


class HandsOnHeadDetector(StateDetector):
    """Both hands resting on top of the head."""
    name = "hands_on_head"
    enter_dwell = 0.25

    def _cond(self, f: Features, radius: float) -> bool:
        import numpy as np
        if not f.visible(P.L_WRIST, P.R_WRIST):
            return False
        head_top = (f.w(P.L_EAR) + f.w(P.R_EAR)) / 2
        head_top = head_top + np.array([0.0, 0.10, 0.0], dtype=head_top.dtype)
        return (float(np.linalg.norm(f.w(P.L_WRIST) - head_top)) < radius
                and float(np.linalg.norm(f.w(P.R_WRIST) - head_top)) < radius)

    def _enter_condition(self, f: Features) -> bool:
        return self._cond(f, 0.25)

    def _exit_condition(self, f: Features) -> bool:
        return not self._cond(f, 0.32)


class ClimbDetector(Detector):
    """Alternating overhead reach-and-pull motions -> 'climb' state
    (ladders, walls, swimming-style locomotion)."""
    name = "climb"
    PULL_VY = -1.5

    def __init__(self, sens: float = 1.0):
        super().__init__(sens)
        self._high_until = {LEFT: -1e9, RIGHT: -1e9}
        self._pulls: list[tuple[float, str]] = []
        self.climbing = False

    def update(self, f: Features) -> list[GestureEvent]:
        events: list[GestureEvent] = []
        if not f.visible(P.L_WRIST, P.R_WRIST):
            if self.climbing:
                self.climbing = False
                events.append(GestureEvent(END, "climb", f.t, 1.0, f.capture_ts))
            return events
        nose_y = float(f.w(P.NOSE)[1])
        for side in (LEFT, RIGHT):
            s = _SIDE[side]
            wrist = f.w(s["wrist"])
            if float(wrist[1]) > nose_y + 0.02:
                self._high_until[side] = f.t + 0.8
            vy = float(f.v(s["wrist"])[1])
            below_shoulder = float(wrist[1]) < float(f.w(s["shoulder"])[1])
            if (f.t < self._high_until[side] and below_shoulder
                    and vy < self.PULL_VY / self.sens):
                if not self._pulls or self._pulls[-1][1] != side or f.t - self._pulls[-1][0] > 0.3:
                    self._pulls.append((f.t, side))
                self._high_until[side] = -1e9
        self._pulls = [(t, s) for t, s in self._pulls if f.t - t < 1.6]

        recent_sides = {s for _, s in self._pulls}
        should_climb = len(self._pulls) >= 2 and len(recent_sides) == 2
        if should_climb and not self.climbing:
            self.climbing = True
            events.append(GestureEvent(START, "climb", f.t, 1.0, f.capture_ts))
        elif self.climbing and (not self._pulls or f.t - self._pulls[-1][0] > 1.0):
            self.climbing = False
            events.append(GestureEvent(END, "climb", f.t, 1.0, f.capture_ts))
        return events

    def reset(self) -> None:
        self._high_until = {LEFT: -1e9, RIGHT: -1e9}
        self._pulls.clear()
        self.climbing = False


# --------------------------------------------------------------------------

GESTURE_DESCRIPTIONS: dict[str, str] = {
    "jump_in_place": "Jump straight up",
    "crouch": "Crouch / squat down (held)",
    "lean_left": "Lean torso to your left (held)",
    "lean_right": "Lean torso to your right (held)",
    "lean_forward": "Lean torso toward the screen (held)",
    "lean_back": "Lean torso away from the screen (held)",
    "walk": "March in place (held while stepping)",
    "sprint": "March in place quickly (held)",
    "punch_left": "Punch forward with your left fist",
    "punch_right": "Punch forward with your right fist",
    "swing_left_arm": "Swing your left arm horizontally",
    "swing_right_arm": "Swing your right arm horizontally",
    "chop_left": "Overhead downward chop, left arm",
    "chop_right": "Overhead downward chop, right arm",
    "throw_left": "Throwing motion, left arm",
    "throw_right": "Throwing motion, right arm",
    "push": "Shove both palms forward",
    "clap": "Clap your hands",
    "wave": "Wave a raised hand side to side",
    "hand_to_mouth": "Bring a hand to your mouth",
    "hand_to_chest": "Tap your chest",
    "raise_arm_left": "Hold your left arm up (held)",
    "raise_arm_right": "Hold your right arm up (held)",
    "arms_up": "Both arms above your head (held)",
    "block": "Raise both forearms in front of you (held)",
    "bow_draw": "Extend one arm, pull the other to your shoulder (held)",
    "kick_left": "Front kick with your left leg",
    "kick_right": "Front kick with your right leg",
    "uppercut_left": "Upward punch with your left fist",
    "uppercut_right": "Upward punch with your right fist",
    "stomp_left": "Raise and stamp your left foot down",
    "stomp_right": "Raise and stamp your right foot down",
    "head_nod": "Nod your head (yes-yes-yes)",
    "head_shake": "Shake your head (no-no-no)",
    "two_hand_swing": "Swing both hands together sideways (bat/axe)",
    "arms_crossed": "Cross your forearms into an X (held)",
    "hands_on_head": "Rest both hands on your head (held)",
    "climb": "Alternating overhead reach-and-pull motions (held)",
    "t_pose": "T-pose ~1s (reserved: pause/resume MotionForge)",
}

STATE_GESTURES = {
    "crouch", "lean_left", "lean_right", "lean_forward", "lean_back", "walk",
    "sprint", "raise_arm_left", "raise_arm_right", "arms_up", "block", "bow_draw",
    "arms_crossed", "hands_on_head", "climb",
}


def build_detectors(sens: float = 1.0, accessibility: str = "standing") -> list[Detector]:
    """Instantiate the detector set, filtered for the player's accessibility mode."""
    ds: list[Detector] = [
        JumpDetector(sens), CrouchDetector(sens),
        LeanSideDetector(LEFT, sens), LeanSideDetector(RIGHT, sens),
        LeanForwardDetector(sens), LeanBackDetector(sens),
        WalkInPlaceDetector(sens),
        PunchDetector(LEFT, sens), PunchDetector(RIGHT, sens),
        SwingDetector(LEFT, sens), SwingDetector(RIGHT, sens),
        ChopDetector(LEFT, sens), ChopDetector(RIGHT, sens),
        ThrowDetector(LEFT, sens), ThrowDetector(RIGHT, sens),
        PushDetector(sens), ClapDetector(sens),
        WaveDetector(RIGHT, sens),
        HandToMouthDetector(sens), HandToChestDetector(sens),
        RaiseArmDetector(LEFT, sens), RaiseArmDetector(RIGHT, sens),
        ArmsUpDetector(sens), BlockDetector(sens), BowDrawDetector(sens),
        KickDetector(LEFT, sens), KickDetector(RIGHT, sens),
        UppercutDetector(LEFT, sens), UppercutDetector(RIGHT, sens),
        StompDetector(LEFT, sens), StompDetector(RIGHT, sens),
        HeadShakeDetector("x", sens), HeadShakeDetector("y", sens),
        TwoHandSwingDetector(sens), ArmsCrossedDetector(sens),
        HandsOnHeadDetector(sens), ClimbDetector(sens),
        TPoseDetector(sens),
    ]
    _TWO_HANDED = {"push", "clap", "arms_up", "block", "bow_draw", "t_pose",
                   "two_hand_swing", "arms_crossed", "hands_on_head", "climb"}
    if accessibility == "seated":
        drop = {"jump_in_place", "crouch"}  # hip baseline is unreliable seated
        ds = [d for d in ds if d.name not in drop]
    elif accessibility == "one_handed_left":
        ds = [d for d in ds if d.uses_arm != RIGHT and d.name not in _TWO_HANDED]
    elif accessibility == "one_handed_right":
        ds = [d for d in ds if d.uses_arm != LEFT and d.name not in _TWO_HANDED]
    return ds
