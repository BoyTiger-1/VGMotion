"""Player calibration: capture a neutral standing (or seated) baseline used to
normalize all gesture math to this player's body and camera placement."""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from motionforge.vision import pose as P


@dataclass
class Calibration:
    hip_y_img: float = 0.6        # baseline hip height in image coords (y down+)
    torso_img: float = 0.22       # shoulder-mid to hip-mid distance in image units
    center_x_img: float = 0.5     # neutral horizontal position
    nose_y_img: float = 0.3
    arm_span_m: float = 1.6       # from world landmarks, wrist-to-wrist in T pose-ish
    height_units: float = 3.0     # nose-to-ankle in torso units
    seated: bool = False
    valid: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Calibration":
        c = cls()
        for k, v in (d or {}).items():
            if hasattr(c, k):
                setattr(c, k, v)
        return c


class CalibrationSession:
    """Accumulates pose frames while the player holds still, then produces a
    Calibration. Used by the setup wizard; the engine also keeps a slow
    auto-baseline as a fallback when the user skips calibration."""

    def __init__(self, duration_s: float = 3.0, seated: bool = False):
        self.duration_s = duration_s
        self.seated = seated
        self._samples: list[P.PoseFrame] = []
        self._t0: float | None = None

    def add(self, pf: P.PoseFrame) -> float:
        """Feed a frame; returns progress 0..1."""
        if not pf.present:
            return self.progress
        if self._t0 is None:
            self._t0 = pf.t
        self._samples.append(pf)
        return self.progress

    @property
    def progress(self) -> float:
        if self._t0 is None or not self._samples:
            return 0.0
        return min(1.0, (self._samples[-1].t - self._t0) / self.duration_s)

    def result(self) -> Calibration:
        if len(self._samples) < 5:
            return Calibration(valid=False, seated=self.seated)
        imgs = np.stack([s.img for s in self._samples])
        worlds = np.stack([s.world for s in self._samples])
        hip = (imgs[:, P.L_HIP] + imgs[:, P.R_HIP]) / 2
        sho = (imgs[:, P.L_SHOULDER] + imgs[:, P.R_SHOULDER]) / 2
        torso = float(np.median(np.linalg.norm(sho - hip, axis=1)))
        nose_y = float(np.median(imgs[:, P.NOSE, 1]))
        ankle_y = float(np.median((imgs[:, P.L_ANKLE, 1] + imgs[:, P.R_ANKLE, 1]) / 2))
        span = float(np.median(np.linalg.norm(worlds[:, P.L_WRIST] - worlds[:, P.R_WRIST], axis=1)))
        height_units = abs(ankle_y - nose_y) / torso if torso > 1e-4 else 3.0
        return Calibration(
            hip_y_img=float(np.median(hip[:, 1])),
            torso_img=max(torso, 0.02),
            center_x_img=float(np.median(hip[:, 0])),
            nose_y_img=nose_y,
            arm_span_m=max(span, 0.5),
            height_units=height_units,
            seated=self.seated,
            valid=True,
        )
