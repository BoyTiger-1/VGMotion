"""Per-frame feature extraction from raw pose landmarks.

All detectors consume `Features` rather than raw landmarks so gesture math is
written once, in body-relative units:
- world coords: meters, origin at hip center, y UP+, z negative toward camera
- hip_units:    vertical body position in torso-lengths above the calibrated
                standing baseline (jump/crouch signal, camera-distance invariant)
- lean_units:   horizontal shoulder offset from hips in torso-lengths
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from motionforge.vision import pose as P
from motionforge.vision.calibration import Calibration


@dataclass
class Features:
    t: float
    dt: float
    present: bool
    img: np.ndarray            # (33,2)
    world: np.ndarray          # (33,3) y up+
    vis: np.ndarray            # (33,)
    vel: np.ndarray            # (33,3) world velocity m/s, smoothed
    torso_img: float
    hip_units: float           # + = above standing baseline (jumping), - = below (crouching)
    hip_units_vel: float
    lean_units: float          # + = leaning toward person's LEFT (image x+)
    elbow_angle_l: float       # degrees, 180 = straight
    elbow_angle_r: float
    capture_ts: float = 0.0

    # ---- convenience helpers used by detectors ----
    def w(self, i: int) -> np.ndarray:
        return self.world[i]

    def v(self, i: int) -> np.ndarray:
        return self.vel[i]

    def visible(self, *idx: int, thresh: float = 0.4) -> bool:
        return all(self.vis[i] >= thresh for i in idx)

    def dist(self, a: int, b: int) -> float:
        return float(np.linalg.norm(self.world[a] - self.world[b]))

    @property
    def shoulder_mid(self) -> np.ndarray:
        return (self.world[P.L_SHOULDER] + self.world[P.R_SHOULDER]) / 2

    @property
    def hip_mid(self) -> np.ndarray:
        return (self.world[P.L_HIP] + self.world[P.R_HIP]) / 2

    @property
    def mouth(self) -> np.ndarray:
        return (self.world[P.MOUTH_L] + self.world[P.MOUTH_R]) / 2

    @property
    def chest(self) -> np.ndarray:
        c = self.shoulder_mid.copy()
        c[1] -= 0.15
        return c


def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at b (degrees) formed by points a-b-c."""
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 180.0
    cosang = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))


class FeatureExtractor:
    """Stateful: computes velocities and maintains the standing baseline."""

    VEL_ALPHA = 0.65        # smoothing for world velocities (landmarks arrive
                            # pre-filtered by One-Euro, so favor responsiveness)
    BASELINE_TAU = 8.0      # seconds; slow auto-baseline when uncalibrated

    def __init__(self, calibration: Calibration | None = None):
        self.calibration = calibration or Calibration()
        self._prev_world: np.ndarray | None = None
        self._prev_t: float | None = None
        self._vel = np.zeros((33, 3), dtype=np.float32)
        # auto-baseline (used when no explicit calibration)
        self._auto_hip_y = None
        self._auto_torso = None
        self._prev_hip_units = 0.0

    def set_calibration(self, cal: Calibration) -> None:
        self.calibration = cal

    def update(self, pf: P.PoseFrame) -> Features | None:
        if not pf.present:
            self._prev_world = None
            self._prev_t = None
            return None

        dt = 1 / 30
        if self._prev_t is not None:
            dt = max(1e-3, min(0.25, pf.t - self._prev_t))

        if self._prev_world is not None:
            raw_vel = (pf.world - self._prev_world) / dt
            self._vel = self.VEL_ALPHA * raw_vel + (1 - self.VEL_ALPHA) * self._vel
        self._prev_world = pf.world.copy()
        self._prev_t = pf.t

        hip_img = (pf.img[P.L_HIP] + pf.img[P.R_HIP]) / 2
        sho_img = (pf.img[P.L_SHOULDER] + pf.img[P.R_SHOULDER]) / 2
        torso = float(np.linalg.norm(sho_img - hip_img))
        torso = max(torso, 0.02)

        # baseline: explicit calibration wins, else slow-adapting auto baseline
        if self.calibration.valid:
            base_y, base_torso = self.calibration.hip_y_img, self.calibration.torso_img
        else:
            if self._auto_hip_y is None:
                self._auto_hip_y, self._auto_torso = float(hip_img[1]), torso
            else:
                # adapt only while roughly still, so jumps/crouches don't drag it
                speed = float(np.linalg.norm(self._vel[P.L_HIP])) + float(np.linalg.norm(self._vel[P.R_HIP]))
                if speed < 0.6:
                    k = min(1.0, dt / self.BASELINE_TAU)
                    self._auto_hip_y += k * (float(hip_img[1]) - self._auto_hip_y)
                    self._auto_torso += k * (torso - self._auto_torso)
            base_y, base_torso = self._auto_hip_y, self._auto_torso

        norm = max(base_torso, 0.02)
        hip_units = (base_y - float(hip_img[1])) / norm   # image y is down+, so up = +
        hip_units_vel = (hip_units - self._prev_hip_units) / dt
        self._prev_hip_units = hip_units
        lean_units = (float(sho_img[0]) - float(hip_img[0])) / norm

        return Features(
            t=pf.t, dt=dt, present=True,
            img=pf.img, world=pf.world, vis=pf.vis, vel=self._vel.copy(),
            torso_img=torso,
            hip_units=hip_units, hip_units_vel=hip_units_vel, lean_units=lean_units,
            elbow_angle_l=_angle_deg(pf.world[P.L_SHOULDER], pf.world[P.L_ELBOW], pf.world[P.L_WRIST]),
            elbow_angle_r=_angle_deg(pf.world[P.R_SHOULDER], pf.world[P.R_ELBOW], pf.world[P.R_WRIST]),
            capture_ts=pf.t,
        )
