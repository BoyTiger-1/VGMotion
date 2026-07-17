"""Synthetic pose streams for tests: a parametric human skeleton performing
scripted gestures, matching the exact coordinate conventions of vision.pose
(img: y down+, world: meters/hip-origin/y up+/z toward camera negative)."""
from __future__ import annotations

import numpy as np

from motionforge.vision import pose as P
from motionforge.vision.pose import PoseFrame

FPS = 30.0
DT = 1.0 / FPS


def base_pose() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    img = np.full((33, 2), 0.5, dtype=np.float32)
    world = np.zeros((33, 3), dtype=np.float32)
    vis = np.ones(33, dtype=np.float32)

    def set_lm(i, wx, wy, wz, ix=None, iy=None):
        world[i] = (wx, wy, wz)
        if ix is not None:
            img[i] = (ix, iy)

    set_lm(P.NOSE, 0.00, 0.60, -0.05, 0.50, 0.30)
    set_lm(P.L_EAR, +0.07, 0.62, 0.00, 0.53, 0.31)
    set_lm(P.R_EAR, -0.07, 0.62, 0.00, 0.47, 0.31)
    set_lm(P.MOUTH_L, +0.03, 0.52, -0.06, 0.51, 0.33)
    set_lm(P.MOUTH_R, -0.03, 0.52, -0.06, 0.49, 0.33)
    set_lm(P.L_SHOULDER, +0.18, 0.45, 0.00, 0.56, 0.38)
    set_lm(P.R_SHOULDER, -0.18, 0.45, 0.00, 0.44, 0.38)
    set_lm(P.L_ELBOW, +0.25, 0.20, 0.00, 0.59, 0.48)
    set_lm(P.R_ELBOW, -0.25, 0.20, 0.00, 0.41, 0.48)
    set_lm(P.L_WRIST, +0.28, -0.05, 0.00, 0.60, 0.58)
    set_lm(P.R_WRIST, -0.28, -0.05, 0.00, 0.40, 0.58)
    set_lm(P.L_HIP, +0.10, 0.00, 0.00, 0.54, 0.60)
    set_lm(P.R_HIP, -0.10, 0.00, 0.00, 0.46, 0.60)
    set_lm(P.L_KNEE, +0.11, -0.45, 0.02, 0.54, 0.78)
    set_lm(P.R_KNEE, -0.11, -0.45, 0.02, 0.46, 0.78)
    set_lm(P.L_ANKLE, +0.12, -0.85, 0.05, 0.55, 0.95)
    set_lm(P.R_ANKLE, -0.12, -0.85, 0.05, 0.45, 0.95)
    set_lm(P.L_FOOT, +0.12, -0.90, -0.05, 0.55, 0.97)
    set_lm(P.R_FOOT, -0.12, -0.90, -0.05, 0.45, 0.97)
    return img, world, vis


class _Builder:
    def __init__(self):
        self.img0, self.world0, self.vis = base_pose()
        self.frames: list[PoseFrame] = []
        self.t = 0.0

    def emit(self, img=None, world=None):
        self.frames.append(PoseFrame(
            t=self.t,
            img=(self.img0 if img is None else img).copy(),
            world=(self.world0 if world is None else world).copy(),
            vis=self.vis.copy(), present=True))
        self.t += DT

    def hold(self, n: int):
        for _ in range(n):
            self.emit()

    def interpolate(self, n: int, world_targets: dict[int, tuple] = None,
                    img_targets: dict[int, tuple] = None, keep: bool = False):
        """Move landmarks linearly to targets over n frames; optionally keep
        the final position as the new base."""
        w_start = {i: self.world0[i].copy() for i in (world_targets or {})}
        i_start = {i: self.img0[i].copy() for i in (img_targets or {})}
        for k in range(1, n + 1):
            a = k / n
            world = self.world0.copy()
            img = self.img0.copy()
            for i, tgt in (world_targets or {}).items():
                world[i] = (1 - a) * w_start[i] + a * np.array(tgt, dtype=np.float32)
            for i, tgt in (img_targets or {}).items():
                img[i] = (1 - a) * i_start[i] + a * np.array(tgt, dtype=np.float32)
            self.frames.append(PoseFrame(t=self.t, img=img, world=world,
                                         vis=self.vis.copy(), present=True))
            self.t += DT
        if keep:
            for i, tgt in (world_targets or {}).items():
                self.world0[i] = np.array(tgt, dtype=np.float32)
            for i, tgt in (img_targets or {}).items():
                self.img0[i] = np.array(tgt, dtype=np.float32)


def synthetic_stream(scenario: str) -> list[PoseFrame]:
    b = _Builder()
    b.hold(30)  # 1s standing to settle velocities and baseline

    if scenario == "punch_right":
        b.interpolate(5, world_targets={
            P.R_WRIST: (-0.18, 0.45, -0.60),
            P.R_ELBOW: (-0.18, 0.45, -0.30)}, keep=True)
        b.hold(3)

    elif scenario == "jump_in_place":
        dy = -0.16
        img_t = {i: (b.img0[i][0], b.img0[i][1] + dy)
                 for i in (P.L_HIP, P.R_HIP, P.L_SHOULDER, P.R_SHOULDER, P.NOSE)}
        b.interpolate(5, img_targets=img_t)
        b.hold(2)

    elif scenario == "crouch":
        dy = +0.14
        down = {i: (b.img0[i][0], b.img0[i][1] + dy)
                for i in (P.L_HIP, P.R_HIP, P.L_SHOULDER, P.R_SHOULDER, P.NOSE)}
        up = {i: (b.img0[i][0], b.img0[i][1])
              for i in (P.L_HIP, P.R_HIP, P.L_SHOULDER, P.R_SHOULDER, P.NOSE)}
        b.interpolate(4, img_targets=down, keep=True)
        b.hold(15)
        b.interpolate(6, img_targets=up, keep=True)
        b.hold(10)

    elif scenario == "walk":
        for side in (P.R_KNEE, P.L_KNEE, P.R_KNEE, P.L_KNEE):
            x = float(b.world0[side][0])
            b.interpolate(3, world_targets={side: (x, -0.12, 0.02)})
            b.hold(2)
            b.interpolate(3, world_targets={side: (x, -0.45, 0.02)})
            b.hold(3)

    elif scenario == "hand_to_mouth":
        b.interpolate(6, world_targets={
            P.R_WRIST: (-0.03, 0.50, -0.08),
            P.R_ELBOW: (-0.20, 0.20, -0.10)}, keep=True)
        b.hold(12)

    elif scenario == "swing_right_arm":
        b.interpolate(3, world_targets={
            P.R_WRIST: (-0.50, 0.25, -0.20),
            P.R_ELBOW: (-0.35, 0.30, -0.10)}, keep=True)
        b.hold(3)
        b.interpolate(5, world_targets={
            P.R_WRIST: (0.45, 0.30, -0.20),
            P.R_ELBOW: (0.20, 0.32, -0.10)}, keep=True)
        b.hold(3)

    elif scenario == "t_pose":
        b.interpolate(6, world_targets={
            P.L_WRIST: (+0.65, 0.45, 0.0), P.L_ELBOW: (+0.42, 0.45, 0.0),
            P.R_WRIST: (-0.65, 0.45, 0.0), P.R_ELBOW: (-0.42, 0.45, 0.0)}, keep=True)
        b.hold(int(1.2 * FPS))

    elif scenario == "lean_left":
        # player's left = image x+ ; shift shoulders right in image
        img_t = {i: (b.img0[i][0] + 0.06, b.img0[i][1])
                 for i in (P.L_SHOULDER, P.R_SHOULDER, P.NOSE)}
        b.interpolate(5, img_targets=img_t, keep=True)
        b.hold(15)

    elif scenario == "block":
        b.interpolate(5, world_targets={
            P.L_WRIST: (+0.10, 0.42, -0.28), P.L_ELBOW: (+0.22, 0.15, -0.12),
            P.R_WRIST: (-0.10, 0.42, -0.28), P.R_ELBOW: (-0.22, 0.15, -0.12)}, keep=True)
        b.hold(15)

    else:
        raise ValueError(f"unknown scenario {scenario!r}")

    b.hold(10)
    return b.frames
