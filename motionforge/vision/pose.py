"""MediaPipe pose estimation wrapper (Tasks API) with automatic performance tuning.

Coordinate conventions produced here (used by the whole gesture engine):
- img:   (33, 2) normalized image coords, x right+, y DOWN+ (raw MediaPipe)
- world: (33, 3) metric meters, origin at hip center, y flipped to UP+,
         z negative = toward the camera (raw MediaPipe convention)
- vis:   (33,) landmark visibility 0..1
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("GLOG_minloglevel", "2")

# MediaPipe landmark indices used across the codebase
NOSE = 0
L_EAR, R_EAR = 7, 8
MOUTH_L, MOUTH_R = 9, 10
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_FOOT, R_FOOT = 31, 32

SKELETON_EDGES = [
    (L_SHOULDER, R_SHOULDER), (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),
    (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST), (L_SHOULDER, L_HIP),
    (R_SHOULDER, R_HIP), (L_HIP, R_HIP), (L_HIP, L_KNEE), (L_KNEE, L_ANKLE),
    (R_HIP, R_KNEE), (R_KNEE, R_ANKLE), (NOSE, L_SHOULDER), (NOSE, R_SHOULDER),
]

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_FILES = {
    0: "pose_landmarker_lite.task",
    1: "pose_landmarker_full.task",
    2: "pose_landmarker_heavy.task",
}
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_{name}/float16/latest/pose_landmarker_{name}.task")


def model_path(complexity: int) -> Path:
    return MODELS_DIR / MODEL_FILES[complexity]


def ensure_model(complexity: int) -> Path:
    """Return the model path, downloading it on first use if missing."""
    path = model_path(complexity)
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    name = {0: "lite", 1: "full", 2: "heavy"}[complexity]
    url = MODEL_URL.format(name=name)
    import requests
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    path.write_bytes(resp.content)
    return path


@dataclass
class PoseFrame:
    t: float                      # capture timestamp (perf_counter seconds)
    img: np.ndarray               # (33,2) normalized image coords
    world: np.ndarray             # (33,3) meters, y up+
    vis: np.ndarray               # (33,)
    frame_bgr: object = None      # original frame for UI overlay (may be None)
    infer_ms: float = 0.0
    present: bool = True          # was a person detected


def _empty_frame(ts: float, frame_bgr, infer_ms: float) -> PoseFrame:
    z = np.zeros((33, 3), dtype=np.float32)
    return PoseFrame(t=ts, img=z[:, :2].copy(), world=z,
                     vis=np.zeros(33, dtype=np.float32),
                     frame_bgr=frame_bgr, infer_ms=infer_ms, present=False)


class PoseEstimator:
    """Wraps the MediaPipe PoseLandmarker with runtime model switching."""

    def __init__(self, model_complexity: int = 1, auto_performance: bool = True):
        self.model_complexity = model_complexity
        self.auto_performance = auto_performance
        self._landmarker = None
        self._infer_ema = 0.0
        self._frames = 0
        self._ts_ms = 0
        self._build()

    def _build(self) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            PoseLandmarker, PoseLandmarkerOptions, RunningMode)

        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model(self.model_complexity))),
            running_mode=RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)
        self._mp = mp

    def process(self, frame_bgr, ts: float) -> PoseFrame:
        import cv2
        t0 = time.perf_counter()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        # VIDEO mode requires strictly increasing integer ms timestamps
        ts_ms = int(ts * 1000)
        if ts_ms <= self._ts_ms:
            ts_ms = self._ts_ms + 1
        self._ts_ms = ts_ms
        result = self._landmarker.detect_for_video(mp_image, ts_ms)
        infer_ms = (time.perf_counter() - t0) * 1000.0
        self._infer_ema = 0.9 * self._infer_ema + 0.1 * infer_ms if self._infer_ema else infer_ms
        self._frames += 1
        if self.auto_performance and self._frames % 90 == 0:
            self._autotune()

        if not result.pose_landmarks:
            return _empty_frame(ts, frame_bgr, infer_ms)

        lm = result.pose_landmarks[0]
        img = np.array([[p.x, p.y] for p in lm], dtype=np.float32)
        vis = np.array([(p.visibility if p.visibility else 0.9) for p in lm], dtype=np.float32)
        if result.pose_world_landmarks:
            wl = result.pose_world_landmarks[0]
            world = np.array([[p.x, -p.y, p.z] for p in wl], dtype=np.float32)  # flip y -> up+
        else:
            world = np.zeros((33, 3), dtype=np.float32)
        return PoseFrame(t=ts, img=img, world=world, vis=vis,
                         frame_bgr=frame_bgr, infer_ms=infer_ms, present=True)

    def set_complexity(self, complexity: int) -> None:
        complexity = max(0, min(2, int(complexity)))
        if complexity != self.model_complexity:
            self.model_complexity = complexity
            self._build()

    def _autotune(self) -> None:
        """Keep inference under ~22ms: drop model size when slow, raise when fast."""
        if self._infer_ema > 26 and self.model_complexity > 0:
            self.set_complexity(self.model_complexity - 1)
        elif self._infer_ema < 11 and self.model_complexity < 1:
            # auto mode caps at 'full' (1); 'heavy' (2) only via explicit setting
            self.set_complexity(self.model_complexity + 1)

    @property
    def infer_ms(self) -> float:
        return self._infer_ema

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
