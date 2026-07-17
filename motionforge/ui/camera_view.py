"""Live camera widget with skeleton overlay, active-state badges, and gesture
flash notifications. The preview is mirrored (like a mirror) because that is
what players expect when watching themselves."""
from __future__ import annotations

import time

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QLabel

from motionforge.vision.pose import SKELETON_EDGES

_JOINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


class CameraView(QLabel):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(480, 360)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background:#0a0d10; border:1px solid #2a3138; border-radius:8px;")
        self.setText("Camera starting…")
        self._flash: list[tuple[str, float]] = []   # (gesture label, expiry)
        self._states: list[str] = []
        self._armed = False

    def flash_gesture(self, label: str) -> None:
        self._flash.append((label, time.perf_counter() + 1.4))

    def set_states(self, states: list[str]) -> None:
        self._states = states

    def set_armed(self, armed: bool) -> None:
        self._armed = armed

    def update_frame(self, pose_frame) -> None:
        frame = pose_frame.frame_bgr
        if frame is None:
            return
        h, w = frame.shape[:2]
        rgb = np.ascontiguousarray(frame[:, :, ::-1])
        img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()

        painter = QPainter(img)
        painter.setRenderHint(QPainter.Antialiasing)
        if pose_frame.present:
            pts = pose_frame.img
            vis = pose_frame.vis
            pen = QPen(QColor("#4fc3f7"), 3)
            painter.setPen(pen)
            for a, b in SKELETON_EDGES:
                if vis[a] > 0.4 and vis[b] > 0.4:
                    painter.drawLine(int(pts[a][0] * w), int(pts[a][1] * h),
                                     int(pts[b][0] * w), int(pts[b][1] * h))
            painter.setPen(Qt.NoPen)
            for j in _JOINTS:
                if vis[j] > 0.4:
                    # joint color shows per-landmark confidence at a glance
                    painter.setBrush(QColor("#69f0ae") if vis[j] > 0.7 else QColor("#ffd740"))
                    painter.drawEllipse(int(pts[j][0] * w) - 4, int(pts[j][1] * h) - 4, 8, 8)
        painter.end()

        img = img.mirrored(True, False)  # mirror AFTER drawing so overlay tracks the body

        painter = QPainter(img)
        painter.setRenderHint(QPainter.Antialiasing)
        # armed indicator
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#69f0ae") if self._armed else QColor("#ff5252"))
        painter.drawEllipse(12, 12, 14, 14)
        painter.setPen(QColor("#e8eaed"))
        painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
        painter.drawText(34, 24, "ARMED" if self._armed else "PAUSED")
        # tracking-quality badge from core-landmark confidence
        if pose_frame.present:
            core = float(pose_frame.vis[[11, 12, 23, 24]].mean())
            if core > 0.75:
                quality, qcolor = "TRACKING: GOOD", "#69f0ae"
            elif core > 0.45:
                quality, qcolor = "TRACKING: PARTIAL", "#ffd740"
            else:
                quality, qcolor = "TRACKING: POOR", "#ff8a80"
        else:
            quality, qcolor = "NO PERSON DETECTED", "#ff8a80"
        painter.setPen(QColor(qcolor))
        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        painter.drawText(w - 190, 24, quality)
        # active held states
        painter.setFont(QFont("Segoe UI", 9))
        painter.setPen(QColor("#ffd740"))
        for i, st in enumerate(self._states[:5]):
            painter.drawText(12, 48 + 18 * i, f"● {st}")
        # gesture flashes
        now = time.perf_counter()
        self._flash = [(g, t) for g, t in self._flash if t > now]
        painter.setFont(QFont("Segoe UI", 16, QFont.Bold))
        for i, (label, _) in enumerate(self._flash[-3:]):
            painter.setPen(QColor("#4fc3f7"))
            painter.drawText(0, h - 60 - 32 * i, w - 16, 30,
                             Qt.AlignRight | Qt.AlignVCenter, label)
        painter.end()

        self.setPixmap(QPixmap.fromImage(img).scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
