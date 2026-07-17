"""In-game overlay HUD.

A small frameless, always-on-top, fully click-through panel that floats over
the game (any borderless/windowed game; exclusive-fullscreen games hide all
overlays — use borderless mode or the sound cues). Shows at a glance:
armed state, detected game, live mini-skeleton (am I in frame?), held
states, the last few gestures with sent/blocked status, and latency.
"""
from __future__ import annotations

import time

from PySide6.QtCore import Qt, QTimer, QRectF
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QApplication, QWidget

from motionforge.core.events import PULSE, START
from motionforge.vision.pose import SKELETON_EDGES

PANEL_W, PANEL_H = 264, 190
MARGIN = 18
FADE_S = 3.5          # gesture feed entries fade out over this long
FLASH_S = 0.45        # border flash after an injected action


class OverlayWindow(QWidget):
    def __init__(self, engine, bridge):
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
                         | Qt.Tool | Qt.WindowTransparentForInput
                         | Qt.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setFixedSize(PANEL_W, PANEL_H)
        self.engine = engine

        self._armed = False
        self._game = "No game detected"
        self._states: list[str] = []
        self._feed: list[tuple[float, str, bool]] = []   # (time, text, injected)
        self._latency = 0.0
        self._fps = 0.0
        self._pose_pts = None
        self._pose_vis = None
        self._flash_until = 0.0

        bridge.active.connect(self._on_active)
        bridge.game.connect(self._on_game)
        bridge.gesture.connect(self._on_gesture)
        bridge.stats.connect(self._on_stats)
        bridge.pose.connect(self._on_pose)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(100)          # 10 Hz repaint is plenty for a HUD
        self.reposition()

    # ------------------------------------------------------------------ slots

    def _on_active(self, armed: bool):
        self._armed = armed

    def _on_game(self, info, profile):
        self._game = info.name if info.is_game else "No game detected"

    def _on_gesture(self, ev, semantic, input_str, injected):
        if ev.kind not in (PULSE, START):
            return
        if semantic:
            text = f"{ev.name} → {semantic}" + ("" if injected else "  (blocked)")
        else:
            text = f"{ev.name}"
            injected = False
        self._feed.append((time.perf_counter(), text, injected))
        del self._feed[:-4]
        if injected:
            self._flash_until = time.perf_counter() + FLASH_S

    def _on_stats(self, st):
        self._latency = st.pipeline_ms
        self._fps = st.camera_fps

    def _on_pose(self, pf, feats, states):
        self._states = states
        if pf.present:
            self._pose_pts = pf.img
            self._pose_vis = pf.vis
        else:
            self._pose_pts = None

    # ------------------------------------------------------------------ layout

    def reposition(self, corner: str | None = None):
        corner = corner or self.engine.settings.get("overlay_corner", "top_right")
        geo = QApplication.primaryScreen().availableGeometry()
        x = geo.left() + MARGIN if "left" in corner else geo.right() - PANEL_W - MARGIN
        y = geo.top() + MARGIN if "top" in corner else geo.bottom() - PANEL_H - MARGIN
        self.move(x, y)

    # ------------------------------------------------------------------ paint

    def paintEvent(self, _):
        now = time.perf_counter()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        # panel
        flash = now < self._flash_until
        border = QColor("#69f0ae") if flash else (
            QColor("#2e7d32") if self._armed else QColor("#5c3030"))
        p.setPen(QPen(border, 3 if flash else 2))
        p.setBrush(QColor(14, 18, 23, 215))
        p.drawRoundedRect(QRectF(1, 1, PANEL_W - 2, PANEL_H - 2), 10, 10)

        # row 1: armed dot + label + latency
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#69f0ae") if self._armed else QColor("#ff5252"))
        p.drawEllipse(12, 12, 12, 12)
        p.setPen(QColor("#e8eaed"))
        p.setFont(QFont("Segoe UI", 10, QFont.Bold))
        p.drawText(31, 23, "ARMED" if self._armed else "PAUSED (F9)")
        p.setPen(QColor("#9fb3c8"))
        p.setFont(QFont("Segoe UI", 8))
        p.drawText(QRectF(0, 10, PANEL_W - 12, 14), Qt.AlignRight,
                   f"{self._latency:.0f} ms · {self._fps:.0f} fps")

        # row 2: game
        p.setPen(QColor("#4fc3f7"))
        p.setFont(QFont("Segoe UI", 9, QFont.Bold))
        p.drawText(QRectF(12, 28, PANEL_W - 100, 16), Qt.AlignLeft,
                   p.fontMetrics().elidedText(self._game, Qt.ElideRight, PANEL_W - 104))

        # mini skeleton box (right side): instant "am I in frame?" check
        box = QRectF(PANEL_W - 86, 30, 74, 96)
        p.setPen(QPen(QColor(70, 90, 105, 140), 1))
        p.setBrush(QColor(8, 11, 14, 160))
        p.drawRoundedRect(box, 6, 6)
        if self._pose_pts is not None:
            pts = self._pose_pts
            vis = self._pose_vis
            p.setPen(QPen(QColor("#4fc3f7"), 2))
            for a, b in SKELETON_EDGES:
                if vis[a] > 0.4 and vis[b] > 0.4:
                    # mirrored, like the main preview
                    ax = box.right() - float(pts[a][0]) * box.width()
                    bx = box.right() - float(pts[b][0]) * box.width()
                    p.drawLine(int(ax), int(box.top() + float(pts[a][1]) * box.height()),
                               int(bx), int(box.top() + float(pts[b][1]) * box.height()))
        else:
            p.setPen(QColor("#ff8a80"))
            p.setFont(QFont("Segoe UI", 7))
            p.drawText(box, Qt.AlignCenter, "not\nin frame")

        # held states
        p.setFont(QFont("Segoe UI", 8))
        p.setPen(QColor("#ffd740"))
        states = "  ".join(f"●{s}" for s in self._states[:4])
        p.drawText(QRectF(12, 46, PANEL_W - 100, 14), Qt.AlignLeft, states)

        # gesture feed with fade-out
        y = 66
        p.setFont(QFont("Segoe UI", 9))
        for t, text, injected in reversed(self._feed):
            age = now - t
            if age > FADE_S:
                continue
            alpha = int(255 * max(0.0, 1.0 - age / FADE_S))
            color = QColor("#69f0ae") if injected else QColor("#ff8a80")
            color.setAlpha(alpha)
            p.setPen(color)
            mark = "✓ " if injected else "✗ "
            p.drawText(QRectF(12, y, PANEL_W - 100, 16), Qt.AlignLeft,
                       p.fontMetrics().elidedText(mark + text, Qt.ElideRight, PANEL_W - 104))
            y += 17
            if y > PANEL_H - 20:
                break
        p.end()
