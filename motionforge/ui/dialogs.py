"""Dialogs: calibration wizard, keybind capture wizard, AI suggestion review."""
from __future__ import annotations

import threading

from PySide6.QtCore import Qt, QObject, Signal
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QProgressBar, QPushButton, QVBoxLayout)

from motionforge.controls.capture import CAPTURE_ACTIONS, wait_for_key
from motionforge.gestures.library import GESTURE_DESCRIPTIONS
from motionforge.vision.calibration import CalibrationSession


class CalibrationDialog(QDialog):
    """Stand/sit still for a few seconds while we learn your body baseline."""

    def __init__(self, engine, bridge, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Calibration — MotionForge")
        self.setMinimumWidth(420)
        self.engine = engine
        self.bridge = bridge
        self.session: CalibrationSession | None = None

        lay = QVBoxLayout(self)
        self.info = QLabel(
            "<b>Player calibration</b><br><br>"
            "1. Step back so your head and hips are visible.<br>"
            "2. Stand (or sit) relaxed, arms at your sides.<br>"
            "3. Press <b>Start</b> and hold still for 3 seconds.")
        self.info.setWordWrap(True)
        lay.addWidget(self.info)
        self.seated = QCheckBox("I'm playing seated")
        lay.addWidget(self.seated)
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        lay.addWidget(self.bar)
        row = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self._start)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        row.addWidget(self.start_btn)
        row.addWidget(close_btn)
        lay.addLayout(row)
        self.bridge.pose.connect(self._on_pose)

    def _start(self):
        self.session = CalibrationSession(3.0, self.seated.isChecked())
        self.start_btn.setEnabled(False)
        self.info.setText("Hold still…")

    def _on_pose(self, pf, feats, states):
        if self.session is None:
            return
        progress = self.session.add(pf)
        self.bar.setValue(int(progress * 100))
        if progress >= 1.0:
            cal = self.session.result()
            self.session = None
            self.engine.apply_calibration(cal)
            if cal.valid:
                self.info.setText("<b>Calibration saved.</b> You're ready to play.")
            else:
                self.info.setText("Could not see you clearly — adjust the camera and retry.")
                self.start_btn.setEnabled(True)
                return
            self.start_btn.setEnabled(True)

    def closeEvent(self, e):
        try:
            self.bridge.pose.disconnect(self._on_pose)
        except (RuntimeError, TypeError):
            pass
        super().closeEvent(e)


class _CaptureWorker(QObject):
    captured = Signal(str, str)     # semantic, binding
    finished = Signal()
    prompt = Signal(str, str)       # semantic, label

    def __init__(self, actions):
        super().__init__()
        self.actions = actions
        self.cancel_current = threading.Event()
        self.stop_all = threading.Event()

    def run(self):
        for semantic, label in self.actions:
            if self.stop_all.is_set():
                break
            self.cancel_current.clear()
            self.prompt.emit(semantic, label)
            binding = wait_for_key(timeout=20.0, cancel=self.cancel_current)
            if self.stop_all.is_set():
                break
            if binding:
                self.captured.emit(semantic, binding)
        self.finished.emit()


class KeybindCaptureDialog(QDialog):
    """Walks the player through pressing each important control once, so we
    can build a profile for games whose configs can't be discovered."""

    def __init__(self, profile, profiles, engine, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Capture keybinds — MotionForge")
        self.setMinimumWidth(460)
        self.profile = profile
        self.profiles = profiles
        self.engine = engine

        lay = QVBoxLayout(self)
        self.head = QLabel("<b>Keybind capture</b> — press the key this game uses for each action. "
                           "Press <b>Skip</b> for actions the game doesn't have.")
        self.head.setWordWrap(True)
        lay.addWidget(self.head)
        self.current = QLabel("")
        self.current.setStyleSheet("font-size:17px; font-weight:700; color:#4fc3f7;")
        self.current.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.current)
        self.log = QListWidget()
        lay.addWidget(self.log)
        row = QHBoxLayout()
        self.skip_btn = QPushButton("Skip")
        self.done_btn = QPushButton("Finish")
        row.addWidget(self.skip_btn)
        row.addWidget(self.done_btn)
        lay.addLayout(row)

        self.worker = _CaptureWorker(CAPTURE_ACTIONS)
        self.worker.prompt.connect(self._on_prompt)
        self.worker.captured.connect(self._on_captured)
        self.worker.finished.connect(self._on_finished)
        self.skip_btn.clicked.connect(lambda: self.worker.cancel_current.set())
        self.done_btn.clicked.connect(self._finish_now)
        self._thread = threading.Thread(target=self.worker.run, daemon=True)
        self._thread.start()

    def _on_prompt(self, semantic, label):
        self.current.setText(f"Press the key for: {label}")

    def _on_captured(self, semantic, binding):
        from motionforge.ai.offline import DEFAULT_INPUTS
        _, mode = DEFAULT_INPUTS.get(semantic, ("none", "tap"))
        existing = self.profile.actions.get(semantic, {})
        self.profile.actions[semantic] = {
            "input": binding, "mode": existing.get("mode", mode),
            "hold_ms": existing.get("hold_ms", 600)}
        self.log.addItem(QListWidgetItem(f"{semantic}  →  {binding}"))

    def _finish_now(self):
        self.worker.stop_all.set()
        self.worker.cancel_current.set()

    def _on_finished(self):
        self.profiles.save(self.profile)
        self.engine.set_profile(self.profile)
        self.current.setText("Saved. Gesture mappings now use your real keybinds.")
        self.done_btn.setText("Close")
        self.done_btn.clicked.disconnect()
        self.done_btn.clicked.connect(self.accept)

    def closeEvent(self, e):
        self.worker.stop_all.set()
        self.worker.cancel_current.set()
        super().closeEvent(e)


class SuggestionDialog(QDialog):
    """Shows AI/offline gesture suggestions with rationale; user applies or rejects."""

    def __init__(self, gestures: dict, rationale: dict, source: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Gesture suggestions ({source}) — MotionForge")
        self.setMinimumSize(560, 420)
        self.accepted_mappings = None

        lay = QVBoxLayout(self)
        origin = ("Generated by Gemini AI for this game"
                  if source == "ai" else "Generated by the built-in heuristic engine")
        head = QLabel(f"<b>Suggested motion controls</b> — {origin}. "
                      "Review the reasoning, then apply or cancel.")
        head.setWordWrap(True)
        lay.addWidget(head)
        self.list = QListWidget()
        self.list.setWordWrap(True)
        for gesture, semantic in gestures.items():
            desc = GESTURE_DESCRIPTIONS.get(gesture, gesture)
            why = rationale.get(gesture, "")
            item = QListWidgetItem(f"{desc}\n    →  {semantic.upper()}    {('— ' + why) if why else ''}")
            self.list.addItem(item)
        lay.addWidget(self.list)
        buttons = QDialogButtonBox(QDialogButtonBox.Apply | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Apply).clicked.connect(
            lambda: (setattr(self, "accepted_mappings", (gestures, rationale)), self.accept()))
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)
