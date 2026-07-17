"""MotionForge main window: dashboard, mapping editor, discovery, settings."""
from __future__ import annotations

import threading

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout,
    QInputDialog, QLabel, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
    QPushButton, QSlider, QSpinBox, QTableWidget, QTableWidgetItem, QTabWidget,
    QVBoxLayout, QWidget)

from motionforge import __app_name__, __version__
from motionforge.ai.offline import SEMANTIC_ACTIONS
from motionforge.core.events import PULSE, START
from motionforge.gestures.library import GESTURE_DESCRIPTIONS, STATE_GESTURES
from motionforge.ui.camera_view import CameraView
from motionforge.ui.dialogs import CalibrationDialog, KeybindCaptureDialog, SuggestionDialog
from motionforge.ui.style import QSS


class EngineBridge(QObject):
    """Marshals engine callbacks (worker threads) onto the Qt main thread."""
    pose = Signal(object, object, object)
    gesture = Signal(object, object, object, bool)
    stats = Signal(object)
    game = Signal(object, object)
    status = Signal(str)
    active = Signal(bool)

    def __init__(self, engine):
        super().__init__()
        engine.on_pose.subscribe(lambda *a: self.pose.emit(*a))
        engine.on_gesture.subscribe(lambda *a: self.gesture.emit(*a))
        engine.on_stats.subscribe(lambda *a: self.stats.emit(*a))
        engine.on_game.subscribe(lambda *a: self.game.emit(*a))
        engine.on_status.subscribe(lambda *a: self.status.emit(*a))
        engine.on_active.subscribe(lambda *a: self.active.emit(*a))


def _stat(title: str) -> tuple[QWidget, QLabel]:
    box = QWidget()
    lay = QVBoxLayout(box)
    lay.setContentsMargins(10, 4, 10, 4)
    t = QLabel(title)
    t.setObjectName("statTitle")
    v = QLabel("—")
    v.setObjectName("statValue")
    lay.addWidget(t)
    lay.addWidget(v)
    return box, v


class MainWindow(QMainWindow):
    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self.bridge = EngineBridge(engine)
        self.setWindowTitle(f"{__app_name__} {__version__} — Universal AI Motion Controls")
        self.setStyleSheet(QSS)
        self.resize(1280, 800)
        self._pose_counter = 0
        self._build()
        self._connect()

    # ------------------------------------------------------------------ layout

    def _build(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # header
        header = QHBoxLayout()
        title = QLabel("⚡ MOTIONFORGE")
        title.setObjectName("appTitle")
        header.addWidget(title)
        header.addStretch(1)
        self.game_label = QLabel("No game detected")
        self.game_label.setObjectName("gameName")
        header.addWidget(self.game_label)
        header.addStretch(1)
        self.arm_btn = QPushButton("ARM  (F9)")
        self.arm_btn.setObjectName("armButton")
        self.arm_btn.setCheckable(True)
        header.addWidget(self.arm_btn)
        root.addLayout(header)

        body = QHBoxLayout()
        root.addLayout(body, 1)

        # left column: camera + stats + gesture feed
        left = QVBoxLayout()
        self.camera = CameraView()
        left.addWidget(self.camera, 3)
        stats_row = QHBoxLayout()
        for key, label in (("fps", "CAMERA FPS"), ("pose", "POSE MS"),
                           ("lat", "LATENCY MS"), ("p95", "P95 MS"), ("model", "MODEL")):
            box, value = _stat(label)
            setattr(self, f"stat_{key}", value)
            stats_row.addWidget(box)
        left.addLayout(stats_row)
        feed_box = QGroupBox("Recognized gestures")
        feed_lay = QVBoxLayout(feed_box)
        self.feed = QListWidget()
        feed_lay.addWidget(self.feed)
        left.addWidget(feed_box, 2)
        body.addLayout(left, 5)

        # right column: tabs
        self.tabs = QTabWidget()
        self.tabs.addTab(self._mappings_tab(), "Mappings")
        self.tabs.addTab(self._discovery_tab(), "Discovery")
        self.tabs.addTab(self._settings_tab(), "Settings")
        self.tabs.addTab(self._ai_tab(), "AI")
        body.addWidget(self.tabs, 5)

        self.statusBar().showMessage("Welcome to MotionForge. Hold a T-pose or press F9 to arm.")

    def _mappings_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        prof_row = QHBoxLayout()
        prof_row.addWidget(QLabel("Profile:"))
        self.variant_combo = QComboBox()
        prof_row.addWidget(self.variant_combo, 1)
        self.new_variant_btn = QPushButton("New variant")
        prof_row.addWidget(self.new_variant_btn)
        lay.addLayout(prof_row)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Gesture", "Action", "Input", "Mode"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnWidth(0, 220)
        self.table.setColumnWidth(1, 130)
        self.table.setColumnWidth(2, 110)
        lay.addWidget(self.table, 1)

        btns = QHBoxLayout()
        self.add_row_btn = QPushButton("Add mapping")
        self.del_row_btn = QPushButton("Remove selected")
        self.suggest_btn = QPushButton("✨ AI suggest")
        self.save_btn = QPushButton("Save profile")
        for b in (self.add_row_btn, self.del_row_btn, self.suggest_btn, self.save_btn):
            btns.addWidget(b)
        lay.addLayout(btns)
        return w

    def _discovery_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(QLabel("Keybinds discovered from the game's config files:"))
        self.binds_table = QTableWidget(0, 2)
        self.binds_table.setHorizontalHeaderLabels(["Game action", "Key"])
        self.binds_table.horizontalHeader().setStretchLastSection(True)
        lay.addWidget(self.binds_table, 1)
        row = QHBoxLayout()
        self.scan_btn = QPushButton("Scan config files now")
        self.capture_btn = QPushButton("Capture keybinds manually…")
        row.addWidget(self.scan_btn)
        row.addWidget(self.capture_btn)
        lay.addLayout(row)
        return w

    def _settings_tab(self) -> QWidget:
        w = QWidget()
        s = self.engine.settings
        form = QFormLayout(w)

        self.camera_spin = QSpinBox()
        self.camera_spin.setRange(0, 8)
        self.camera_spin.setValue((s.get("camera_indices") or [0])[0])
        form.addRow("Camera index", self.camera_spin)

        self.look_combo = QComboBox()
        self.look_combo.addItems(["head", "lean", "right_hand", "left_hand", "cursor_hand", "off"])
        self.look_combo.setCurrentText(s.get("look_mode"))
        form.addRow("Look / aim mode", self.look_combo)

        self.look_sens = QDoubleSpinBox()
        self.look_sens.setRange(0.1, 5.0)
        self.look_sens.setSingleStep(0.1)
        self.look_sens.setValue(float(s.get("look_sensitivity")))
        form.addRow("Look sensitivity", self.look_sens)

        self.gest_sens = QDoubleSpinBox()
        self.gest_sens.setRange(0.25, 3.0)
        self.gest_sens.setSingleStep(0.05)
        self.gest_sens.setValue(float(s.get("gesture_sensitivity")))
        form.addRow("Gesture sensitivity", self.gest_sens)

        self.access_combo = QComboBox()
        self.access_combo.addItems(["standing", "seated", "one_handed_left", "one_handed_right"])
        self.access_combo.setCurrentText(s.get("accessibility"))
        form.addRow("Play style / accessibility", self.access_combo)

        self.model_combo = QComboBox()
        self.model_combo.addItems(["auto", "0 (lite/fastest)", "1 (full)", "2 (heavy/accurate)"])
        self.model_combo.setCurrentIndex(0 if s.get("auto_performance") else s.get("model_complexity") + 1)
        form.addRow("Pose model", self.model_combo)

        self.foreground_chk = QCheckBox("Inject input only while the game window has focus")
        self.foreground_chk.setChecked(bool(s.get("inject_only_foreground")))
        form.addRow(self.foreground_chk)

        self.dryrun_chk = QCheckBox("Dry run — recognize gestures but never inject input")
        self.dryrun_chk.setChecked(bool(s.get("dry_run")))
        form.addRow(self.dryrun_chk)

        self.calibrate_btn = QPushButton("Run calibration…")
        form.addRow(self.calibrate_btn)
        return w

    def _ai_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        online = self.engine.reasoner.online
        self.ai_status = QLabel(
            f"<b>AI engine:</b> {'🟢 Gemini connected' if online else '⚪ offline heuristics only'}<br>"
            "<span style='color:#9fb3c8'>Gemini identifies unknown games from screenshots, reads "
            "keybind meanings, and designs gesture mappings. Offline heuristics cover everything "
            "when no API is available. Set GEMINI_API_KEY in .env to enable.</span>")
        self.ai_status.setWordWrap(True)
        lay.addWidget(self.ai_status)
        self.identify_btn = QPushButton("Identify foreground window with AI")
        self.identify_btn.setEnabled(online)
        lay.addWidget(self.identify_btn)
        box = QGroupBox("Why these gestures? (AI rationale for the current profile)")
        v = QVBoxLayout(box)
        self.rationale_list = QListWidget()
        self.rationale_list.setWordWrap(True)
        v.addWidget(self.rationale_list)
        lay.addWidget(box, 1)
        return w

    # ------------------------------------------------------------------ wiring

    def _connect(self):
        b = self.bridge
        b.pose.connect(self._on_pose)
        b.gesture.connect(self._on_gesture)
        b.stats.connect(self._on_stats)
        b.game.connect(self._on_game)
        b.status.connect(lambda m: self.statusBar().showMessage(m, 8000))
        b.active.connect(self._on_active)

        self.arm_btn.clicked.connect(lambda: self.engine.set_active(self.arm_btn.isChecked()))
        self.calibrate_btn.clicked.connect(self._open_calibration)
        self.capture_btn.clicked.connect(self._open_capture)
        self.scan_btn.clicked.connect(self._scan_now)
        self.suggest_btn.clicked.connect(self._ai_suggest)
        self.save_btn.clicked.connect(self._save_profile)
        self.add_row_btn.clicked.connect(self._add_mapping_row)
        self.del_row_btn.clicked.connect(self._del_mapping_row)
        self.new_variant_btn.clicked.connect(self._new_variant)
        self.variant_combo.currentIndexChanged.connect(self._variant_changed)
        self.identify_btn.clicked.connect(self._identify_now)

        # settings side effects
        self.look_combo.currentTextChanged.connect(self._apply_settings)
        self.look_sens.valueChanged.connect(self._apply_settings)
        self.gest_sens.valueChanged.connect(self._apply_settings)
        self.access_combo.currentTextChanged.connect(self._apply_settings)
        self.model_combo.currentIndexChanged.connect(self._apply_settings)
        self.foreground_chk.toggled.connect(self._apply_settings)
        self.dryrun_chk.toggled.connect(self._apply_settings)
        self.camera_spin.valueChanged.connect(
            lambda v: self.engine.settings.set("camera_indices", [int(v)]))

        self._load_profile_ui(self.engine.profile)

    # ------------------------------------------------------------------ slots

    def _on_pose(self, pf, feats, states):
        self._pose_counter += 1
        self.camera.set_states(states)
        self.camera.update_frame(pf)

    def _on_gesture(self, ev, semantic, input_str, injected):
        if ev.kind not in (PULSE, START):
            return
        pretty = GESTURE_DESCRIPTIONS.get(ev.name, ev.name)
        if not semantic:
            suffix = "   (unmapped)"
        elif injected:
            suffix = f"  →  {semantic} ({input_str})  ✓ sent"
        else:
            why = ("disarmed — press F9" if not self.engine.active
                   else "game window not focused")
            suffix = f"  →  {semantic}  ✗ blocked: {why}"
        self.feed.insertItem(0, QListWidgetItem(f"{ev.name}{suffix}"))
        if self.feed.count() > 60:
            self.feed.takeItem(60)
        self.camera.flash_gesture(pretty if not semantic else f"{ev.name} → {semantic}")

    def _on_stats(self, st):
        self.stat_fps.setText(f"{st.camera_fps:.0f}")
        self.stat_pose.setText(f"{st.pose_ms:.1f}")
        self.stat_lat.setText(f"{st.pipeline_ms:.0f}")
        self.stat_p95.setText(f"{st.pipeline_p95_ms:.0f}")
        self.stat_model.setText(["lite", "full", "heavy"][st.model_complexity])

    def _on_game(self, info, profile):
        self.game_label.setText(f"🎮 {info.name}" if info.is_game else "No game detected")
        self._load_profile_ui(profile)

    def _on_active(self, active):
        self.arm_btn.setChecked(active)
        self.arm_btn.setText("DISARM  (F9)" if active else "ARM  (F9)")
        self.camera.set_armed(active)

    # ------------------------------------------------------------------ profile UI

    def _load_profile_ui(self, profile):
        self._loading = True
        self.variant_combo.blockSignals(True)
        self.variant_combo.clear()
        for v in self.engine.profiles.variants(profile.id) or [profile]:
            self.variant_combo.addItem(f"{profile.name} — {v.variant}", v)
        idx = self.variant_combo.findText(f"{profile.name} — {profile.variant}")
        self.variant_combo.setCurrentIndex(max(0, idx))
        self.variant_combo.blockSignals(False)

        self.table.setRowCount(0)
        for gesture, semantic in profile.gestures.items():
            self._append_row(profile, gesture, semantic)
        self.binds_table.setRowCount(0)
        for action, key in profile.discovered_binds.items():
            r = self.binds_table.rowCount()
            self.binds_table.insertRow(r)
            self.binds_table.setItem(r, 0, QTableWidgetItem(action))
            self.binds_table.setItem(r, 1, QTableWidgetItem(key))
        self.rationale_list.clear()
        for gesture, why in profile.rationale.items():
            desc = GESTURE_DESCRIPTIONS.get(gesture, gesture)
            self.rationale_list.addItem(QListWidgetItem(f"{desc}\n    {why}"))
        self._loading = False

    def _append_row(self, profile, gesture: str, semantic: str):
        r = self.table.rowCount()
        self.table.insertRow(r)
        g_combo = QComboBox()
        for g, desc in GESTURE_DESCRIPTIONS.items():
            if g != "t_pose":
                g_combo.addItem(f"{g} — {desc}", g)
        g_combo.setCurrentIndex(max(0, g_combo.findData(gesture)))
        a_combo = QComboBox()
        for sem in sorted(set(list(profile.actions.keys()) + SEMANTIC_ACTIONS)):
            a_combo.addItem(sem)
        a_combo.setCurrentText(semantic)
        action = profile.actions.get(semantic, {})
        input_item = QTableWidgetItem(action.get("input", "none"))
        mode_combo = QComboBox()
        mode_combo.addItems(["tap", "double", "toggle", "hold", "hold_pulse"])
        mode_combo.setCurrentText(action.get("mode", "tap"))
        self.table.setCellWidget(r, 0, g_combo)
        self.table.setCellWidget(r, 1, a_combo)
        self.table.setItem(r, 2, input_item)
        self.table.setCellWidget(r, 3, mode_combo)

    def _collect_table(self, profile) -> None:
        """Read the mapping table back into the profile (learning included)."""
        old = dict(profile.gestures)
        profile.gestures = {}
        for r in range(self.table.rowCount()):
            gesture = self.table.cellWidget(r, 0).currentData()
            semantic = self.table.cellWidget(r, 1).currentText()
            input_str = self.table.item(r, 2).text().strip() if self.table.item(r, 2) else "none"
            mode = self.table.cellWidget(r, 3).currentText()
            if not gesture or not semantic:
                continue
            profile.gestures[gesture] = semantic
            entry = profile.actions.setdefault(semantic, {"input": "none", "mode": mode, "hold_ms": 600})
            from motionforge.inputs.injector import InputInjector
            if InputInjector.is_valid_binding(input_str):
                entry["input"] = input_str
            entry["mode"] = mode
            # preference learning: user changed which gesture drives this action
            prev_gesture = next((g for g, s in old.items() if s == semantic), None)
            if prev_gesture and prev_gesture != gesture:
                self.engine.record_mapping_choice(semantic, gesture)

    def _save_profile(self):
        profile = self.engine.profile
        self._collect_table(profile)
        profile.source = "user"
        self.engine.profiles.save(profile)
        self.engine.set_profile(profile)
        self.statusBar().showMessage("Profile saved.", 5000)

    def _add_mapping_row(self):
        self._append_row(self.engine.profile, "punch_right", "attack")

    def _del_mapping_row(self):
        rows = {i.row() for i in self.table.selectedIndexes()}
        for r in sorted(rows, reverse=True):
            self.table.removeRow(r)

    def _new_variant(self):
        name, ok = QInputDialog.getText(self, "New profile variant",
                                        "Variant name (e.g. seated, competitive, fitness):")
        if ok and name.strip():
            p = self.engine.profiles.duplicate_as_variant(self.engine.profile, name.strip())
            self.engine.set_profile(p)

    def _variant_changed(self, idx):
        if getattr(self, "_loading", False):
            return
        p = self.variant_combo.currentData()
        if p is not None:
            self.engine.set_profile(p)

    # ------------------------------------------------------------------ actions

    def _open_calibration(self):
        CalibrationDialog(self.engine, self.bridge, self).exec()

    def _open_capture(self):
        KeybindCaptureDialog(self.engine.profile, self.engine.profiles, self.engine, self).exec()

    def _scan_now(self):
        info = self.engine.detector.current
        if not info.is_game:
            QMessageBox.information(self, "MotionForge", "No game detected to scan for.")
            return
        self.engine.profile.discovered_binds = {}
        threading.Thread(target=self.engine._enrich_profile,
                         args=(info, self.engine.profile), daemon=True).start()
        self.statusBar().showMessage("Scanning config files in the background…", 5000)

    def _ai_suggest(self):
        profile = self.engine.profile
        info = self.engine.detector.current
        gestures, rationale, source = self.engine.reasoner.suggest_mappings(
            info if info.is_game else _fake_game(profile),
            list(profile.actions.keys()), self.engine.settings.get("accessibility"))
        dlg = SuggestionDialog(gestures, rationale, source, self)
        if dlg.exec() and dlg.accepted_mappings:
            profile.gestures, profile.rationale = dlg.accepted_mappings
            self.engine.profiles.save(profile)
            self._load_profile_ui(profile)

    def _identify_now(self):
        self.statusBar().showMessage("Asking Gemini about the foreground window…", 5000)

        def work():
            hwnd, _, title = __import__("motionforge.detection.gamedetect",
                                        fromlist=["get_foreground"]).get_foreground()
            result = self.engine._ai_identify(hwnd, title)
            msg = (f"AI: {result.name} ({result.genre})" if result
                   else "AI: not recognized as a game.")
            self.bridge.status.emit(msg)
        threading.Thread(target=work, daemon=True).start()

    def _apply_settings(self, *_):
        s = self.engine.settings
        s.set("look_mode", self.look_combo.currentText())
        s.set("look_sensitivity", float(self.look_sens.value()))
        s.set("gesture_sensitivity", float(self.gest_sens.value()))
        s.set("accessibility", self.access_combo.currentText())
        idx = self.model_combo.currentIndex()
        s.set("auto_performance", idx == 0)
        if idx > 0:
            s.set("model_complexity", idx - 1)
            if self.engine.pose:
                self.engine.pose.auto_performance = False
                self.engine.pose.set_complexity(idx - 1)
        elif self.engine.pose:
            self.engine.pose.auto_performance = True
        s.set("inject_only_foreground", self.foreground_chk.isChecked())
        s.set("dry_run", self.dryrun_chk.isChecked())
        self.engine.injector.dry_run = self.dryrun_chk.isChecked()
        self.engine.look.set_mode(self.look_combo.currentText())
        self.engine.look_thread.sensitivity = float(self.look_sens.value())
        self.engine.recognizer.configure(float(self.gest_sens.value()),
                                         self.access_combo.currentText())

    def closeEvent(self, e):
        self.engine.stop()
        super().closeEvent(e)


def _fake_game(profile):
    from motionforge.core.events import GameInfo
    return GameInfo(id=profile.id, name=profile.name, genre=profile.genre)
