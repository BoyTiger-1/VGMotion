"""MotionEngine: owns the full pipeline
camera -> pose -> features -> gestures -> profile mapping -> input injection
plus game detection, profile switching, AI enrichment, latency tracking, and
the safety systems (kill switch, T-pose toggle, foreground-focus guard)."""
from __future__ import annotations

import ctypes
import threading
import time

from motionforge import config
from motionforge.ai.gemini import GeminiClient
from motionforge.ai.reasoning import AIReasoner, PreferenceStore
from motionforge.controls import discovery
from motionforge.core.events import Callbacks, GameInfo, GestureEvent, PipelineStats, PULSE
from motionforge.inputs.actions import ActionBinding
from motionforge.core.latency import LatencyMonitor
from motionforge.detection.gamedetect import GameDetector, screenshot_window
from motionforge.gestures.continuous import LookController
from motionforge.gestures.primitives import FeatureExtractor
from motionforge.gestures.recognizer import GestureRecognizer
from motionforge.inputs.actions import ActionExecutor, LookThread
from motionforge.inputs.injector import InputInjector
from motionforge.profiles.manager import Profile, ProfileManager
from motionforge.vision.calibration import Calibration
from motionforge.vision.camera import MultiCamera
from motionforge.vision.pose import PoseEstimator

_VK = {f"f{i}": 0x6F + i for i in range(1, 13)}
_VK.update({"esc": 0x1B, "pause": 0x13, "scrolllock": 0x91})


class MotionEngine:
    def __init__(self, settings: config.Settings | None = None):
        config.ensure_app_dirs()
        self.settings = settings or config.Settings()

        # callbacks the UI subscribes to
        self.on_pose = Callbacks()       # (PoseFrame, Features|None, [active_state_names])
        self.on_gesture = Callbacks()    # (GestureEvent, semantic|None, input|None, injected: bool)
        self.on_stats = Callbacks()      # (PipelineStats)
        self.on_game = Callbacks()       # (GameInfo, Profile)
        self.on_status = Callbacks()     # (str)
        self.on_active = Callbacks()     # (bool armed)

        # AI
        key = config.get_gemini_api_key()
        self.gemini = GeminiClient(
            key, self.settings.get("gemini_model"),
            self.settings.get("gemini_fallback_models")) if (key and self.settings.get("ai_enabled")) else None
        self.prefs = PreferenceStore(config.PREFS_PATH)
        self.reasoner = AIReasoner(self.gemini, self.prefs)

        # input
        self.injector = InputInjector(dry_run=self.settings.get("dry_run"))
        self.executor = ActionExecutor(self.injector)
        self.latency = LatencyMonitor()
        self.executor.on_inject = self._on_inject
        self.look_thread = LookThread(self.injector)
        self.look_thread.sensitivity = self.settings.get("look_sensitivity")

        # vision + gestures
        self.cameras: MultiCamera | None = None
        self.pose: PoseEstimator | None = None
        self.extractor = FeatureExtractor(
            Calibration.from_dict(self.settings.get("calibration") or {}))
        self.recognizer = GestureRecognizer(self.settings.get("gesture_sensitivity"),
                                            self.settings.get("accessibility"))
        self.look = LookController(self.settings.get("look_mode"))

        # game detection + profiles
        self.profiles = ProfileManager(settings=self.settings)
        self.profile: Profile = self.profiles.get("generic") or Profile()
        self.recognizer.set_mapped(self.profile.gestures.keys())
        self.detector = GameDetector(ai_identifier=self._ai_identify
                                     if self.gemini and self.settings.get("ai_screenshot_identify") else None)
        self.detector.on_game_changed.subscribe(self._on_game_changed)

        # state
        self.active = False              # motion control armed
        self.running = False
        self._inject_ok = False
        self._kill_prev = False
        self._stats = PipelineStats()
        self._last_stats_emit = 0.0
        self._low_vis_since: float | None = None
        self._vision_thread: threading.Thread | None = None

    # ------------------------------------------------------------------ setup

    def start(self, start_camera: bool = True) -> None:
        self.running = True
        self.detector.start()
        if start_camera:
            self.start_camera()
        self.on_status.emit("Engine started. Press F9 or hold a T-pose to arm/disarm.")

    def start_camera(self) -> bool:
        s = self.settings
        self.cameras = MultiCamera(s.get("camera_indices"), s.get("frame_width"),
                                   s.get("frame_height"), s.get("target_fps"))
        if not self.cameras.start():
            self.on_status.emit("No camera could be opened — check Settings ▸ Camera.")
            return False
        if self.pose is None:
            self.pose = PoseEstimator(s.get("model_complexity"), s.get("auto_performance"))
        self._vision_thread = threading.Thread(target=self._vision_loop,
                                               name="mf-vision", daemon=True)
        self._vision_thread.start()
        return True

    def stop(self) -> None:
        self.running = False
        self.set_active(False)
        if self._vision_thread:
            self._vision_thread.join(timeout=2.0)
        if self.cameras:
            self.cameras.stop()
        self.detector.stop()
        self.look_thread.stop()
        self.executor.stop()
        if self.pose:
            self.pose.close()

    # -------------------------------------------------------------- arming

    def set_active(self, active: bool) -> None:
        if active == self.active:
            return
        self.active = active
        if not active:
            self.executor.release_all()
            self.look_thread.set_enabled(False)
        self.on_active.emit(active)
        self.on_status.emit("Motion control ARMED — your body is the controller."
                            if active else "Motion control paused.")

    def toggle_active(self) -> None:
        self.set_active(not self.active)

    # ------------------------------------------------------------ vision loop

    def _vision_loop(self) -> None:
        while self.running and self.cameras:
            got = self.cameras.latest(timeout=0.5)
            if got is None:
                self._check_kill_switch()
                continue
            frame, ts = got
            try:
                self._process_frame(frame, ts)
            except Exception:
                import traceback
                traceback.print_exc()

    def _process_frame(self, frame, ts: float) -> None:
        pf = self.pose.process(frame, ts)
        feats = self.extractor.update(pf)
        events = self.recognizer.update(feats)
        self._check_kill_switch()
        self._maybe_switch_camera(pf, ts)

        # T-pose is reserved: always toggles arming, never reaches the game
        for ev in list(events):
            if ev.name == "t_pose" and ev.kind == PULSE:
                events.remove(ev)
                self.toggle_active()

        inject_ok = (self.active and self.profile is not None
                     and (not self.settings.get("inject_only_foreground")
                          or self.detector.foreground_is_game))
        if self._inject_ok and not inject_ok:
            self.executor.release_all()          # focus lost / disarmed: clean up
        self._inject_ok = inject_ok

        # continuous look / cursor channel
        out = self.look.update(feats)
        self.look_thread.set_enabled(inject_ok and out.active)
        self.look_thread.set_signal(out.mode, out.x, out.y, out.active)
        if out.click:   # dwell-click from cursor_hand mode
            binding = ActionBinding.from_dict(self.profile.actions.get(
                "click", {"input": "mouse:left", "mode": "tap"})) if self.profile else \
                ActionBinding("mouse:left", "tap")
            if inject_ok:
                self.executor.handle(binding, PULSE, ts)
            self.on_gesture.emit(GestureEvent(PULSE, "dwell_click", ts, 1.0, ts),
                                 "click", binding.input, inject_ok)

        # discrete gestures -> semantic actions -> injected input
        for ev in events:
            mapped = self.profile.binding_for_gesture(ev.name) if self.profile else None
            semantic, binding = mapped if mapped else (None, None)
            injected = bool(inject_ok and binding)
            if injected:
                self.executor.handle(binding, ev.kind, ev.capture_ts)
            self.on_gesture.emit(ev, semantic, binding.input if binding else None, injected)

        # stats + UI frame
        self._stats.frames += 1
        self._stats.camera_fps = self.cameras.camera_fps
        self._stats.pose_ms = self.pose.infer_ms
        self._stats.pipeline_ms = self.latency.ema_ms
        self._stats.pipeline_p95_ms = self.latency.p95_ms
        self._stats.model_complexity = self.pose.model_complexity
        self._stats.active = self.active
        self.on_pose.emit(pf, feats, self.recognizer.active_states())
        now = time.perf_counter()
        if now - self._last_stats_emit > 0.5:
            self._last_stats_emit = now
            self.on_stats.emit(self._stats)

    def _on_inject(self, capture_ts: float) -> None:
        self.latency.add_ms((time.perf_counter() - capture_ts) * 1000.0)

    def _check_kill_switch(self) -> None:
        vk = _VK.get(self.settings.get("kill_switch_key", "f9"), 0x78)
        down = bool(ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000)
        if down and not self._kill_prev:
            self.toggle_active()
        self._kill_prev = down

    def _maybe_switch_camera(self, pf, ts: float) -> None:
        """If the subject is lost on the primary camera and another camera is
        configured, fail over to it."""
        if not self.cameras or len(self.cameras.sources) < 2:
            return
        core = pf.vis[[11, 12, 23, 24]].mean() if pf.present else 0.0
        if core < 0.35:
            if self._low_vis_since is None:
                self._low_vis_since = ts
            elif ts - self._low_vis_since > 2.0:
                idx = self.cameras.switch_primary()
                self._low_vis_since = None
                self.on_status.emit(f"Low visibility — switched to camera {self.cameras.sources[idx].index}.")
        else:
            self._low_vis_since = None

    # ------------------------------------------------------------ game switch

    def _on_game_changed(self, info: GameInfo) -> None:
        self.executor.release_all()
        self.profile = self.profiles.for_game(info)
        self._apply_profile(self.profile)
        self.on_game.emit(info, self.profile)
        if info.is_game and info.id:
            threading.Thread(target=self._enrich_profile, args=(info, self.profile),
                             name="mf-enrich", daemon=True).start()

    def _apply_profile(self, p: Profile) -> None:
        self.look.set_mode(p.look_mode)
        self.look.set_dwell(p.dwell_click)
        total_look = p.look_sensitivity * self.settings.get("look_sensitivity")
        self.look.cursor_scale = total_look
        self.look_thread.sensitivity = total_look
        self.recognizer.configure(p.gesture_sensitivity * self.settings.get("gesture_sensitivity"),
                                  self.settings.get("accessibility"))
        self.recognizer.set_mapped(p.gestures.keys())

    def set_profile(self, p: Profile) -> None:
        self.executor.release_all()
        self.profile = p
        self.profiles.remember_variant(p.id, p.variant)
        self._apply_profile(p)
        self.on_game.emit(self.detector.current, p)

    def _enrich_profile(self, info: GameInfo, profile: Profile) -> None:
        """Background: discover real keybinds, infer semantics, and (for
        auto-derived profiles) generate AI gesture mappings."""
        try:
            if profile.discovered_binds:
                return  # already enriched
            bare_id = info.id.split(":", 1)[-1]
            found = discovery.discover_for_game(bare_id, info.exe_path)
            changed = False
            if found:
                profile.discovered_binds = found
                semantics = self.reasoner.infer_semantics(info, found)
                for semantic, binding in semantics.items():
                    if semantic in profile.actions:
                        if profile.actions[semantic]["input"] != binding:
                            profile.actions[semantic]["input"] = binding
                            changed = True
                    else:
                        from motionforge.ai.offline import DEFAULT_INPUTS
                        _, mode = DEFAULT_INPUTS.get(semantic, ("none", "tap"))
                        profile.actions[semantic] = {"input": binding, "mode": mode, "hold_ms": 600}
                        changed = True
                if changed:
                    self.on_status.emit(f"Discovered {len(found)} keybinds for {info.name}.")
            # AI gesture suggestions for auto-derived (non-bundled, non-user)
            # profiles. Pointer-genre profiles are excluded: their scheme
            # (hand cursor + left-hand click + dwell) is deliberate and the AI
            # must not replace it with gestures that would move the cursor.
            if profile.source == "offline" and profile.genre != "pointer" and self.reasoner.online:
                gestures, rationale, src = self.reasoner.suggest_mappings(
                    info, list(profile.actions.keys()), self.settings.get("accessibility"))
                if gestures:
                    profile.gestures, profile.rationale = gestures, rationale
                    profile.source = src
                    changed = True
                    self.on_status.emit(f"AI generated {len(gestures)} gesture mappings for {info.name}.")
            if changed:
                self.profiles.save(profile)
                if self.profile.id == profile.id and self.profile.variant == profile.variant:
                    self.on_game.emit(info, profile)
        except Exception:
            import traceback
            traceback.print_exc()

    def _ai_identify(self, hwnd: int, title: str) -> GameInfo | None:
        shot = screenshot_window(hwnd)
        return self.reasoner.identify_game(shot, title)

    # ------------------------------------------------------------ user actions

    def apply_calibration(self, cal: Calibration) -> None:
        self.extractor.set_calibration(cal)
        self.settings.set("calibration", cal.to_dict())
        self.on_status.emit("Calibration saved." if cal.valid else "Calibration failed — try again.")

    def record_mapping_choice(self, semantic: str, gesture: str) -> None:
        """Called by the UI when the user remaps a gesture (preference learning)."""
        self.prefs.record_choice(semantic, gesture)
