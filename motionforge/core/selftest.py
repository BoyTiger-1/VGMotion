"""End-to-end self-test: exercises every pipeline stage without needing a
game, and reports a pass/fail summary. Run with `python -m motionforge --selftest`."""
from __future__ import annotations

import time
import traceback

import numpy as np

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str):
    def deco(fn):
        def wrapper(*a, **kw):
            try:
                detail = fn(*a, **kw) or ""
                RESULTS.append((name, True, str(detail)))
            except Exception as e:
                RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
                traceback.print_exc()
        return wrapper
    return deco


@check("imports")
def _imports():
    import mediapipe, cv2, PySide6, psutil, requests  # noqa
    import motionforge.core.engine  # noqa
    return "all modules import"


@check("pose models present")
def _models():
    from motionforge.vision.pose import model_path
    sizes = [model_path(i).stat().st_size for i in range(3)]
    assert all(s > 1_000_000 for s in sizes), sizes
    return f"lite/full/heavy = {[f'{s//1024}KB' for s in sizes]}"


@check("micro gestures: hand model + recognizer")
def _hands():
    import cv2
    from motionforge.vision.hands import ensure_hand_model, HandGestureEstimator
    path = ensure_hand_model()
    assert path.stat().st_size > 1_000_000
    est = HandGestureEstimator()
    img = np.full((480, 640, 3), 70, dtype=np.uint8)
    obs = est.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), time.perf_counter())
    est.close()
    return f"model {path.stat().st_size // 1024}KB, recognizer runs (hands seen: {len(obs or [])})"


@check("profiles: defaults + matching")
def _profiles():
    import tempfile
    from pathlib import Path
    from motionforge.profiles.manager import ProfileManager
    from motionforge.core.events import GameInfo
    with tempfile.TemporaryDirectory() as td:
        pm = ProfileManager(profile_dir=Path(td))
        mc = pm.for_game(GameInfo(id="minecraft", name="Minecraft", process="javaw.exe",
                                  window_title="Minecraft 1.21"))
        assert mc.id == "minecraft" and mc.gestures.get("chop_right") == "mine"
        kr = pm.for_game(GameInfo(id="browser:krunker", name="Krunker.io", process="chrome.exe",
                                  window_title="Krunker.io", is_browser=True))
        assert kr.id == "krunker"
        unknown = pm.for_game(GameInfo(id="unknown:foo.exe", name="Foo Quest",
                                       process="foo.exe", genre="fps"))
        assert unknown.gestures and unknown.id == "foo.exe"
        return f"{len(pm.all())} profiles, matching OK"


@check("game detection: knowledge base")
def _detection():
    from motionforge.detection import knowledgebase as kb
    assert kb.match_process("robloxplayerbeta.exe", "Roblox").id == "roblox"
    assert kb.match_process("javaw.exe", "Minecraft 1.21").id == "minecraft"
    assert kb.match_process("javaw.exe", "Eclipse IDE") is None
    assert kb.match_browser_title("Krunker.io - Google Chrome").id == "krunker"
    assert kb.match_browser_title("Play Chess Online - Chess.com").id == "chess"
    return "process/browser matching OK"


@check("gesture engine: synthetic punch + jump + walk")
def _gestures():
    from motionforge.core.selftest_data import synthetic_stream
    from motionforge.gestures.primitives import FeatureExtractor
    from motionforge.gestures.recognizer import GestureRecognizer

    fired = set()
    for scenario in ("punch_right", "jump_in_place", "walk", "hand_to_mouth", "crouch"):
        ex = FeatureExtractor()
        rec = GestureRecognizer()
        for pf in synthetic_stream(scenario):
            for ev in rec.update(ex.update(pf)):
                fired.add(ev.name)
    for expected in ("punch_right", "jump_in_place", "walk", "hand_to_mouth", "crouch"):
        assert expected in fired, f"{expected} not recognized (got {fired})"
    return f"recognized: {sorted(fired)}"


@check("input injector (dry run)")
def _injector():
    from motionforge.inputs.injector import InputInjector
    from motionforge.inputs.actions import ActionExecutor, ActionBinding
    from motionforge.core.events import PULSE, START, END
    inj = InputInjector(dry_run=True)
    ex = ActionExecutor(inj)
    ex.handle(ActionBinding("key:space", "tap"), PULSE, time.perf_counter())
    ex.handle(ActionBinding("key:w", "hold"), START)
    ex.handle(ActionBinding("mouse:left", "hold_pulse", 80), PULSE)
    time.sleep(0.25)
    ex.handle(ActionBinding("key:w", "hold"), END)
    ex.stop()
    log = " | ".join(inj.log)
    for needle in ("key down space", "key up space", "key down w", "key up w",
                   "mouse down left", "mouse up left"):
        assert needle in log, f"missing {needle!r} in {log}"
    assert not inj.held
    return "tap/hold/hold_pulse sequencing OK"


@check("real input injection (cursor)")
def _real_injection():
    """Injects an actual relative mouse move and confirms the OS cursor moved
    (then restores it). Verifies SendInput works end-to-end on this system."""
    import ctypes
    from ctypes import wintypes
    from motionforge.inputs.injector import InputInjector

    pt = wintypes.POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
    x0, y0 = pt.x, pt.y
    inj = InputInjector(dry_run=False)
    inj.move_rel(9, 7)
    time.sleep(0.05)
    ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
    moved = (pt.x, pt.y) != (x0, y0)
    ctypes.windll.user32.SetCursorPos(x0, y0)
    assert moved, "SendInput mouse move had no effect"
    return "OS cursor responded to injected movement"


@check("pose estimation on synthetic image")
def _pose_infer():
    import cv2
    from motionforge.vision.pose import PoseEstimator
    est = PoseEstimator(model_complexity=0, auto_performance=False)
    img = np.full((480, 640, 3), 60, dtype=np.uint8)
    cv2.circle(img, (320, 120), 40, (200, 180, 170), -1)   # not a person; just exercising
    t0 = time.perf_counter()
    pf = est.process(img, t0)
    est.close()
    return f"infer {pf.infer_ms:.1f}ms (no person expected: present={pf.present})"


@check("live camera + pose FPS")
def _camera(run: bool):
    if not run:
        return "skipped"
    from motionforge.vision.camera import CameraSource
    from motionforge.vision.pose import PoseEstimator
    cam = CameraSource(0)
    assert cam.start(), "camera 0 failed to open"
    est = PoseEstimator(model_complexity=1, auto_performance=False)
    frames, present = 0, 0
    t0 = time.perf_counter()
    try:
        while frames < 45 and time.perf_counter() - t0 < 10:
            got = cam.latest(timeout=1.0)
            if not got:
                continue
            pf = est.process(got[0], got[1])
            frames += 1
            present += int(pf.present)
    finally:
        cam.stop()
        est.close()
    dt = time.perf_counter() - t0
    fps = frames / dt if dt else 0
    assert frames > 10, f"only {frames} frames in {dt:.1f}s"
    return f"{fps:.1f} FPS end-to-end, person visible in {present}/{frames} frames, pose {est.infer_ms:.1f}ms"


@check("Gemini API")
def _gemini():
    from motionforge import config
    from motionforge.ai.gemini import GeminiClient
    key = config.get_gemini_api_key()
    if not key:
        return "no GEMINI_API_KEY configured — offline heuristics active"
    client = GeminiClient(key, config.Settings().get("gemini_model"),
                          config.Settings().get("gemini_fallback_models"))
    ok = client.ping()
    if not ok and "quota" in (client.last_error or "").lower():
        # daily free-tier cap: integration works, budget is spent for today
        return "key valid; free-tier daily quota exhausted — offline heuristics active until reset"
    assert ok, f"Gemini unreachable: {client.last_error}"
    return "Gemini responded OK"


@check("offline gesture suggestions")
def _offline_ai():
    from motionforge.ai import offline
    g, r = offline.suggest_gestures("fps", ["move_forward", "jump", "shoot", "reload", "aim"])
    assert g.get("walk") == "move_forward" and g.get("punch_right") == "shoot"
    g2, _ = offline.suggest_gestures("fps", ["move_forward", "jump", "shoot"], "one_handed_left")
    assert all("right" not in k for k in g2), g2
    return f"fps template: {len(g)} mappings; accessibility filters OK"


def run_selftest(camera: bool = True) -> int:
    print("MotionForge self-test\n" + "=" * 60)
    _imports()
    _models()
    _hands()
    _profiles()
    _detection()
    _gestures()
    _injector()
    _real_injection()
    _pose_infer()
    _camera(camera)
    _gemini()
    _offline_ai()

    print()
    failed = 0
    for name, ok, detail in RESULTS:
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {name:<38} {detail}")
        failed += 0 if ok else 1
    print("=" * 60)
    print(f"{len(RESULTS) - failed}/{len(RESULTS)} checks passed")
    return 1 if failed else 0
