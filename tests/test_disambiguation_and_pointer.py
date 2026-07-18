"""Regression tests for the two field-reported bugs: similar gestures
cross-firing, and pointer mode (Chess.com) failing to click."""
import json

import numpy as np

from motionforge.core.selftest_data import _Builder, synthetic_stream, FPS
from motionforge.gestures.continuous import LookController
from motionforge.gestures.primitives import FeatureExtractor
from motionforge.gestures.recognizer import GestureRecognizer
from motionforge.profiles.manager import ProfileManager
from motionforge.vision import pose as P


def run(scenario: str):
    ex = FeatureExtractor()
    rec = GestureRecognizer()
    events = []
    for pf in synthetic_stream(scenario):
        events.extend(rec.update(ex.update(pf)))
    return [e.name for e in events if e.kind == "pulse"]


# ---- gesture confusion ------------------------------------------------------

def test_punch_does_not_fire_swing_or_chop():
    pulses = run("punch_right")
    assert "punch_right" in pulses
    assert "swing_right_arm" not in pulses
    assert "chop_right" not in pulses
    assert "throw_right" not in pulses


def test_swing_does_not_fire_punch():
    pulses = run("swing_right_arm")
    assert "swing_right_arm" in pulses
    assert "punch_right" not in pulses


def test_one_motion_one_pulse():
    """A single physical motion must emit exactly one pulse event."""
    for scenario in ("punch_right", "swing_right_arm", "jump_in_place"):
        pulses = run(scenario)
        assert len(pulses) == 1, f"{scenario} emitted {pulses}"


def test_arm_drop_is_not_a_chop():
    """Relaxing a raised arm down to your side must not read as a chop."""
    b = _Builder()
    b.hold(30)
    # raise the arm straight up
    b.interpolate(8, world_targets={P.R_WRIST: (-0.20, 0.75, 0.0),
                                    P.R_ELBOW: (-0.22, 0.45, 0.0)}, keep=True)
    b.hold(10)
    # drop it quickly back to the side (at the body, not out front)
    b.interpolate(4, world_targets={P.R_WRIST: (-0.28, -0.05, 0.0),
                                    P.R_ELBOW: (-0.25, 0.20, 0.0)}, keep=True)
    b.hold(10)
    ex, rec = FeatureExtractor(), GestureRecognizer()
    pulses = []
    for pf in b.frames:
        pulses += [e.name for e in rec.update(ex.update(pf)) if e.kind == "pulse"]
    assert "chop_right" not in pulses, pulses


def test_no_phantom_walk_when_legs_hidden():
    """Seated at a desk with legs out of frame: no walking, ever."""
    b = _Builder()
    b.vis[[P.L_KNEE, P.R_KNEE, P.L_ANKLE, P.R_ANKLE, P.L_FOOT, P.R_FOOT]] = 0.1
    b.hold(30)
    # noisy leg estimates jitter wildly when legs are guessed
    rng = np.random.default_rng(7)
    for _ in range(60):
        b.world0[P.L_KNEE][1] = -0.45 + rng.uniform(-0.3, 0.35)
        b.world0[P.R_KNEE][1] = -0.45 + rng.uniform(-0.3, 0.35)
        b.emit()
    ex, rec = FeatureExtractor(), GestureRecognizer()
    events = []
    for pf in b.frames:
        events += rec.update(ex.update(pf))
    assert all(e.name not in ("walk", "sprint") or e.kind == "end" for e in events)


def test_hand_at_mouth_fires_once_until_rearmed():
    ex, rec = FeatureExtractor(), GestureRecognizer()
    b = _Builder()
    b.hold(30)
    b.interpolate(6, world_targets={P.R_WRIST: (-0.03, 0.50, -0.08),
                                    P.R_ELBOW: (-0.20, 0.20, -0.10)}, keep=True)
    b.hold(int(3 * FPS))  # rest the hand at the mouth for 3 seconds
    fires = 0
    for pf in b.frames:
        fires += sum(1 for e in rec.update(ex.update(pf)) if e.name == "hand_to_mouth")
    assert fires == 1, f"hand resting at mouth fired {fires} times"


# ---- pointer mode / chess.com ----------------------------------------------

def _features_stream(builder):
    ex = FeatureExtractor()
    return [ex.update(pf) for pf in builder.frames]


def test_cursor_hand_tracks_and_dwell_clicks():
    look = LookController("cursor_hand")
    look.set_dwell(True)
    b = _Builder()
    # hand raised in front of the chest, inside the control box
    b.world0[P.R_WRIST] = (-0.10, 0.30, -0.30)
    b.img0[P.R_WRIST] = (0.42, 0.42)
    b.hold(int(1.6 * FPS))
    clicks, active = 0, 0
    for f in _features_stream(b):
        out = look.update(f)
        active += int(out.active and out.mode == "abs")
        clicks += int(out.click)
    assert active > 30, "cursor mode never engaged"
    assert clicks == 1, f"expected exactly one dwell click, got {clicks}"


def test_cursor_releases_when_hand_at_side():
    look = LookController("cursor_hand")
    look.set_dwell(True)
    b = _Builder()  # base pose: hands hanging at the sides
    b.hold(int(2 * FPS))
    for f in _features_stream(b):
        out = look.update(f)
        assert not out.active and not out.click


def test_pointer_template_uses_left_hand_and_dwell(tmp_path):
    pm = ProfileManager(profile_dir=tmp_path)
    pointer = pm.get("pointer")
    assert pointer.dwell_click is True
    assert pointer.gestures.get("punch_left") == "click"
    assert pointer.gestures.get("pinch_right") == "drag"
    assert pointer.actions["drag"] == {"input": "mouse:left", "mode": "hold", "hold_ms": 600}
    # no right-ARM gesture may be bound (the right hand owns the cursor);
    # a right-hand finger pinch is fine — it doesn't move the wrist
    assert all("right" not in g or g == "pinch_right" for g in pointer.gestures), pointer.gestures


def test_old_pointer_profile_is_migrated(tmp_path):
    """Profiles created by the old push-to-click build are auto-upgraded."""
    tmp_path.mkdir(exist_ok=True)
    old = {
        "id": "chess", "name": "Chess.com", "variant": "default",
        "match": {"processes": [], "titles": [], "steam_appids": [],
                  "browser_titles": ["chess.com"]},
        "genre": "pointer", "look_mode": "cursor_hand", "movement_mode": "off",
        "actions": {"click": {"input": "mouse:left", "mode": "tap", "hold_ms": 600}},
        "gestures": {"push": "click", "clap": "right_click"},
        "rationale": {}, "discovered_binds": {}, "source": "offline",
    }
    (tmp_path / "chess__default.json").write_text(json.dumps(old), encoding="utf-8")
    pm = ProfileManager(profile_dir=tmp_path)
    chess = pm.get("chess")
    assert chess.dwell_click is True
    assert chess.gestures.get("punch_left") == "click"
    assert chess.gestures.get("pinch_right") == "drag"
    assert "push" not in chess.gestures
    assert "drag" in chess.actions
    # migration persisted to disk
    stored = json.loads((tmp_path / "chess__default.json").read_text(encoding="utf-8"))
    assert stored["dwell_click"] is True
