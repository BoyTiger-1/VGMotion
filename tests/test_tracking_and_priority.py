"""Tests for the One-Euro landmark filter, mapped-gesture priority, look
smoothing, and AI action restriction."""
import numpy as np

from motionforge.core.events import GestureEvent, PULSE
from motionforge.core.selftest_data import _Builder, synthetic_stream
from motionforge.gestures.library import PulseDetector
from motionforge.gestures.primitives import FeatureExtractor
from motionforge.gestures.recognizer import GestureRecognizer
from motionforge.vision.filters import OneEuroFilter


# ---- One-Euro filter --------------------------------------------------------

def test_one_euro_removes_jitter():
    f = OneEuroFilter()
    rng = np.random.default_rng(3)
    raw, smoothed = [], []
    for i in range(120):
        x = np.array([0.5 + rng.normal(0, 0.01)], dtype=np.float32)  # jittery but static
        raw.append(float(x[0]))
        smoothed.append(float(f.filter(x, i / 30.0)[0]))
    assert np.std(smoothed[30:]) < 0.35 * np.std(raw[30:])   # jitter crushed at rest


def test_one_euro_tracks_fast_motion():
    f = OneEuroFilter()
    t = 0.0
    for i in range(30):                                # settle at 0
        f.filter(np.array([0.0], dtype=np.float32), t)
        t += 1 / 30
    # step through a fast punch-like ramp: 0 -> 0.6m in 5 frames (3.6 m/s)
    out = 0.0
    for i in range(5):
        out = float(f.filter(np.array([(i + 1) * 0.12], dtype=np.float32), t)[0])
        t += 1 / 30
    assert out > 0.42, f"fast motion over-smoothed: reached {out:.3f} of 0.6"


def test_one_euro_reset():
    f = OneEuroFilter()
    f.filter(np.array([1.0], dtype=np.float32), 0.0)
    f.reset()
    assert float(f.filter(np.array([5.0], dtype=np.float32), 1.0)[0]) == 5.0


# ---- mapped-gesture priority ------------------------------------------------

class _Stub(PulseDetector):
    def __init__(self, name, priority, fire_at):
        super().__init__()
        self.name = name
        self.priority = priority
        self.cooldown = 99.0
        self._fire_at = fire_at
        self._count = 0

    def update(self, f):
        self._count += 1
        if self._count == self._fire_at:
            return self._fire(f)
        return []


def _run_stubs(mapped, stubs):
    rec = GestureRecognizer()
    rec.detectors = stubs
    rec.set_mapped(mapped)
    ex = FeatureExtractor()
    b = _Builder()
    b.hold(30)
    pulses = []
    for pf in b.frames:
        pulses += [e.name for e in rec.update(ex.update(pf)) if e.kind == PULSE]
    return pulses


def test_mapped_gesture_beats_higher_priority_unmapped():
    """The user's punch is bound to click; a same-motion 'uppercut' reading
    with higher priority must not steal it."""
    pulses = _run_stubs(
        {"punch_right"},
        [_Stub("uppercut_right", 90, fire_at=5), _Stub("punch_right", 50, fire_at=6)])
    assert pulses == ["punch_right"], pulses


def test_priority_decides_between_two_mapped():
    pulses = _run_stubs(
        {"punch_right", "uppercut_right"},
        [_Stub("uppercut_right", 90, fire_at=5), _Stub("punch_right", 50, fire_at=6)])
    assert pulses == ["uppercut_right"], pulses


def test_unmapped_emit_does_not_suppress_mapped():
    """An unmapped gesture firing first must not mute a mapped gesture that
    follows moments later (outside the decision window)."""
    pulses = _run_stubs(
        {"punch_right"},
        [_Stub("uppercut_right", 90, fire_at=3), _Stub("punch_right", 50, fire_at=9)])
    assert "punch_right" in pulses, pulses


def test_without_profile_priority_rules():
    pulses = _run_stubs(
        set(),
        [_Stub("uppercut_right", 90, fire_at=5), _Stub("punch_right", 50, fire_at=6)])
    assert pulses == ["uppercut_right"], pulses


def test_real_punch_still_wins_with_mapping():
    ex = FeatureExtractor()
    rec = GestureRecognizer()
    rec.set_mapped({"punch_right"})
    pulses = []
    for pf in synthetic_stream("punch_right"):
        pulses += [e.name for e in rec.update(ex.update(pf)) if e.kind == PULSE]
    assert pulses == ["punch_right"], pulses


# ---- AI restricted to the game's real actions -------------------------------

class _FakeGemini:
    def __init__(self, payload):
        self.payload = payload

    def generate_json(self, prompt, image_jpeg=None):
        return self.payload


def test_ai_suggestions_hard_filtered_to_real_actions(tmp_path):
    from motionforge.ai.reasoning import AIReasoner, PreferenceStore
    from motionforge.core.events import GameInfo
    reasoner = AIReasoner(
        _FakeGemini({"gestures": {
            "punch_right": "click",
            "wave": "made_up_action",          # invented action -> dropped
            "made_up_gesture": "click",        # invented gesture -> dropped
            "clap": "Right_Click ",            # normalized, but not offered -> dropped
        }, "rationale": {"punch_right": "jab to click"}}),
        PreferenceStore(tmp_path / "prefs.json"))
    gestures, rationale, source = reasoner.suggest_mappings(
        GameInfo(id="x", name="Pointer Game", genre="pointer"), ["click"])
    assert source == "ai"
    assert gestures == {"punch_right": "click"}, gestures
