"""Tests for the expanded gesture library."""
import pytest

from motionforge.core.events import PULSE, START
from motionforge.core.selftest_data import synthetic_stream
from motionforge.gestures.primitives import FeatureExtractor
from motionforge.gestures.recognizer import GestureRecognizer


def run(scenario: str, **kw):
    ex = FeatureExtractor()
    rec = GestureRecognizer(**kw)
    events = []
    for pf in synthetic_stream(scenario):
        events.extend(rec.update(ex.update(pf)))
    return events


def names(events, kind=None):
    return [e.name for e in events if kind is None or e.kind == kind]


@pytest.mark.parametrize("scenario,expected,kind", [
    ("uppercut_right", "uppercut_right", PULSE),
    ("stomp_right", "stomp_right", PULSE),
    ("two_hand_swing", "two_hand_swing", PULSE),
    ("arms_crossed", "arms_crossed", START),
    ("hands_on_head", "hands_on_head", START),
    ("climb", "climb", START),
    ("head_shake", "head_shake", PULSE),
])
def test_new_scenarios(scenario, expected, kind):
    evs = run(scenario)
    assert expected in names(evs, kind), f"{scenario}: got {[(e.kind, e.name) for e in evs]}"


def test_uppercut_is_not_punch_or_chop():
    pulses = names(run("uppercut_right"), PULSE)
    assert "punch_right" not in pulses
    assert "chop_right" not in pulses


def test_stomp_is_not_kick():
    pulses = names(run("stomp_right"), PULSE)
    assert "kick_right" not in pulses


def test_two_hand_swing_beats_single_swings():
    pulses = names(run("two_hand_swing"), PULSE)
    assert "swing_left_arm" not in pulses
    assert "swing_right_arm" not in pulses


def test_arms_crossed_is_not_block():
    evs = run("arms_crossed")
    assert "block" not in names(evs)


def test_hands_on_head_is_not_arms_up():
    evs = run("hands_on_head")
    assert "arms_up" not in names(evs)


def test_one_handed_drops_two_handed_gestures():
    for scenario in ("two_hand_swing", "arms_crossed", "hands_on_head", "climb"):
        evs = run(scenario, accessibility="one_handed_left")
        assert scenario not in names(evs), scenario


def test_new_gestures_are_documented_and_selectable():
    from motionforge.gestures.library import GESTURE_DESCRIPTIONS, STATE_GESTURES
    for g in ("uppercut_left", "uppercut_right", "stomp_left", "stomp_right",
              "head_nod", "head_shake", "two_hand_swing", "arms_crossed",
              "hands_on_head", "climb"):
        assert g in GESTURE_DESCRIPTIONS, g
    assert {"arms_crossed", "hands_on_head", "climb"} <= STATE_GESTURES
