"""Gesture engine tests over synthetic pose streams."""
import pytest

from motionforge.core.events import PULSE, START, END
from motionforge.core.selftest_data import synthetic_stream
from motionforge.gestures.primitives import FeatureExtractor
from motionforge.gestures.recognizer import GestureRecognizer


def run(scenario: str, **rec_kwargs):
    ex = FeatureExtractor()
    rec = GestureRecognizer(**rec_kwargs)
    events = []
    for pf in synthetic_stream(scenario):
        events.extend(rec.update(ex.update(pf)))
    return events


def names(events, kind=None):
    return [e.name for e in events if kind is None or e.kind == kind]


@pytest.mark.parametrize("scenario,expected,kind", [
    ("punch_right", "punch_right", PULSE),
    ("jump_in_place", "jump_in_place", PULSE),
    ("crouch", "crouch", START),
    ("walk", "walk", START),
    ("hand_to_mouth", "hand_to_mouth", PULSE),
    ("swing_right_arm", "swing_right_arm", PULSE),
    ("t_pose", "t_pose", PULSE),
    ("lean_left", "lean_left", START),
    ("block", "block", START),
])
def test_scenarios(scenario, expected, kind):
    evs = run(scenario)
    assert expected in names(evs, kind), f"{scenario}: got {[(e.kind, e.name) for e in evs]}"


def test_standing_is_quiet():
    """A person standing still must produce no pulse gestures (false-positive guard)."""
    ex = FeatureExtractor()
    rec = GestureRecognizer()
    quiet = []
    from motionforge.core.selftest_data import _Builder
    b = _Builder()
    b.hold(90)  # 3 seconds of stillness
    for pf in b.frames:
        quiet.extend(rec.update(ex.update(pf)))
    assert names(quiet, PULSE) == [], quiet


def test_crouch_releases():
    evs = run("crouch")
    kinds = [(e.kind, e.name) for e in evs if e.name == "crouch"]
    assert (START, "crouch") in kinds and (END, "crouch") in kinds


def test_one_handed_filters_other_arm():
    evs = run("punch_right", accessibility="one_handed_left")
    assert "punch_right" not in names(evs)


def test_seated_disables_jump():
    evs = run("jump_in_place", accessibility="seated")
    assert "jump_in_place" not in names(evs)


def test_person_leaving_frame_releases_states():
    ex = FeatureExtractor()
    rec = GestureRecognizer()
    stream = synthetic_stream("crouch")
    active_seen = False
    for pf in stream[:40]:  # enough to enter crouch
        rec.update(ex.update(pf))
        if "crouch" in rec.active_states():
            active_seen = True
            break
    assert active_seen
    evs = rec.update(None)  # person left the frame
    assert ("end", "crouch") in [(e.kind, e.name) for e in evs]
