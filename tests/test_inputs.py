"""Injector and action-executor tests (dry run: nothing actually injected)."""
import time

from motionforge.core.events import PULSE, START, END
from motionforge.inputs.actions import ActionBinding, ActionExecutor
from motionforge.inputs.injector import InputInjector, SCANCODES


def make():
    inj = InputInjector(dry_run=True)
    return inj, ActionExecutor(inj)


def test_scancodes_cover_bindings():
    for key in ("w", "a", "s", "d", "space", "lshift", "lctrl", "e", "r", "q",
                "f", "g", "h", "m", "x", "v", "1", "2", "3", "f5", "esc", "grave"):
        assert key in SCANCODES, key


def test_binding_validation():
    assert InputInjector.is_valid_binding("key:w")
    assert InputInjector.is_valid_binding("mouse:left")
    assert InputInjector.is_valid_binding("wheel:up")
    assert InputInjector.is_valid_binding("none")
    assert not InputInjector.is_valid_binding("key:notakey")
    assert not InputInjector.is_valid_binding("banana")


def test_tap_press_and_release():
    inj, ex = make()
    ex.handle(ActionBinding("key:space", "tap"), PULSE)
    time.sleep(0.12)
    assert "key down space" in inj.log and "key up space" in inj.log
    assert not inj.held
    ex.stop()


def test_hold_follows_state():
    inj, ex = make()
    ex.handle(ActionBinding("key:w", "hold"), START)
    assert "w" in inj.held
    ex.handle(ActionBinding("key:w", "hold"), END)
    assert "w" not in inj.held
    ex.stop()


def test_hold_pulse_extends():
    inj, ex = make()
    b = ActionBinding("mouse:left", "hold_pulse", hold_ms=120)
    ex.handle(b, PULSE)
    time.sleep(0.08)
    ex.handle(b, PULSE)          # extend before expiry
    assert "mouse:left" in inj.held
    time.sleep(0.25)
    assert "mouse:left" not in inj.held
    down_count = sum(1 for line in inj.log if line == "mouse down left")
    assert down_count == 1       # extended, not re-pressed
    ex.stop()


def test_toggle():
    inj, ex = make()
    b = ActionBinding("key:c", "toggle")
    ex.handle(b, PULSE)
    assert "c" in inj.held
    ex.handle(b, PULSE)
    assert "c" not in inj.held
    ex.stop()


def test_release_all_cleans_up():
    inj, ex = make()
    ex.handle(ActionBinding("key:w", "hold"), START)
    ex.handle(ActionBinding("key:lshift", "hold"), START)
    ex.release_all()
    assert not inj.held
    ex.stop()
