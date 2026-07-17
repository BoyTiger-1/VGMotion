"""Virtual gamepad output (future expansion).

Architected as a drop-in alternative to keyboard/mouse bindings using the
"pad:*" binding namespace (e.g. "pad:a", "pad:rt", "pad:ls_up"). Enabling it
requires the ViGEmBus kernel driver plus the `vgamepad` package; until then
this stub reports unavailability and the UI hides pad bindings.
"""
from __future__ import annotations


def is_available() -> bool:
    try:
        import vgamepad  # noqa: F401
        return True
    except Exception:
        return False


class VirtualGamepad:
    """Minimal wrapper around vgamepad's VX360Gamepad (Xbox 360 pad)."""

    def __init__(self):
        if not is_available():
            raise RuntimeError(
                "Virtual gamepad support requires the ViGEmBus driver and "
                "`pip install vgamepad`. See README 'Controller emulation'.")
        import vgamepad
        self._vg = vgamepad
        self._pad = vgamepad.VX360Gamepad()

    def button(self, name: str, down: bool) -> None:
        btn = getattr(self._vg.XUSB_BUTTON, f"XUSB_GAMEPAD_{name.upper()}", None)
        if btn is None:
            return
        if down:
            self._pad.press_button(btn)
        else:
            self._pad.release_button(btn)
        self._pad.update()

    def left_stick(self, x: float, y: float) -> None:
        self._pad.left_joystick_float(x_value_float=x, y_value_float=y)
        self._pad.update()

    def right_stick(self, x: float, y: float) -> None:
        self._pad.right_joystick_float(x_value_float=x, y_value_float=y)
        self._pad.update()

    def trigger(self, side: str, value: float) -> None:
        if side == "left":
            self._pad.left_trigger_float(value_float=value)
        else:
            self._pad.right_trigger_float(value_float=value)
        self._pad.update()

    def reset(self) -> None:
        self._pad.reset()
        self._pad.update()
