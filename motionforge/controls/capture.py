"""Manual keybind capture: when automatic discovery finds nothing, the UI
walks the player through pressing each important control once."""
from __future__ import annotations

import ctypes
import threading
import time

from motionforge.inputs.injector import key_name_from_vk

user32 = ctypes.windll.user32
GetAsyncKeyState = user32.GetAsyncKeyState

# VKs we never treat as a captured bind (modifiers handled separately below)
_IGNORED_VKS = {0x5B, 0x5C}  # win keys


def wait_for_key(timeout: float = 10.0, poll_hz: int = 125,
                 cancel: threading.Event | None = None) -> str | None:
    """Block until the user presses any key or mouse button; returns our
    binding string ("key:r", "mouse:left") or None on timeout/cancel.

    Uses GetAsyncKeyState edge detection so it works regardless of window
    focus (the game keeps focus while the player presses its real controls).
    """
    # drain: wait for all keys to be released first so held keys don't fire
    deadline = time.perf_counter() + timeout
    interval = 1.0 / poll_hz
    prev = {vk: bool(GetAsyncKeyState(vk) & 0x8000) for vk in range(1, 255)}
    while time.perf_counter() < deadline:
        if cancel is not None and cancel.is_set():
            return None
        time.sleep(interval)
        for vk in range(1, 255):
            down = bool(GetAsyncKeyState(vk) & 0x8000)
            if down and not prev.get(vk) and vk not in _IGNORED_VKS:
                name = key_name_from_vk(vk)
                if name is None:
                    prev[vk] = down
                    continue
                return name if name.startswith("mouse:") else f"key:{name}"
            prev[vk] = down
    return None


# Semantic actions offered by the capture wizard, most important first
CAPTURE_ACTIONS: list[tuple[str, str]] = [
    ("move_forward", "Move forward"),
    ("move_back", "Move backward"),
    ("move_left", "Strafe left"),
    ("move_right", "Strafe right"),
    ("jump", "Jump"),
    ("crouch", "Crouch / sneak"),
    ("sprint", "Sprint"),
    ("attack", "Attack / fire (usually left click)"),
    ("use", "Use / aim / secondary (usually right click)"),
    ("reload", "Reload"),
    ("interact", "Interact / pick up"),
    ("inventory", "Inventory / menu"),
    ("ability1", "Ability 1"),
    ("ability2", "Ability 2"),
    ("heal", "Heal / consume item"),
    ("melee", "Melee"),
]
