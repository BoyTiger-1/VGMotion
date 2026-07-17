"""Low-level input injection via Win32 SendInput.

Keys are injected as hardware SCANCODES (not virtual keys) because DirectInput
and raw-input games ignore virtual-key events. Mouse movement is injected as
relative motion, which FPS cameras require. Everything the game sees is
indistinguishable from a real keyboard/mouse.

Binding string syntax used throughout profiles:
    "key:w"  "key:space"  "key:lshift"  "mouse:left"  "mouse:right"
    "mouse:middle"  "wheel:up"  "wheel:down"  "none"
"""
from __future__ import annotations

import ctypes
import threading
import time
from ctypes import wintypes

SendInput = ctypes.windll.user32.SendInput

# ---- SendInput structures -------------------------------------------------

PUL = ctypes.POINTER(ctypes.c_ulong)


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort), ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong), ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long), ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong), ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong), ("dwExtraInfo", PUL)]


class _INPUTunion(ctypes.Union):
    _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("union", _INPUTunion)]


INPUT_MOUSE, INPUT_KEYBOARD = 0, 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008
KEYEVENTF_EXTENDEDKEY = 0x0001
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_VIRTUALDESK = 0x4000
MOUSEEVENTF_WHEEL = 0x0800
_MOUSE_BTN_FLAGS = {
    "left": (0x0002, 0x0004),
    "right": (0x0008, 0x0010),
    "middle": (0x0020, 0x0040),
}

# ---- scancode table (PS/2 set 1 make codes; 0xE000 marks extended) --------

SCANCODES: dict[str, int] = {
    "esc": 0x01, "1": 0x02, "2": 0x03, "3": 0x04, "4": 0x05, "5": 0x06,
    "6": 0x07, "7": 0x08, "8": 0x09, "9": 0x0A, "0": 0x0B, "minus": 0x0C,
    "equals": 0x0D, "backspace": 0x0E, "tab": 0x0F,
    "q": 0x10, "w": 0x11, "e": 0x12, "r": 0x13, "t": 0x14, "y": 0x15,
    "u": 0x16, "i": 0x17, "o": 0x18, "p": 0x19, "lbracket": 0x1A,
    "rbracket": 0x1B, "enter": 0x1C, "lctrl": 0x1D,
    "a": 0x1E, "s": 0x1F, "d": 0x20, "f": 0x21, "g": 0x22, "h": 0x23,
    "j": 0x24, "k": 0x25, "l": 0x26, "semicolon": 0x27, "apostrophe": 0x28,
    "grave": 0x29, "lshift": 0x2A, "backslash": 0x2B,
    "z": 0x2C, "x": 0x2D, "c": 0x2E, "v": 0x2F, "b": 0x30, "n": 0x31,
    "m": 0x32, "comma": 0x33, "period": 0x34, "slash": 0x35, "rshift": 0x36,
    "lalt": 0x38, "space": 0x39, "capslock": 0x3A,
    "f1": 0x3B, "f2": 0x3C, "f3": 0x3D, "f4": 0x3E, "f5": 0x3F, "f6": 0x40,
    "f7": 0x41, "f8": 0x42, "f9": 0x43, "f10": 0x44, "f11": 0x57, "f12": 0x58,
    # extended keys
    "rctrl": 0xE01D, "ralt": 0xE038, "up": 0xE048, "down": 0xE050,
    "left": 0xE04B, "right": 0xE04D, "insert": 0xE052, "delete": 0xE053,
    "home": 0xE047, "end": 0xE04F, "pageup": 0xE049, "pagedown": 0xE051,
}

VK_TO_NAME: dict[int, str] = {}  # filled below for keybind capture
_VK_MAP = {
    0x08: "backspace", 0x09: "tab", 0x0D: "enter", 0x10: "lshift", 0x11: "lctrl",
    0x12: "lalt", 0x1B: "esc", 0x20: "space", 0x21: "pageup", 0x22: "pagedown",
    0x23: "end", 0x24: "home", 0x25: "left", 0x26: "up", 0x27: "right",
    0x28: "down", 0x2D: "insert", 0x2E: "delete",
    0xA0: "lshift", 0xA1: "rshift", 0xA2: "lctrl", 0xA3: "rctrl",
    0xA4: "lalt", 0xA5: "ralt", 0xBD: "minus", 0xBB: "equals",
    0xDB: "lbracket", 0xDD: "rbracket", 0xBA: "semicolon", 0xDE: "apostrophe",
    0xC0: "grave", 0xDC: "backslash", 0xBC: "comma", 0xBE: "period", 0xBF: "slash",
    0x01: "mouse:left", 0x02: "mouse:right", 0x04: "mouse:middle",
}
for _c in range(0x30, 0x3A):
    _VK_MAP[_c] = chr(_c)                      # digits
for _c in range(0x41, 0x5B):
    _VK_MAP[_c] = chr(_c).lower()              # letters
for _i in range(1, 13):
    _VK_MAP[0x6F + _i] = f"f{_i}"              # F-keys
VK_TO_NAME.update(_VK_MAP)


def _send(inputs: list[INPUT]) -> None:
    n = len(inputs)
    arr = (INPUT * n)(*inputs)
    SendInput(n, arr, ctypes.sizeof(INPUT))


def _key_input(scan: int, up: bool) -> INPUT:
    flags = KEYEVENTF_SCANCODE | (KEYEVENTF_KEYUP if up else 0)
    if scan & 0xE000:
        flags |= KEYEVENTF_EXTENDEDKEY
        scan &= 0xFF
    inp = INPUT()
    inp.type = INPUT_KEYBOARD
    inp.union.ki = KEYBDINPUT(0, scan, flags, 0, None)
    return inp


def _mouse_input(dx: int = 0, dy: int = 0, data: int = 0, flags: int = 0) -> INPUT:
    inp = INPUT()
    inp.type = INPUT_MOUSE
    inp.union.mi = MOUSEINPUT(dx, dy, data, flags, 0, None)
    return inp


class InputInjector:
    """Thread-safe injector tracking pressed state so it can always clean up.

    dry_run=True logs instead of injecting (used in tests/selftest and the
    UI's 'test gestures without a game' mode)."""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.enabled = True
        self._pressed_keys: set[str] = set()
        self._pressed_buttons: set[str] = set()
        self._lock = threading.Lock()
        self.log: list[str] = []          # recent injected events (for UI/tests)
        self._log_cap = 200

    def _record(self, entry: str) -> None:
        self.log.append(entry)
        if len(self.log) > self._log_cap:
            del self.log[: len(self.log) - self._log_cap]

    # -- keyboard ---------------------------------------------------------

    def key(self, name: str, down: bool) -> None:
        name = name.lower()
        scan = SCANCODES.get(name)
        if scan is None:
            return
        with self._lock:
            if down:
                self._pressed_keys.add(name)
            else:
                self._pressed_keys.discard(name)
            self._record(f"key {'down' if down else 'up'} {name}")
            if not self.dry_run and self.enabled:
                _send([_key_input(scan, up=not down)])

    def tap_key(self, name: str, duration: float = 0.03) -> None:
        self.key(name, True)
        time.sleep(duration)
        self.key(name, False)

    # -- mouse --------------------------------------------------------------

    def button(self, btn: str, down: bool) -> None:
        btn = btn.lower()
        flags = _MOUSE_BTN_FLAGS.get(btn)
        if not flags:
            return
        with self._lock:
            if down:
                self._pressed_buttons.add(btn)
            else:
                self._pressed_buttons.discard(btn)
            self._record(f"mouse {'down' if down else 'up'} {btn}")
            if not self.dry_run and self.enabled:
                _send([_mouse_input(flags=flags[0] if down else flags[1])])

    def move_rel(self, dx: int, dy: int) -> None:
        if dx == 0 and dy == 0:
            return
        if not self.dry_run and self.enabled:
            _send([_mouse_input(dx=dx, dy=dy, flags=MOUSEEVENTF_MOVE)])

    def move_abs_norm(self, nx: float, ny: float) -> None:
        """Absolute cursor position, 0..1 across the virtual desktop."""
        x = int(max(0.0, min(1.0, nx)) * 65535)
        y = int(max(0.0, min(1.0, ny)) * 65535)
        if not self.dry_run and self.enabled:
            _send([_mouse_input(dx=x, dy=y,
                                flags=MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK)])

    def wheel(self, direction: str) -> None:
        delta = 120 if direction == "up" else -120
        with self._lock:
            self._record(f"wheel {direction}")
            if not self.dry_run and self.enabled:
                _send([_mouse_input(data=ctypes.c_ulong(delta & 0xFFFFFFFF).value,
                                    flags=MOUSEEVENTF_WHEEL)])

    # -- binding strings ----------------------------------------------------

    def binding_down(self, binding: str) -> None:
        kind, _, val = binding.partition(":")
        if kind == "key":
            self.key(val, True)
        elif kind == "mouse":
            self.button(val, True)
        elif kind == "wheel":
            self.wheel(val)

    def binding_up(self, binding: str) -> None:
        kind, _, val = binding.partition(":")
        if kind == "key":
            self.key(val, False)
        elif kind == "mouse":
            self.button(val, False)

    @staticmethod
    def is_valid_binding(binding: str) -> bool:
        kind, _, val = binding.partition(":")
        if kind == "key":
            return val in SCANCODES
        if kind == "mouse":
            return val in _MOUSE_BTN_FLAGS
        if kind == "wheel":
            return val in ("up", "down")
        return binding == "none"

    # -- safety ---------------------------------------------------------------

    def release_all(self) -> None:
        """Release every key/button we're holding. Called on pause, game
        switch, focus loss, and shutdown so the game is never left running."""
        with self._lock:
            keys, btns = list(self._pressed_keys), list(self._pressed_buttons)
        for k in keys:
            self.key(k, False)
        for b in btns:
            self.button(b, False)

    @property
    def held(self) -> list[str]:
        with self._lock:
            return sorted(self._pressed_keys) + [f"mouse:{b}" for b in sorted(self._pressed_buttons)]


def key_name_from_vk(vk: int) -> str | None:
    """Translate a Win32 virtual-key code to our key naming (keybind capture)."""
    return VK_TO_NAME.get(vk)
