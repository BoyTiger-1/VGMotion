"""Automatic control discovery.

Strategies, in order:
1. Game-specific parsers (Minecraft options.txt) with exact semantics.
2. Generic scan: look for keybind-shaped config files (json/ini/xml/yaml/cfg)
   near the game install dir and in the usual Windows config locations, and
   extract action->key pairs best-effort.
Discovered binds are stored on the game's profile so discovery runs once.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

# Minecraft key.keyboard.* -> our injector key names
_MC_KEY = {
    "key.keyboard.space": "space", "key.keyboard.left.shift": "lshift",
    "key.keyboard.right.shift": "rshift", "key.keyboard.left.control": "lctrl",
    "key.keyboard.right.control": "rctrl", "key.keyboard.left.alt": "lalt",
    "key.keyboard.right.alt": "ralt", "key.keyboard.tab": "tab",
    "key.keyboard.enter": "enter", "key.keyboard.escape": "esc",
    "key.keyboard.backspace": "backspace", "key.keyboard.caps.lock": "capslock",
    "key.keyboard.up": "up", "key.keyboard.down": "down",
    "key.keyboard.left": "left", "key.keyboard.right": "right",
    "key.mouse.left": "mouse:left", "key.mouse.right": "mouse:right",
    "key.mouse.middle": "mouse:middle",
}
for _ch in "abcdefghijklmnopqrstuvwxyz0123456789":
    _MC_KEY[f"key.keyboard.{_ch}"] = _ch
for _i in range(1, 13):
    _MC_KEY[f"key.keyboard.f{_i}"] = f"f{_i}"

# Minecraft option key -> semantic action
_MC_SEMANTIC = {
    "key_key.jump": "jump", "key_key.sneak": "sneak", "key_key.sprint": "sprint",
    "key_key.forward": "move_forward", "key_key.back": "move_back",
    "key_key.left": "move_left", "key_key.right": "move_right",
    "key_key.attack": "attack", "key_key.use": "use",
    "key_key.inventory": "inventory", "key_key.drop": "drop",
    "key_key.swapOffhand": "swap_hands", "key_key.togglePerspective": "camera",
    "key_key.chat": "chat", "key_key.pickItem": "pick_block",
}


def _binding_from_mc(value: str) -> str | None:
    value = value.strip().strip('"')
    mapped = _MC_KEY.get(value)
    if not mapped:
        return None
    return mapped if mapped.startswith("mouse:") else f"key:{mapped}"


def discover_minecraft(minecraft_dir: Path | None = None) -> dict[str, str]:
    """Parse .minecraft/options.txt into {semantic_action: binding}."""
    if minecraft_dir is None:
        appdata = os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming"))
        minecraft_dir = Path(appdata) / ".minecraft"
    options = minecraft_dir / "options.txt"
    binds: dict[str, str] = {}
    if not options.exists():
        return binds
    try:
        for line in options.read_text(encoding="utf-8", errors="ignore").splitlines():
            key, _, value = line.partition(":")
            semantic = _MC_SEMANTIC.get(key.strip())
            if not semantic:
                continue
            binding = _binding_from_mc(value)
            if binding:
                binds[semantic] = binding
    except OSError:
        pass
    return binds


# --------------------------------------------------------------------------
# Generic keybind file scanning
# --------------------------------------------------------------------------

_BIND_FILE_HINTS = re.compile(r"(keybind|keybinding|input|control|binding)", re.I)
_BIND_EXTS = {".json", ".ini", ".cfg", ".xml", ".yml", ".yaml", ".txt", ".config"}
_KEY_TOKEN = re.compile(r"^(?:key\.)?([a-z0-9]|f[0-9]{1,2}|space|tab|enter|shift|lshift|rshift|ctrl|lctrl|rctrl|alt|lalt|ralt|mouse[12345]|lmb|rmb|mmb|up|down|left|right)$", re.I)

_ACTION_WORDS = re.compile(
    r"(jump|crouch|sneak|sprint|run|attack|fire|shoot|reload|use|interact"
    r"|inventory|forward|backward|back|strafe|left|right|aim|ads|block|parry"
    r"|dodge|roll|heal|build|place|mine|map|melee|grenade|ability|ultimate"
    r"|craft|drop|swap|zoom|scope)", re.I)

_MOUSE_TOKEN = {"mouse1": "mouse:left", "lmb": "mouse:left", "mouse2": "mouse:right",
                "rmb": "mouse:right", "mouse3": "mouse:middle", "mmb": "mouse:middle"}


def _normalize_key_token(tok: str) -> str | None:
    tok = tok.strip().strip('"').strip("'").lower().removeprefix("key.")
    if tok in _MOUSE_TOKEN:
        return _MOUSE_TOKEN[tok]
    if tok in ("shift", "lshift"):
        return "key:lshift"
    if tok in ("ctrl", "control", "lctrl"):
        return "key:lctrl"
    if tok in ("alt", "lalt"):
        return "key:lalt"
    from motionforge.inputs.injector import SCANCODES
    return f"key:{tok}" if tok in SCANCODES else None


def find_bind_files(game_dir: Path, extra_dirs: list[Path] | None = None,
                    max_files: int = 40) -> list[Path]:
    """Locate plausible keybind config files near a game install."""
    roots: list[Path] = []
    if game_dir and game_dir.exists():
        roots.append(game_dir)
    home = Path.home()
    for env, sub in (("APPDATA", ""), ("LOCALAPPDATA", "")):
        base = os.environ.get(env)
        if base:
            roots.append(Path(base))
    roots += [home / "Documents" / "My Games", home / "Documents", home / "Saved Games"]

    found: list[Path] = []
    game_name = game_dir.name.lower() if game_dir else ""
    for root in roots:
        if not root.exists():
            continue
        depth_limit = 3 if root == game_dir else 2
        try:
            for path in _walk_limited(root, depth_limit):
                if len(found) >= max_files:
                    return found
                if path.suffix.lower() not in _BIND_EXTS:
                    continue
                rel = str(path).lower()
                near_game = game_name and game_name in rel
                if _BIND_FILE_HINTS.search(path.name) or (near_game and _BIND_FILE_HINTS.search(rel)):
                    if path.stat().st_size < 2_000_000:
                        found.append(path)
        except (OSError, PermissionError):
            continue
    return found


def _walk_limited(root: Path, max_depth: int):
    base_depth = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root):
        if len(Path(dirpath).parts) - base_depth >= max_depth:
            dirnames[:] = []
        for f in filenames:
            yield Path(dirpath) / f


def parse_bind_file(path: Path) -> dict[str, str]:
    """Best-effort extraction of {action_name: binding} from a config file."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return {}
    binds: dict[str, str] = {}

    if path.suffix.lower() == ".json":
        try:
            data = json.loads(text)
            _walk_json(data, binds)
            return binds
        except json.JSONDecodeError:
            pass

    # line-oriented: `action=key`, `action: key`, `bind "key" "action"`
    for line in text.splitlines()[:5000]:
        line = line.strip()
        m = re.match(r'bind\s+"?([\w]+)"?\s+"?([\w+]+)"?', line, re.I)  # source-engine style
        if m:
            key, action = m.group(1), m.group(2)
            if _ACTION_WORDS.search(action):
                b = _normalize_key_token(key)
                if b:
                    binds[action.lower()] = b
            continue
        m = re.match(r'"?([\w .]+?)"?\s*[=:]\s*"?([\w.]+)"?$', line)
        if m and _ACTION_WORDS.search(m.group(1)):
            b = _normalize_key_token(m.group(2))
            if b:
                binds[m.group(1).strip().lower()] = b
    return binds


def _walk_json(node, binds: dict, prefix: str = "") -> None:
    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(v, str) and _ACTION_WORDS.search(str(k)):
                b = _normalize_key_token(v)
                if b:
                    binds[str(k).lower()] = b
            else:
                _walk_json(v, binds, f"{prefix}{k}.")
    elif isinstance(node, list):
        for item in node:
            _walk_json(item, binds, prefix)


def discover_for_game(game_id: str, exe_path: str = "") -> dict[str, str]:
    """Entry point used by the engine when a game with no stored binds appears."""
    if game_id.startswith("minecraft"):
        return discover_minecraft()
    game_dir = Path(exe_path).parent if exe_path else None
    merged: dict[str, str] = {}
    if game_dir:
        for f in find_bind_files(game_dir):
            merged.update(parse_bind_file(f))
    return merged
