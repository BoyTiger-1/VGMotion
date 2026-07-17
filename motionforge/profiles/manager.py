"""Profile persistence and matching.

A Profile stores everything needed to play one game: discovered keybinds,
semantic actions, gesture mappings with AI rationale, look/movement modes,
and sensitivities. Profiles live as JSON in ~/.motionforge/profiles as
`{id}__{variant}.json`; multiple variants per game (competitive, seated,
fitness, ...) are first-class.
"""
from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

from motionforge import config
from motionforge.core.events import GameInfo
from motionforge.inputs.actions import ActionBinding
from motionforge.profiles.defaults import build_default_profiles


@dataclass
class Profile:
    id: str = "generic"
    name: str = "Generic"
    variant: str = "default"
    match: dict = field(default_factory=lambda: {"processes": [], "titles": [],
                                                 "steam_appids": [], "browser_titles": []})
    genre: str = "generic"
    look_mode: str = "head"
    movement_mode: str = "walk_in_place"
    look_sensitivity: float = 1.0
    gesture_sensitivity: float = 1.0
    actions: dict = field(default_factory=dict)      # semantic -> ActionBinding dict
    gestures: dict = field(default_factory=dict)     # gesture id -> semantic
    rationale: dict = field(default_factory=dict)    # gesture id -> why
    discovered_binds: dict = field(default_factory=dict)
    source: str = "user"                              # bundled | offline | ai | user
    dwell_click: bool = False                         # cursor_hand: still cursor clicks

    def binding_for_gesture(self, gesture: str) -> tuple[str, ActionBinding] | None:
        semantic = self.gestures.get(gesture)
        if not semantic:
            return None
        action = self.actions.get(semantic)
        if not action:
            return None
        return semantic, ActionBinding.from_dict(action)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "name": self.name, "variant": self.variant,
            "match": self.match, "genre": self.genre, "look_mode": self.look_mode,
            "movement_mode": self.movement_mode,
            "look_sensitivity": self.look_sensitivity,
            "gesture_sensitivity": self.gesture_sensitivity,
            "actions": self.actions, "gestures": self.gestures,
            "rationale": self.rationale, "discovered_binds": self.discovered_binds,
            "source": self.source, "dwell_click": self.dwell_click,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Profile":
        p = cls()
        for k, v in d.items():
            if hasattr(p, k):
                setattr(p, k, v)
        return p

    def matches(self, info: GameInfo) -> bool:
        m = self.match or {}
        pname = (info.process or "").lower()
        title = (info.window_title or "").lower()
        if pname and pname in [x.lower() for x in m.get("processes", [])]:
            if self.id.startswith("minecraft") and pname in ("javaw.exe", "java.exe"):
                return "minecraft" in title
            return True
        if info.steam_appid and info.steam_appid in m.get("steam_appids", []):
            return True
        if info.is_browser and any(t in title for t in m.get("browser_titles", [])):
            return True
        non_browser_titles = m.get("titles", [])
        if not info.is_browser and title and any(t in title for t in non_browser_titles):
            return True
        return False


def _safe_name(s: str) -> str:
    return re.sub(r"[^a-z0-9_\-]", "_", s.lower())


class ProfileManager:
    def __init__(self, profile_dir: Path | None = None, settings: config.Settings | None = None):
        self.dir = profile_dir or config.PROFILE_DIR
        self.settings = settings
        self._lock = threading.RLock()
        self._profiles: dict[tuple[str, str], Profile] = {}   # (id, variant) -> Profile
        self.dir.mkdir(parents=True, exist_ok=True)
        self._ensure_defaults()
        self.reload()

    # -- storage -----------------------------------------------------------

    def _path(self, pid: str, variant: str) -> Path:
        return self.dir / f"{_safe_name(pid)}__{_safe_name(variant)}.json"

    def _ensure_defaults(self) -> None:
        for d in build_default_profiles():
            path = self._path(d["id"], d.get("variant", "default"))
            if not path.exists():
                path.write_text(json.dumps(d, indent=2), encoding="utf-8")
                continue
            # refresh untouched bundled profiles so shipped fixes propagate;
            # anything the user saved (source == "user") is never overwritten
            try:
                stored = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                stored = None
            if stored and stored.get("source") == "bundled" and stored != d:
                path.write_text(json.dumps(d, indent=2), encoding="utf-8")

    def reload(self) -> None:
        with self._lock:
            self._profiles.clear()
            for f in self.dir.glob("*.json"):
                try:
                    p = Profile.from_dict(json.loads(f.read_text(encoding="utf-8")))
                except (json.JSONDecodeError, OSError):
                    continue
                if self._migrate(p):
                    self._path(p.id, p.variant).write_text(
                        json.dumps(p.to_dict(), indent=2), encoding="utf-8")
                self._profiles[(p.id, p.variant)] = p

    def _migrate(self, p: Profile) -> bool:
        """Upgrade auto-derived profiles created by older builds. Pointer-genre
        profiles regenerate from the current pointer template (the old
        push-to-click scheme couldn't click reliably)."""
        if p.genre != "pointer" or p.source == "user" or p.source == "bundled":
            return False
        needs = (not p.dwell_click or "drag" not in p.actions
                 or p.gestures.get("push") == "click")
        if not needs:
            return False
        template = next((d for d in build_default_profiles() if d["id"] == "pointer"), None)
        if template:
            p.actions = dict(template["actions"])
            p.gestures = dict(template["gestures"])
            p.rationale = dict(template["rationale"])
            p.look_mode = template["look_mode"]
            p.movement_mode = template["movement_mode"]
        p.dwell_click = True
        return True

    def save(self, profile: Profile) -> None:
        with self._lock:
            self._profiles[(profile.id, profile.variant)] = profile
            self._path(profile.id, profile.variant).write_text(
                json.dumps(profile.to_dict(), indent=2), encoding="utf-8")

    def delete(self, profile: Profile) -> None:
        with self._lock:
            self._profiles.pop((profile.id, profile.variant), None)
            path = self._path(profile.id, profile.variant)
            if path.exists():
                path.unlink()

    def all(self) -> list[Profile]:
        with self._lock:
            return sorted(self._profiles.values(), key=lambda p: (p.name, p.variant))

    def variants(self, pid: str) -> list[Profile]:
        with self._lock:
            return sorted((p for (i, _), p in self._profiles.items() if i == pid),
                          key=lambda p: p.variant)

    def get(self, pid: str, variant: str = "default") -> Profile | None:
        with self._lock:
            return self._profiles.get((pid, variant))

    # -- matching ------------------------------------------------------------

    def _preferred_variant(self, pid: str) -> str:
        if self.settings:
            return self.settings.get("last_variant", {}).get(pid, "default")
        return "default"

    def remember_variant(self, pid: str, variant: str) -> None:
        if self.settings:
            lv = self.settings.get("last_variant", {})
            lv[pid] = variant
            self.settings.set("last_variant", lv)

    def for_game(self, info: GameInfo) -> Profile:
        """Best profile for a detected game; creates a per-game profile from
        the genre template on first encounter with an unknown game."""
        with self._lock:
            # 1) exact id (strip source prefixes like "browser:")
            bare_id = info.id.split(":", 1)[-1] if info.id else ""
            for pid in (info.id, bare_id):
                if pid:
                    p = self.get(pid, self._preferred_variant(pid))
                    if p:
                        return p
            # 2) match rules
            candidates = [p for p in self._profiles.values()
                          if p.variant == "default" and p.matches(info)]
            if candidates:
                best = candidates[0]
                return self.get(best.id, self._preferred_variant(best.id)) or best
            # 3) knowledge-base profile hint / genre template
            template_id = info.genre if info.genre in (
                "fps", "melee", "sandbox", "platformer", "racing", "pointer") else "generic"
            template = self.get(template_id) or self.get("generic")
            if not info.id or not info.is_game:
                return template
            # 4) derive and persist a per-game profile so edits stick
            derived = Profile.from_dict(template.to_dict())
            derived.id = bare_id or _safe_name(info.name)
            derived.name = info.name
            derived.variant = "default"
            derived.source = "offline"
            derived.match = {"processes": [info.process] if info.process and not info.is_browser else [],
                             "titles": [], "steam_appids": [info.steam_appid] if info.steam_appid else [],
                             "browser_titles": [info.name.lower()] if info.is_browser else []}
            self.save(derived)
            return derived

    def duplicate_as_variant(self, profile: Profile, variant: str) -> Profile:
        copy = Profile.from_dict(profile.to_dict())
        copy.variant = _safe_name(variant)
        copy.source = "user"
        self.save(copy)
        return copy
