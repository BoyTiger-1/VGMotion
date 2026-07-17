"""AI reasoning engine: Gemini-backed game identification, semantic action
inference, and gesture mapping generation — always validated against the
gesture/semantic vocabularies and always with an offline fallback.

Also learns player preferences: when the user remaps a suggested gesture, the
replacement is recorded and biases every future suggestion (offline and AI).
"""
from __future__ import annotations

import io
import json
import threading
from pathlib import Path

from motionforge.ai import offline
from motionforge.ai.gemini import GeminiClient
from motionforge.core.events import GameInfo
from motionforge.gestures.library import GESTURE_DESCRIPTIONS, STATE_GESTURES


class PreferenceStore:
    """Persists 'the user replaced gesture X with gesture Y for semantic S'."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._data: dict = {"gesture_for_semantic": {}}
        if path.exists():
            try:
                self._data.update(json.loads(path.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError):
                pass

    def record_choice(self, semantic: str, gesture: str) -> None:
        with self._lock:
            counts = self._data["gesture_for_semantic"].setdefault(semantic, {})
            counts[gesture] = counts.get(gesture, 0) + 1
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")

    def preferred_gesture(self, semantic: str) -> str | None:
        with self._lock:
            counts = self._data["gesture_for_semantic"].get(semantic, {})
        if not counts:
            return None
        best = max(counts.items(), key=lambda kv: kv[1])
        return best[0] if best[1] >= 2 else None   # need 2+ confirmations

    def summary(self) -> str:
        with self._lock:
            prefs = self._data["gesture_for_semantic"]
        lines = []
        for sem, counts in prefs.items():
            top = max(counts.items(), key=lambda kv: kv[1])
            if top[1] >= 2:
                lines.append(f"- for '{sem}' the player prefers the '{top[0]}' gesture")
        return "\n".join(lines)


class AIReasoner:
    def __init__(self, client: GeminiClient | None, prefs: PreferenceStore):
        self.client = client
        self.prefs = prefs

    @property
    def online(self) -> bool:
        return self.client is not None

    # -- game identification from a screenshot ------------------------------

    def identify_game(self, screenshot, window_title: str) -> GameInfo | None:
        """screenshot: PIL Image or None. Returns GameInfo or None."""
        if not self.client:
            return None
        jpeg = None
        if screenshot is not None:
            buf = io.BytesIO()
            screenshot.convert("RGB").resize(
                (min(screenshot.width, 1024),
                 max(1, int(screenshot.height * min(screenshot.width, 1024) / screenshot.width)))
            ).save(buf, format="JPEG", quality=70)
            jpeg = buf.getvalue()
        prompt = (
            "You identify PC games from a window screenshot and title.\n"
            f"Window title: {window_title!r}\n"
            "Reply with JSON only: {\"is_game\": bool, \"name\": str, "
            "\"genre\": one of [\"fps\",\"melee\",\"sandbox\",\"platformer\","
            "\"racing\",\"pointer\",\"generic\"], \"confidence\": 0..1}.\n"
            "is_game=false for desktop apps, IDEs, video players, documents. "
            "Use \"pointer\" for mouse-driven games (chess, puzzles, strategy)."
        )
        data = self.client.generate_json(prompt, jpeg)
        if not data or not isinstance(data, dict):
            return None
        if not data.get("is_game") or float(data.get("confidence", 0)) < 0.5:
            return None
        genre = data.get("genre", "generic")
        if genre not in offline.GENRE_TEMPLATES:
            genre = "generic"
        name = str(data.get("name", window_title))[:80]
        return GameInfo(id=f"ai:{name.lower().replace(' ', '_')[:40]}", name=name,
                        window_title=window_title, source="ai", genre=genre)

    # -- semantic inference ---------------------------------------------------

    def infer_semantics(self, game: GameInfo, discovered: dict[str, str]) -> dict[str, str]:
        """{raw_bind_name: binding} -> {semantic: binding}. Offline first; the
        AI only gets raw names heuristics could not classify."""
        result = offline.infer_semantics(discovered)
        classified_bindings = set(result.values())
        leftovers = {k: v for k, v in discovered.items()
                     if v not in classified_bindings and k not in offline.SEMANTIC_ACTIONS}
        if leftovers and self.client:
            prompt = (
                f"Game: {game.name} (genre: {game.genre or 'unknown'}).\n"
                "These raw keybinding names come from the game's config file. Map each to one "
                f"semantic action from this list: {offline.SEMANTIC_ACTIONS}.\n"
                f"Raw names: {list(leftovers.keys())}\n"
                'Reply JSON only: {"mapping": {"raw_name": "semantic_action", ...}} '
                "and omit raw names that fit nothing."
            )
            data = self.client.generate_json(prompt)
            if data and isinstance(data.get("mapping"), dict):
                for raw, semantic in data["mapping"].items():
                    if raw in leftovers and semantic in offline.SEMANTIC_ACTIONS:
                        result.setdefault(semantic, leftovers[raw])
        return result

    # -- gesture mapping generation -------------------------------------------

    def suggest_mappings(self, game: GameInfo, semantics: list[str],
                         accessibility: str = "standing"
                         ) -> tuple[dict[str, str], dict[str, str], str]:
        """Returns (gestures, rationale, source). Applies learned preferences."""
        gestures, rationale = offline.suggest_gestures(game.genre or "generic",
                                                       semantics, accessibility)
        source = "offline"
        if self.client:
            ai = self._suggest_ai(game, semantics, accessibility)
            if ai:
                gestures, rationale = ai
                source = "ai"
        gestures, rationale = self._apply_preferences(gestures, rationale)
        return gestures, rationale, source

    def _suggest_ai(self, game: GameInfo, semantics: list[str], accessibility: str):
        vocab = {g: d for g, d in GESTURE_DESCRIPTIONS.items() if g != "t_pose"}
        prefs_note = self.prefs.summary()
        prompt = (
            "You design intuitive full-body webcam motion controls for PC games.\n"
            f"Game: {game.name} | genre: {game.genre or 'unknown'} | player mode: {accessibility}\n"
            f"Game actions to cover (most important first): {semantics}\n"
            "Available gestures (id: description):\n"
            + "\n".join(f"  {g}: {d}" for g, d in vocab.items()) + "\n"
            + (f"Player preferences learned from past sessions:\n{prefs_note}\n" if prefs_note else "")
            + "\nRules: gestures marked (held) suit held actions (movement, aim, block, "
            "crouch); momentary gestures suit one-shot actions. Each gesture maps to at "
            "most one action. Prefer physically mimetic mappings (swing=melee, "
            "throw=grenade, hand to mouth=eat/heal). Cover movement and the core "
            "actions first.\n"
            'Reply JSON only: {"gestures": {"gesture_id": "action"}, '
            '"rationale": {"gesture_id": "one short sentence"}}'
        )
        data = self.client.generate_json(prompt)
        if not data or not isinstance(data.get("gestures"), dict):
            return None
        valid_g: dict[str, str] = {}
        valid_r: dict[str, str] = {}
        have = set(semantics)
        for g, sem in data["gestures"].items():
            if g in vocab and sem in have and sem not in valid_g.values():
                valid_g[g] = sem
                valid_r[g] = str(data.get("rationale", {}).get(g, ""))[:200]
        # require a reasonable mapping, else fall back to offline
        return (valid_g, valid_r) if len(valid_g) >= min(3, len(have)) else None

    def _apply_preferences(self, gestures: dict[str, str], rationale: dict[str, str]):
        for semantic in set(gestures.values()):
            pref = self.prefs.preferred_gesture(semantic)
            if not pref or gestures.get(pref) == semantic or pref not in GESTURE_DESCRIPTIONS:
                continue
            # don't map a held action to a momentary gesture
            current = [g for g, s in gestures.items() if s == semantic][0]
            if (current in STATE_GESTURES) != (pref in STATE_GESTURES):
                continue
            del gestures[current]
            rationale.pop(current, None)
            gestures[pref] = semantic
            rationale[pref] = "Chosen because you've preferred this gesture before."
        return gestures, rationale
