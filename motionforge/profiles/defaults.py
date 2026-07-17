"""Bundled default profiles, materialized into the user profile dir on first
run. Genre templates are generated from the offline heuristic engine so the
template and the fallback logic can never drift apart; the requested test
games (Minecraft, Roblox, Krunker, pointer games) get hand-tuned profiles.
"""
from __future__ import annotations

from motionforge.ai import offline


def _genre_profile(genre: str, look_mode: str = "head") -> dict:
    semantics = list(offline.GENRE_TEMPLATES[genre].keys())
    wanted = {sem for sem, _ in offline.GENRE_TEMPLATES[genre].values()}
    actions = {}
    for sem in wanted:
        inp, mode = offline.DEFAULT_INPUTS.get(sem, ("none", "tap"))
        actions[sem] = {"input": inp, "mode": mode, "hold_ms": 600}
    gestures, rationale = offline.suggest_gestures(genre, list(wanted))
    return {
        "id": genre, "name": f"{genre.title()} (template)", "variant": "default",
        "match": {"processes": [], "titles": [], "steam_appids": [], "browser_titles": []},
        "genre": genre, "look_mode": look_mode, "movement_mode": "walk_in_place",
        "look_sensitivity": 1.0, "gesture_sensitivity": 1.0,
        "actions": actions, "gestures": gestures, "rationale": rationale,
        "discovered_binds": {}, "source": "bundled",
        "dwell_click": genre == "pointer",
    }


def build_default_profiles() -> list[dict]:
    profiles: list[dict] = []

    # ---- genre templates -------------------------------------------------
    profiles.append(_genre_profile("fps"))
    profiles.append(_genre_profile("melee"))
    profiles.append(_genre_profile("sandbox"))
    profiles.append(_genre_profile("platformer", look_mode="off"))
    profiles.append(_genre_profile("racing", look_mode="off"))
    profiles.append(_genre_profile("generic"))
    pointer = _genre_profile("pointer", look_mode="cursor_hand")
    pointer["movement_mode"] = "off"
    profiles.append(pointer)

    # ---- Minecraft (Java + Bedrock) --------------------------------------
    profiles.append({
        "id": "minecraft", "name": "Minecraft", "variant": "default",
        "match": {"processes": ["javaw.exe", "minecraft.windows.exe"],
                  "titles": ["minecraft"], "steam_appids": [], "browser_titles": []},
        "genre": "sandbox", "look_mode": "head", "movement_mode": "walk_in_place",
        "look_sensitivity": 1.0, "gesture_sensitivity": 1.0,
        "actions": {
            "move_forward": {"input": "key:w", "mode": "hold", "hold_ms": 600},
            "move_left":    {"input": "key:a", "mode": "hold", "hold_ms": 600},
            "move_right":   {"input": "key:d", "mode": "hold", "hold_ms": 600},
            "sprint":       {"input": "key:lctrl", "mode": "hold", "hold_ms": 600},
            "jump":         {"input": "key:space", "mode": "tap", "hold_ms": 600},
            "sneak":        {"input": "key:lshift", "mode": "hold", "hold_ms": 600},
            "attack":       {"input": "mouse:left", "mode": "tap", "hold_ms": 600},
            "mine":         {"input": "mouse:left", "mode": "hold_pulse", "hold_ms": 700},
            "use":          {"input": "mouse:right", "mode": "tap", "hold_ms": 600},
            "place":        {"input": "mouse:right", "mode": "tap", "hold_ms": 600},
            "eat":          {"input": "mouse:right", "mode": "hold_pulse", "hold_ms": 1800},
            "inventory":    {"input": "key:e", "mode": "tap", "hold_ms": 600},
            "drop":         {"input": "key:q", "mode": "tap", "hold_ms": 600},
            "swap_hands":   {"input": "key:f", "mode": "tap", "hold_ms": 600},
        },
        "gestures": {
            "walk": "move_forward", "sprint": "sprint",
            "lean_left": "move_left", "lean_right": "move_right",
            "jump_in_place": "jump", "crouch": "sneak",
            "swing_right_arm": "attack", "chop_right": "mine",
            "push": "place", "raise_arm_right": "use",
            "hand_to_mouth": "eat", "hand_to_chest": "inventory",
            "throw_right": "drop", "swing_left_arm": "swap_hands",
        },
        "rationale": {
            "walk": "March in place to walk forward.",
            "sprint": "March faster to sprint.",
            "lean_left": "Lean left to strafe left.",
            "lean_right": "Lean right to strafe right.",
            "jump_in_place": "Jump to jump — one to one.",
            "crouch": "Squat to sneak.",
            "swing_right_arm": "A horizontal swing attacks mobs like a sword slash.",
            "chop_right": "Overhead chops swing your pickaxe; keep chopping to keep mining.",
            "push": "Push your palms forward to place a block in front of you.",
            "raise_arm_right": "Raise your item hand to use what you're holding.",
            "hand_to_mouth": "Hand to mouth eats your held food (hold is auto-timed).",
            "hand_to_chest": "Tap your chest to check your inventory.",
            "throw_right": "A throwing motion drops/tosses your held item.",
            "swing_left_arm": "Off-hand swing swaps main/off hand.",
        },
        "discovered_binds": {}, "source": "bundled",
    })

    # ---- Roblox ------------------------------------------------------------
    profiles.append({
        "id": "roblox", "name": "Roblox", "variant": "default",
        "match": {"processes": ["robloxplayerbeta.exe"], "titles": ["roblox"],
                  "steam_appids": [], "browser_titles": []},
        "genre": "generic", "look_mode": "head", "movement_mode": "walk_in_place",
        "look_sensitivity": 1.0, "gesture_sensitivity": 1.0,
        "actions": {
            "move_forward": {"input": "key:w", "mode": "hold", "hold_ms": 600},
            "move_left":    {"input": "key:a", "mode": "hold", "hold_ms": 600},
            "move_right":   {"input": "key:d", "mode": "hold", "hold_ms": 600},
            "jump":         {"input": "key:space", "mode": "tap", "hold_ms": 600},
            "sprint":       {"input": "key:lshift", "mode": "hold", "hold_ms": 600},
            "attack":       {"input": "mouse:left", "mode": "tap", "hold_ms": 600},
            "use":          {"input": "mouse:right", "mode": "tap", "hold_ms": 600},
            "interact":     {"input": "key:e", "mode": "tap", "hold_ms": 600},
            "inventory":    {"input": "key:grave", "mode": "tap", "hold_ms": 600},
        },
        "gestures": {
            "walk": "move_forward", "sprint": "sprint",
            "lean_left": "move_left", "lean_right": "move_right",
            "jump_in_place": "jump",
            "punch_right": "attack", "raise_arm_right": "use",
            "push": "interact", "hand_to_chest": "inventory",
        },
        "rationale": {
            "walk": "March in place to walk.",
            "sprint": "March faster to sprint.",
            "lean_left": "Lean left to strafe left.",
            "lean_right": "Lean right to strafe right.",
            "jump_in_place": "Jump to jump.",
            "punch_right": "Punch to click/attack — works across most Roblox experiences.",
            "raise_arm_right": "Raise your right hand for the secondary action.",
            "push": "Push forward to interact (E).",
            "hand_to_chest": "Tap your chest to open your backpack.",
        },
        "discovered_binds": {}, "source": "bundled",
    })

    # ---- Krunker.io (browser FPS) -------------------------------------------
    krunker = _genre_profile("fps")
    krunker.update({
        "id": "krunker", "name": "Krunker.io",
        "match": {"processes": [], "titles": [], "steam_appids": [],
                  "browser_titles": ["krunker"]},
        "source": "bundled",
    })
    krunker["actions"]["crouch"] = {"input": "key:lshift", "mode": "hold", "hold_ms": 600}
    profiles.append(krunker)

    return profiles
