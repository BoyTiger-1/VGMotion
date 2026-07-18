"""Offline heuristic engine: semantic inference and gesture suggestions with
no network/API required. Also the validator/fallback for AI-generated output.
"""
from __future__ import annotations

import re

# Canonical semantic actions the platform understands
SEMANTIC_ACTIONS = [
    "move_forward", "move_back", "move_left", "move_right",
    "jump", "crouch", "sneak", "sprint", "dodge",
    "attack", "shoot", "aim", "melee", "block", "parry",
    "use", "reload", "interact", "heal", "eat",
    "inventory", "map", "build", "mine", "place", "drop",
    "ability1", "ability2", "ability3", "ultimate", "grenade",
    "swap_hands", "pick_block", "camera", "click", "right_click", "drag", "pause",
]

# action-name regex -> semantic (for interpreting discovered keybind names)
_SEMANTIC_PATTERNS: list[tuple[str, str]] = [
    (r"forward|moveup|walk", "move_forward"),
    (r"back(ward)?|movedown", "move_back"),
    (r"strafe.?left|move.?left|^left$", "move_left"),
    (r"strafe.?right|move.?right|^right$", "move_right"),
    (r"jump", "jump"),
    (r"crouch|duck", "crouch"),
    (r"sneak", "sneak"),
    (r"sprint|run", "sprint"),
    (r"dodge|roll", "dodge"),
    (r"reload", "reload"),
    (r"aim|ads|scope|zoom", "aim"),
    (r"shoot|fire", "shoot"),
    (r"attack|primary", "attack"),
    (r"melee", "melee"),
    (r"block|parry|guard", "block"),
    (r"grenade|throw", "grenade"),
    (r"heal|medkit|bandage", "heal"),
    (r"eat|drink|consume", "eat"),
    (r"inventory|backpack", "inventory"),
    (r"^map$|minimap", "map"),
    (r"build", "build"),
    (r"mine|dig|harvest", "mine"),
    (r"place", "place"),
    (r"drop", "drop"),
    (r"interact|activate|pickup|pick.?up|(?<![a-z])use(?![a-z])", "interact"),
    (r"ability.?1|skill.?1", "ability1"),
    (r"ability.?2|skill.?2", "ability2"),
    (r"ability.?3|skill.?3", "ability3"),
    (r"ultimate|(?<![a-z])ult(?![a-z])", "ultimate"),
    (r"swap|switch", "swap_hands"),
]

# default input guesses when a semantic action has no discovered bind
DEFAULT_INPUTS = {
    "move_forward": ("key:w", "hold"), "move_back": ("key:s", "hold"),
    "move_left": ("key:a", "hold"), "move_right": ("key:d", "hold"),
    "jump": ("key:space", "tap"), "crouch": ("key:lctrl", "hold"),
    "sneak": ("key:lshift", "hold"), "sprint": ("key:lshift", "hold"),
    "dodge": ("key:lalt", "tap"),
    "attack": ("mouse:left", "tap"), "shoot": ("mouse:left", "hold_pulse"),
    "aim": ("mouse:right", "hold"), "melee": ("key:v", "tap"),
    "block": ("mouse:right", "hold"), "parry": ("key:q", "tap"),
    "use": ("mouse:right", "tap"), "reload": ("key:r", "tap"),
    "interact": ("key:e", "tap"), "heal": ("key:h", "tap"), "eat": ("mouse:right", "hold_pulse"),
    "inventory": ("key:e", "tap"), "map": ("key:m", "tap"),
    "build": ("key:q", "tap"), "mine": ("mouse:left", "hold_pulse"),
    "place": ("mouse:right", "tap"), "drop": ("key:q", "tap"),
    "ability1": ("key:1", "tap"), "ability2": ("key:2", "tap"),
    "ability3": ("key:3", "tap"), "ultimate": ("key:x", "tap"),
    "grenade": ("key:g", "tap"), "swap_hands": ("key:f", "tap"),
    "pick_block": ("mouse:middle", "tap"), "camera": ("key:f5", "tap"),
    "click": ("mouse:left", "tap"), "right_click": ("mouse:right", "tap"),
    "drag": ("mouse:left", "hold"), "pause": ("key:esc", "tap"),
}


def infer_semantics(discovered_binds: dict[str, str]) -> dict[str, str]:
    """{raw_action_name: binding} -> {semantic: binding}.
    Already-canonical names pass straight through."""
    out: dict[str, str] = {}
    for raw, binding in discovered_binds.items():
        if raw in SEMANTIC_ACTIONS:
            out.setdefault(raw, binding)
            continue
        # CamelCase -> snake_case so word-boundary lookarounds work ("UseItem"
        # matches 'use', "Pause"/"VaultOver" don't)
        norm = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", raw).lower()
        for pattern, semantic in _SEMANTIC_PATTERNS:
            if re.search(pattern, norm):
                out.setdefault(semantic, binding)
                break
    return out


# --------------------------------------------------------------------------
# Gesture suggestion templates per genre
# --------------------------------------------------------------------------
# gesture -> (semantic, rationale)
_MOVEMENT = {
    "walk": ("move_forward", "March in place to walk forward — the most natural full-body locomotion."),
    "sprint": ("sprint", "Speed up your marching cadence to sprint, just like real running."),
    "lean_left": ("move_left", "Lean your torso left to strafe left."),
    "lean_right": ("move_right", "Lean your torso right to strafe right."),
    "jump_in_place": ("jump", "Physically jumping is the most intuitive jump trigger."),
    "crouch": ("crouch", "Squatting mirrors crouching in-game one-to-one."),
}

GENRE_TEMPLATES: dict[str, dict[str, tuple[str, str]]] = {
    "fps": {
        **_MOVEMENT,
        "punch_right": ("shoot", "A sharp forward jab fires your weapon — fast to repeat under pressure."),
        "raise_arm_left": ("aim", "Raising your support arm mimics steadying a weapon to aim down sights."),
        "hand_to_chest": ("reload", "Reaching to your chest mirrors grabbing a fresh magazine from your vest."),
        "throw_right": ("grenade", "A real overhand throw lobs your grenade."),
        "hand_to_mouth": ("heal", "Bringing your hand to your mouth 'drinks' a health item."),
        "push": ("interact", "A two-handed shove opens doors and grabs objectives."),
        "kick_right": ("melee", "A front kick delivers your melee strike."),
    },
    "melee": {
        **_MOVEMENT,
        "swing_right_arm": ("attack", "A horizontal arm swing is a natural sword slash."),
        "chop_right": ("attack", "An overhead chop lands a heavy strike."),
        "block": ("block", "Raising both forearms guards, exactly like a real block."),
        "bow_draw": ("aim", "Drawing an imaginary bow aims your ranged weapon."),
        "punch_left": ("parry", "A quick off-hand jab parries."),
        "kick_right": ("dodge", "A kick shifts your weight — mapped to dodge/roll."),
        "hand_to_mouth": ("heal", "Hand to mouth drinks your healing flask."),
    },
    "sandbox": {
        **_MOVEMENT,
        "swing_right_arm": ("attack", "Swinging your arm attacks, just like swinging a sword."),
        "chop_right": ("mine", "Repeated overhead chops keep the mine button held — swing your pickaxe for real."),
        "push": ("place", "Pushing forward places a block in front of you."),
        "hand_to_chest": ("inventory", "Tapping your chest opens your inventory (checking your pockets)."),
        "hand_to_mouth": ("eat", "Hand to mouth eats the food you're holding."),
        "raise_arm_right": ("use", "Raising your item hand uses/places what you're holding."),
    },
    "pointer": {
        # right hand steers the cursor; clicking must not move it, so clicks
        # come from the left hand, a finger pinch, or dwell
        "punch_left": ("click", "Jab your left fist to click — your right hand keeps aiming, so the cursor never moves while you click. Holding the cursor still also dwell-clicks."),
        "pinch_right": ("drag", "Pinch your right thumb and index to grab (mouse held) — your wrist doesn't move at all; drag with the same hand, spread your fingers to drop."),
        "hand_to_chest": ("right_click", "Tap your chest with your left hand for a right-click."),
    },
    "platformer": {
        "lean_left": ("move_left", "Lean left to run left."),
        "lean_right": ("move_right", "Lean right to run right."),
        "jump_in_place": ("jump", "Jump for jump — the classic."),
        "crouch": ("crouch", "Squat to duck."),
        "punch_right": ("attack", "Jab to attack."),
        "walk": ("move_forward", "March in place to advance."),
    },
    "racing": {
        "lean_forward": ("move_forward", "Lean in to accelerate."),
        "lean_back": ("move_back", "Lean back to brake/reverse."),
        "lean_left": ("move_left", "Lean left to steer left."),
        "lean_right": ("move_right", "Lean right to steer right."),
        "jump_in_place": ("jump", "Hop for handbrake/boost."),
    },
    "generic": {
        **_MOVEMENT,
        "punch_right": ("attack", "Punch forward for your primary action."),
        "push": ("interact", "Push forward to use/interact."),
        "hand_to_chest": ("inventory", "Tap your chest to open menus/inventory."),
        "hand_to_mouth": ("heal", "Hand to mouth to heal/consume."),
    },
}

# per-accessibility substitutions applied over any template
_SEATED_OVERRIDES = {
    "jump_in_place": None,             # unavailable seated
    "crouch": None,
    "arms_up": ("jump", "Raise both arms to jump — no standing required."),
    "clap": ("crouch", "Clap to toggle crouch."),
}
_ONE_HANDED_RIGHT_DROPS = {"punch_left", "swing_left_arm", "chop_left", "throw_left",
                           "raise_arm_left", "kick_left", "push", "clap", "block", "bow_draw"}
_ONE_HANDED_LEFT_DROPS = {"punch_right", "swing_right_arm", "chop_right", "throw_right",
                          "raise_arm_right", "kick_right", "push", "clap", "block", "bow_draw"}


def suggest_gestures(genre: str, available_semantics: list[str],
                     accessibility: str = "standing") -> tuple[dict[str, str], dict[str, str]]:
    """Returns (gestures {gesture: semantic}, rationale {gesture: why})."""
    template = GENRE_TEMPLATES.get(genre or "generic", GENRE_TEMPLATES["generic"])
    working: dict[str, tuple[str, str] | None] = dict(template)

    if accessibility == "seated":
        for g, repl in _SEATED_OVERRIDES.items():
            if repl is None:
                working.pop(g, None)
            else:
                working[g] = repl
    elif accessibility == "one_handed_left":
        for g in _ONE_HANDED_LEFT_DROPS:
            working.pop(g, None)
        # mirror right-arm suggestions onto the left arm
        for g in list(working):
            if g.endswith("_right") or g == "swing_right_arm":
                mirrored = g.replace("right", "left")
                working[mirrored] = working.pop(g)
    elif accessibility == "one_handed_right":
        for g in _ONE_HANDED_RIGHT_DROPS:
            working.pop(g, None)

    gestures: dict[str, str] = {}
    rationale: dict[str, str] = {}
    have = set(available_semantics)
    for gesture, pair in working.items():
        if pair is None:
            continue
        semantic, why = pair
        if semantic in have and semantic not in gestures.values():
            gestures[gesture] = semantic
            rationale[gesture] = why
    return gestures, rationale
