"""Control discovery, offline semantics, gesture suggestion, and AI plumbing."""
import json

from motionforge.ai import offline
from motionforge.ai.gemini import extract_json
from motionforge.controls import discovery


def test_minecraft_options_parsing(tmp_path):
    (tmp_path / "options.txt").write_text(
        "version:3955\n"
        "key_key.jump:key.keyboard.space\n"
        "key_key.sneak:key.keyboard.left.shift\n"
        "key_key.sprint:key.keyboard.left.control\n"
        "key_key.forward:key.keyboard.w\n"
        "key_key.attack:key.mouse.left\n"
        "key_key.use:key.mouse.right\n"
        "key_key.inventory:key.keyboard.e\n"
        "key_key.drop:key.keyboard.q\n"
        "fov:0.0\n", encoding="utf-8")
    binds = discovery.discover_minecraft(tmp_path)
    assert binds == {
        "jump": "key:space", "sneak": "key:lshift", "sprint": "key:lctrl",
        "move_forward": "key:w", "attack": "mouse:left", "use": "mouse:right",
        "inventory": "key:e", "drop": "key:q"}


def test_generic_bind_file_json(tmp_path):
    f = tmp_path / "keybindings.json"
    f.write_text(json.dumps({"Jump": "SPACE", "Fire": "MOUSE1", "Reload": "R",
                             "Volume": "0.5"}), encoding="utf-8")
    binds = discovery.parse_bind_file(f)
    assert binds["jump"] == "key:space"
    assert binds["fire"] == "mouse:left"
    assert binds["reload"] == "key:r"


def test_generic_bind_file_ini(tmp_path):
    f = tmp_path / "input.ini"
    f.write_text("Jump=Space\nCrouch=LCtrl\nSprint = LShift\nname=Bob\n", encoding="utf-8")
    binds = discovery.parse_bind_file(f)
    assert binds["jump"] == "key:space"
    assert binds["crouch"] == "key:lctrl"
    assert "name" not in binds


def test_semantic_inference():
    sem = offline.infer_semantics({
        "jump": "key:space", "PrimaryFire": "mouse:left", "QuickMelee": "key:v",
        "StrafeLeft": "key:a", "unrelated_thing": "key:p"})
    assert sem["jump"] == "key:space"
    assert sem["shoot"] == "mouse:left"     # PrimaryFire -> shoot
    assert sem["melee"] == "key:v"
    assert sem["move_left"] == "key:a"


def test_semantic_inference_word_boundaries():
    sem = offline.infer_semantics({"VaultOver": "key:space", "Pause": "key:esc",
                                   "UseItem": "key:e", "Ult": "key:x"})
    assert sem.get("ultimate") == "key:x"          # 'Ult' matches, 'VaultOver' must not
    assert sem.get("interact") == "key:e"          # 'UseItem' matches, 'Pause' must not
    assert "key:esc" not in sem.values()


def test_suggestions_cover_and_respect_accessibility():
    semantics = ["move_forward", "jump", "shoot", "reload", "aim", "heal"]
    g, r = offline.suggest_gestures("fps", semantics)
    assert g["walk"] == "move_forward" and r["walk"]
    assert set(g.values()) <= set(semantics)
    assert len(set(g.values())) == len(g.values())          # one gesture per action
    seated, _ = offline.suggest_gestures("fps", semantics, "seated")
    assert "jump_in_place" not in seated and seated.get("arms_up") == "jump"
    one_left, _ = offline.suggest_gestures("fps", semantics, "one_handed_left")
    assert all("right" not in gesture for gesture in one_left), one_left


def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Here you go:\n{"a": {"b": 2}}\nEnjoy!') == {"a": {"b": 2}}
    assert extract_json("no json here") is None


def test_preference_learning(tmp_path):
    from motionforge.ai.reasoning import PreferenceStore
    store = PreferenceStore(tmp_path / "prefs.json")
    store.record_choice("shoot", "clap")
    assert store.preferred_gesture("shoot") is None      # one vote isn't a pattern
    store.record_choice("shoot", "clap")
    assert store.preferred_gesture("shoot") == "clap"
    # survives reload
    store2 = PreferenceStore(tmp_path / "prefs.json")
    assert store2.preferred_gesture("shoot") == "clap"
