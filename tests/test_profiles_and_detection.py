"""Profile matching/persistence and game-detection knowledge base tests."""
import json
from pathlib import Path

import pytest

from motionforge.core.events import GameInfo
from motionforge.detection import knowledgebase as kb
from motionforge.profiles.manager import Profile, ProfileManager


@pytest.fixture
def pm(tmp_path):
    return ProfileManager(profile_dir=tmp_path / "profiles")


def test_defaults_materialized(pm):
    ids = {p.id for p in pm.all()}
    assert {"minecraft", "roblox", "krunker", "fps", "melee", "pointer", "generic"} <= ids


def test_minecraft_match_requires_title_for_javaw(pm):
    mc = pm.for_game(GameInfo(id="", name="Minecraft", process="javaw.exe",
                              window_title="Minecraft 1.21.4"))
    assert mc.id == "minecraft"
    other = pm.for_game(GameInfo(id="", name="IDE", process="javaw.exe",
                                 window_title="IntelliJ IDEA", is_game=False))
    assert other.id != "minecraft"


def test_unknown_game_derives_profile(pm):
    info = GameInfo(id="steam:9999", name="Mystery Shooter", process="mystery.exe",
                    genre="fps", steam_appid="9999")
    p = pm.for_game(info)
    assert p.id == "9999" and p.genre == "fps" and p.gestures
    # second detection loads the persisted profile, not a new derivation
    p2 = pm.for_game(info)
    assert p2.id == p.id and (pm.dir / "9999__default.json").exists()


def test_variants(pm):
    base = pm.get("minecraft")
    seated = pm.duplicate_as_variant(base, "seated")
    assert seated.variant == "seated"
    assert {v.variant for v in pm.variants("minecraft")} == {"default", "seated"}


def test_profile_binding_lookup(pm):
    mc = pm.get("minecraft")
    semantic, binding = mc.binding_for_gesture("chop_right")
    assert semantic == "mine" and binding.input == "mouse:left" and binding.mode == "hold_pulse"
    assert mc.binding_for_gesture("no_such_gesture") is None


def test_knowledgebase_matching():
    assert kb.match_process("robloxplayerbeta.exe", "Roblox").id == "roblox"
    assert kb.match_process("cs2.exe", "Counter-Strike 2").id == "cs2"
    assert kb.match_process("chrome.exe", "whatever") is None
    assert kb.match_browser_title("KRUNKER.IO - Chrome").id == "krunker"
    assert kb.match_browser_title("Play Chess Online - Chess.com - Google Chrome").id == "chess"
    assert kb.match_browser_title("BBC News") is None
    assert kb.browser_title_looks_like_game("Subway Surfers - Poki - Chrome")
    assert kb.match_steam_appid("730").id == "cs2"


def test_corrupt_profile_ignored(tmp_path):
    d = tmp_path / "profiles"
    d.mkdir()
    (d / "bad__default.json").write_text("{not json", encoding="utf-8")
    pm = ProfileManager(profile_dir=d)
    assert all(p.id != "bad" for p in pm.all())
