"""Built-in knowledge of games, launchers, and browsers.

This is the offline backbone of game detection: process names, window-title
fragments, Steam app ids, and browser-game URL/title fragments for popular
titles across Steam, Epic, Xbox PC, Riot, Battle.net, Ubisoft, Roblox,
Minecraft, browser games, and emulators. Unknown titles fall through to the
AI screenshot identifier (when enabled) or a generic template.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class KnownGame:
    id: str
    name: str
    genre: str                       # fps | melee | sandbox | platformer | racing |
                                     # pointer | sports | fighting | generic
    processes: list = field(default_factory=list)     # lowercase exe names
    titles: list = field(default_factory=list)        # lowercase window-title fragments
    steam_appids: list = field(default_factory=list)
    browser: bool = False            # matched inside a browser tab
    profile: str = ""                # bundled profile id, if any


KNOWN_GAMES: list[KnownGame] = [
    # -- test targets requested by the user --------------------------------
    KnownGame("minecraft", "Minecraft (Java)", "sandbox",
              processes=["javaw.exe"], titles=["minecraft"], profile="minecraft"),
    KnownGame("minecraft_bedrock", "Minecraft (Bedrock)", "sandbox",
              processes=["minecraft.windows.exe"], titles=["minecraft"], profile="minecraft"),
    KnownGame("roblox", "Roblox", "generic",
              processes=["robloxplayerbeta.exe", "windows10universal.exe"],
              titles=["roblox"], profile="roblox"),
    KnownGame("krunker", "Krunker.io", "fps",
              titles=["krunker"], browser=True, profile="krunker"),
    KnownGame("chess", "Chess.com", "pointer",
              titles=["chess.com", "play chess"], browser=True, profile="pointer"),
    KnownGame("geoguessr", "GeoGuessr", "pointer",
              titles=["geoguessr"], browser=True, profile="pointer"),
    # -- popular PC titles ---------------------------------------------------
    KnownGame("cs2", "Counter-Strike 2", "fps",
              processes=["cs2.exe"], steam_appids=["730"], profile="fps"),
    KnownGame("valorant", "VALORANT", "fps",
              processes=["valorant.exe", "valorant-win64-shipping.exe"], profile="fps"),
    KnownGame("fortnite", "Fortnite", "fps",
              processes=["fortniteclient-win64-shipping.exe"], profile="fps"),
    KnownGame("apex", "Apex Legends", "fps",
              processes=["r5apex.exe"], steam_appids=["1172470"], profile="fps"),
    KnownGame("overwatch", "Overwatch 2", "fps",
              processes=["overwatch.exe"], profile="fps"),
    KnownGame("cod", "Call of Duty", "fps",
              processes=["cod.exe", "modernwarfare.exe"], profile="fps"),
    KnownGame("tf2", "Team Fortress 2", "fps",
              processes=["tf_win64.exe", "hl2.exe"], steam_appids=["440"], profile="fps"),
    KnownGame("doom", "DOOM", "fps",
              processes=["doometernalx64vk.exe", "doomx64.exe"], profile="fps"),
    KnownGame("halo", "Halo Infinite", "fps",
              processes=["haloinfinite.exe"], profile="fps"),
    KnownGame("destiny2", "Destiny 2", "fps",
              processes=["destiny2.exe"], profile="fps"),
    KnownGame("terraria", "Terraria", "platformer",
              processes=["terraria.exe"], steam_appids=["105600"]),
    KnownGame("stardew", "Stardew Valley", "pointer",
              processes=["stardew valley.exe", "stardewvalley.exe"], profile="pointer"),
    KnownGame("rocketleague", "Rocket League", "racing",
              processes=["rocketleague.exe"]),
    KnownGame("gta5", "Grand Theft Auto V", "generic",
              processes=["gta5.exe"], steam_appids=["271590"]),
    KnownGame("eldenring", "Elden Ring", "melee",
              processes=["eldenring.exe"], steam_appids=["1245620"], profile="melee"),
    KnownGame("skyrim", "Skyrim", "melee",
              processes=["skyrimse.exe", "tesv.exe"], profile="melee"),
    KnownGame("mordhau", "Mordhau", "melee",
              processes=["mordhau-win64-shipping.exe"], profile="melee"),
    KnownGame("chivalry2", "Chivalry 2", "melee",
              processes=["chivalry2-win64-shipping.exe"], profile="melee"),
    KnownGame("beamng", "BeamNG.drive", "racing",
              processes=["beamng.drive.x64.exe"]),
    KnownGame("leagueoflegends", "League of Legends", "pointer",
              processes=["league of legends.exe"], profile="pointer"),
    # -- browser games -------------------------------------------------------
    KnownGame("slither", "Slither.io", "pointer", titles=["slither.io"],
              browser=True, profile="pointer"),
    KnownGame("agar", "Agar.io", "pointer", titles=["agar.io"],
              browser=True, profile="pointer"),
    KnownGame("1v1lol", "1v1.LOL", "fps", titles=["1v1.lol"],
              browser=True, profile="fps"),
    KnownGame("shellshock", "Shell Shockers", "fps", titles=["shell shockers", "shellshock.io"],
              browser=True, profile="fps"),
    KnownGame("venge", "Venge.io", "fps", titles=["venge.io"], browser=True, profile="fps"),
    KnownGame("smashkarts", "Smash Karts", "racing", titles=["smash karts", "smashkarts"],
              browser=True),
    KnownGame("wordle", "Wordle", "pointer", titles=["wordle"], browser=True, profile="pointer"),
    # -- emulators (treated as games; profile per-core is future work) -------
    KnownGame("retroarch", "RetroArch", "generic", processes=["retroarch.exe"]),
    KnownGame("dolphin", "Dolphin Emulator", "generic", processes=["dolphin.exe"]),
    KnownGame("pcsx2", "PCSX2", "generic", processes=["pcsx2.exe", "pcsx2-qt.exe"]),
    KnownGame("cemu", "Cemu", "generic", processes=["cemu.exe"]),
    KnownGame("yuzu", "yuzu/suyu", "generic", processes=["yuzu.exe", "suyu.exe"]),
]

BROWSER_PROCESSES = {
    "chrome.exe", "msedge.exe", "firefox.exe", "brave.exe", "opera.exe",
    "opera_gx.exe", "vivaldi.exe", "arc.exe",
}

# Foreground apps that are definitely not games (skip AI identification)
NON_GAME_PROCESSES = {
    "explorer.exe", "code.exe", "devenv.exe", "notepad.exe", "notepad++.exe",
    "discord.exe", "slack.exe", "spotify.exe", "steam.exe", "steamwebhelper.exe",
    "epicgameslauncher.exe", "riotclientservices.exe", "battle.net.exe",
    "ubisoftconnect.exe", "upc.exe", "galaxyclient.exe", "playnite.desktopapp.exe",
    "obs64.exe", "taskmgr.exe", "systemsettings.exe", "powershell.exe", "cmd.exe",
    "windowsterminal.exe", "wt.exe", "python.exe", "pythonw.exe", "javaws.exe",
    "excel.exe", "winword.exe", "powerpnt.exe", "outlook.exe", "onenote.exe",
    "acrobat.exe", "sumatrapdf.exe", "vlc.exe", "mpc-hc64.exe",
}

# Window-title fragments that suggest a browser tab is a game even when the
# specific title is unknown (e.g. hosted game portals)
BROWSER_GAME_HINTS = [
    "poki", "crazygames", "coolmathgames", "itch.io", "armor games",
    "kongregate", "newgrounds", "miniclip", "y8", "friv", ".io game",
]


def match_process(process: str, title: str) -> KnownGame | None:
    process = (process or "").lower()
    title = (title or "").lower()
    for g in KNOWN_GAMES:
        if g.browser:
            continue
        if process in (p.lower() for p in g.processes):
            # javaw.exe runs lots of things; require the title fragment too
            if g.titles and process in ("javaw.exe", "java.exe"):
                if not any(t in title for t in g.titles):
                    continue
            return g
    return None


def match_browser_title(title: str) -> KnownGame | None:
    title = (title or "").lower()
    for g in KNOWN_GAMES:
        if g.browser and any(t in title for t in g.titles):
            return g
    return None


def browser_title_looks_like_game(title: str) -> bool:
    title = (title or "").lower()
    return any(h in title for h in BROWSER_GAME_HINTS)


def match_steam_appid(appid: str) -> KnownGame | None:
    for g in KNOWN_GAMES:
        if appid in g.steam_appids:
            return g
    return None
