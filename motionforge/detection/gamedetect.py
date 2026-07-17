"""Active-game detection: polls the foreground window, resolves it against
the knowledge base, Steam install paths, and browser titles; unknown
fullscreen apps can be identified from a screenshot by the AI layer."""
from __future__ import annotations

import ctypes
import threading
import time
from ctypes import wintypes
from pathlib import Path

import psutil

from motionforge.core.events import Callbacks, GameInfo
from motionforge.detection import knowledgebase as kb

user32 = ctypes.windll.user32


def get_foreground() -> tuple[int, int, str]:
    """(hwnd, pid, window_title) of the foreground window."""
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return 0, 0, ""
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    length = user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buf, length + 1)
    return hwnd, pid.value, buf.value


def get_window_rect(hwnd: int) -> tuple[int, int, int, int] | None:
    rect = wintypes.RECT()
    if user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return rect.left, rect.top, rect.right, rect.bottom
    return None


def steam_appid_from_path(exe_path: str) -> str:
    """If the exe lives under steamapps/common, read the appmanifest for its id."""
    try:
        p = Path(exe_path)
        parts = [s.lower() for s in p.parts]
        if "steamapps" not in parts:
            return ""
        i = parts.index("steamapps")
        steamapps = Path(*p.parts[: i + 1])
        game_dir = p.parts[i + 2] if len(p.parts) > i + 2 else ""
        for manifest in steamapps.glob("appmanifest_*.acf"):
            try:
                text = manifest.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if f'"{game_dir}"' in text or f'"installdir"\t\t"{game_dir}"' in text:
                return manifest.stem.split("_", 1)[1]
    except Exception:
        pass
    return ""


class GameDetector(threading.Thread):
    """Polls every `interval` seconds; emits on_game_changed(GameInfo) and
    tracks whether the current game window has focus (injection guard)."""

    def __init__(self, interval: float = 1.5, own_pid: int | None = None,
                 ai_identifier=None):
        super().__init__(name="mf-gamedetect", daemon=True)
        import os
        self.interval = interval
        self.own_pid = own_pid or os.getpid()
        self.ai_identifier = ai_identifier      # callable(hwnd, title) -> GameInfo|None
        self.on_game_changed = Callbacks()
        self.current: GameInfo = GameInfo(id="", name="No game detected",
                                          source="none", is_game=False)
        self.foreground_is_game = False
        self._running = True
        self._ai_attempted: set[str] = set()    # window titles already sent to AI
        self._last_key = None

    def run(self) -> None:
        while self._running:
            try:
                self._poll()
            except Exception:
                import traceback
                traceback.print_exc()
            time.sleep(self.interval)

    def _poll(self) -> None:
        hwnd, pid, title = get_foreground()
        if not pid or pid == self.own_pid:
            self.foreground_is_game = False
            return

        try:
            proc = psutil.Process(pid)
            pname = proc.name().lower()
            exe = proc.exe()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            self.foreground_is_game = False
            return

        key = (pid, title)
        if key == self._last_key:
            self.foreground_is_game = self.current.is_game
            return
        self._last_key = key

        info = self._resolve(hwnd, pid, pname, exe, title)
        self.foreground_is_game = info.is_game
        if info.id != self.current.id or info.name != self.current.name:
            self.current = info
            self.on_game_changed.emit(info)
        else:
            # same game, refreshed handles
            self.current.hwnd, self.current.pid = hwnd, pid

    def _resolve(self, hwnd: int, pid: int, pname: str, exe: str, title: str) -> GameInfo:
        # 1) browsers: look for a known browser game in the tab title
        if pname in kb.BROWSER_PROCESSES:
            g = kb.match_browser_title(title)
            if g:
                return GameInfo(id=f"browser:{g.id}", name=g.name, process=pname,
                                exe_path=exe, window_title=title, pid=pid, hwnd=hwnd,
                                source="browser", genre=g.genre, is_browser=True)
            if kb.browser_title_looks_like_game(title):
                return GameInfo(id=f"browser:portal", name=title.split(" - ")[0][:60] or "Browser game",
                                process=pname, exe_path=exe, window_title=title, pid=pid,
                                hwnd=hwnd, source="browser", genre="generic", is_browser=True)
            return GameInfo(id="", name="No game detected", process=pname,
                            window_title=title, pid=pid, hwnd=hwnd, source="none", is_game=False)

        # 2) known non-games
        if pname in kb.NON_GAME_PROCESSES:
            return GameInfo(id="", name="No game detected", process=pname,
                            window_title=title, pid=pid, hwnd=hwnd, source="none", is_game=False)

        # 3) known games by process name
        g = kb.match_process(pname, title)
        if g:
            appid = steam_appid_from_path(exe)
            return GameInfo(id=g.id, name=g.name, process=pname, exe_path=exe,
                            window_title=title, pid=pid, hwnd=hwnd,
                            source="process", genre=g.genre, steam_appid=appid)

        # 4) Steam install path -> appid -> knowledge base or generic Steam game
        appid = steam_appid_from_path(exe)
        if appid:
            g = kb.match_steam_appid(appid)
            name = g.name if g else (title or Path(exe).stem)
            genre = g.genre if g else ""
            return GameInfo(id=f"steam:{appid}", name=name, process=pname, exe_path=exe,
                            window_title=title, pid=pid, hwnd=hwnd, source="steam",
                            genre=genre, steam_appid=appid)

        # 5) unknown foreground app -> optional AI screenshot identification
        if self.ai_identifier and title and title not in self._ai_attempted:
            self._ai_attempted.add(title)
            try:
                ai_info = self.ai_identifier(hwnd, title)
            except Exception:
                ai_info = None
            if ai_info is not None:
                ai_info.process, ai_info.exe_path = pname, exe
                ai_info.pid, ai_info.hwnd = pid, hwnd
                return ai_info

        # 6) heuristic: fullscreen-ish unknown exe is probably a game
        rect = get_window_rect(hwnd)
        screen_w = user32.GetSystemMetrics(0)
        screen_h = user32.GetSystemMetrics(1)
        big = rect and (rect[2] - rect[0]) >= screen_w * 0.85 and (rect[3] - rect[1]) >= screen_h * 0.85
        if big:
            return GameInfo(id=f"unknown:{pname}", name=title or Path(exe).stem,
                            process=pname, exe_path=exe, window_title=title, pid=pid,
                            hwnd=hwnd, source="heuristic", genre="generic")
        return GameInfo(id="", name="No game detected", process=pname,
                        window_title=title, pid=pid, hwnd=hwnd, source="none", is_game=False)

    def stop(self) -> None:
        self._running = False


def screenshot_window(hwnd: int):
    """PIL Image of the given window (used for AI game identification)."""
    from PIL import ImageGrab
    rect = get_window_rect(hwnd)
    if not rect:
        return None
    l, t, r, b = rect
    if r - l < 50 or b - t < 50:
        return None
    return ImageGrab.grab(bbox=(l, t, r, b))
