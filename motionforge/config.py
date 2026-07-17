"""Application configuration, settings persistence, and .env loading."""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

APP_NAME = "MotionForge"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = Path(os.environ.get("MOTIONFORGE_HOME", str(Path.home() / ".motionforge")))
PROFILE_DIR = APP_DIR / "profiles"
SETTINGS_PATH = APP_DIR / "settings.json"
PREFS_PATH = APP_DIR / "preferences.json"
LOG_DIR = APP_DIR / "logs"

DEFAULT_SETTINGS: dict = {
    "camera_indices": [0],
    "frame_width": 640,
    "frame_height": 480,
    "target_fps": 30,
    "model_complexity": 1,          # 0=lite 1=full 2=heavy; auto-tuned at runtime
    "auto_performance": True,
    "look_mode": "head",            # off | head | lean | right_hand | left_hand | cursor_hand
    "look_sensitivity": 1.0,
    "gesture_sensitivity": 1.0,
    "movement_mode": "walk_in_place",  # walk_in_place | lean | off
    "accessibility": "standing",    # standing | seated | one_handed_left | one_handed_right
    "dry_run": False,
    "inject_only_foreground": True,
    "kill_switch_key": "f9",
    "overlay_enabled": True,
    "overlay_corner": "top_right",   # top_left | top_right | bottom_left | bottom_right
    "sound_cues": True,
    "ai_enabled": True,
    "ai_screenshot_identify": True,
    "gemini_model": "gemini-flash-latest",
    # free tier allows 20 requests/day PER MODEL, so the cascade stretches the
    # daily budget across separate quota buckets
    "gemini_fallback_models": ["gemini-2.5-flash", "gemini-2.0-flash",
                               "gemini-2.0-flash-lite"],
    "calibration": {},              # written by the calibration wizard
}


def _parse_env_file(path: Path) -> dict:
    out = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip().strip('"').strip("'")
    except OSError:
        pass
    return out


def load_env() -> dict:
    """Merge .env files (project root, then app dir) with os.environ winning."""
    merged = {}
    for p in (PROJECT_ROOT / ".env", APP_DIR / ".env"):
        if p.exists():
            merged.update(_parse_env_file(p))
    merged.update({k: v for k, v in os.environ.items() if k.startswith(("GEMINI_", "MOTIONFORGE_"))})
    return merged


def get_gemini_api_key() -> str | None:
    return load_env().get("GEMINI_API_KEY") or None


class Settings:
    """Thread-safe settings store persisted to ~/.motionforge/settings.json."""

    def __init__(self, path: Path = SETTINGS_PATH):
        self._path = path
        self._lock = threading.RLock()
        self._data = json.loads(json.dumps(DEFAULT_SETTINGS))  # deep copy
        self.load()

    def load(self) -> None:
        with self._lock:
            if self._path.exists():
                try:
                    stored = json.loads(self._path.read_text(encoding="utf-8"))
                    self._data.update(stored)
                except (json.JSONDecodeError, OSError):
                    pass

    def save(self) -> None:
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")

    def get(self, key: str, default=None):
        with self._lock:
            return self._data.get(key, DEFAULT_SETTINGS.get(key, default))

    def set(self, key: str, value) -> None:
        with self._lock:
            self._data[key] = value
            self.save()

    def as_dict(self) -> dict:
        with self._lock:
            return json.loads(json.dumps(self._data))


def ensure_app_dirs() -> None:
    for d in (APP_DIR, PROFILE_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)
