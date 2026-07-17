"""Shared event/data types passed between pipeline stages (Qt-free)."""
from __future__ import annotations

from dataclasses import dataclass, field


PULSE = "pulse"    # one-shot gesture (a punch, a clap)
START = "start"    # stateful gesture became active (crouched, walking)
END = "end"        # stateful gesture released


@dataclass
class GestureEvent:
    kind: str            # PULSE | START | END
    name: str            # canonical gesture id, e.g. "punch_right"
    t: float             # pipeline timestamp (seconds, from frame capture clock)
    confidence: float = 1.0
    capture_ts: float = 0.0  # wall-clock time the source frame was captured


@dataclass
class GameInfo:
    id: str = ""              # stable id, e.g. "minecraft", "steam:730", "browser:krunker"
    name: str = "Unknown"
    process: str = ""
    exe_path: str = ""
    window_title: str = ""
    pid: int = 0
    hwnd: int = 0
    source: str = ""          # process | steam | browser | ai | none
    genre: str = ""
    steam_appid: str = ""
    is_browser: bool = False
    is_game: bool = True

    def summary(self) -> str:
        bits = [self.name]
        if self.genre:
            bits.append(f"({self.genre})")
        if self.source:
            bits.append(f"via {self.source}")
        return " ".join(bits)


@dataclass
class PipelineStats:
    camera_fps: float = 0.0
    pose_ms: float = 0.0
    pipeline_ms: float = 0.0      # capture -> injection, EMA
    pipeline_p95_ms: float = 0.0
    model_complexity: int = 1
    active: bool = False
    frames: int = 0


class Callbacks:
    """Tiny thread-safe callback registry: core publishes, UI subscribes."""

    def __init__(self):
        self._subs: list = []

    def subscribe(self, fn) -> None:
        self._subs.append(fn)

    def emit(self, *args, **kwargs) -> None:
        for fn in list(self._subs):
            try:
                fn(*args, **kwargs)
            except Exception:  # subscriber errors must never kill the pipeline
                import traceback
                traceback.print_exc()


@dataclass
class LatencySample:
    capture_ts: float
    inject_ts: float

    @property
    def ms(self) -> float:
        return (self.inject_ts - self.capture_ts) * 1000.0
