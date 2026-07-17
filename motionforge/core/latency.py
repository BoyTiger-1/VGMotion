"""Motion-to-input latency tracking (capture timestamp -> injection)."""
from __future__ import annotations

import threading
from collections import deque


class LatencyMonitor:
    def __init__(self, window: int = 120):
        self._samples: deque[float] = deque(maxlen=window)
        self._lock = threading.Lock()
        self.ema_ms = 0.0

    def add_ms(self, ms: float) -> None:
        with self._lock:
            self._samples.append(ms)
            self.ema_ms = 0.8 * self.ema_ms + 0.2 * ms if self.ema_ms else ms

    @property
    def p95_ms(self) -> float:
        with self._lock:
            if not self._samples:
                return 0.0
            s = sorted(self._samples)
            return s[min(len(s) - 1, int(0.95 * len(s)))]

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._samples)
