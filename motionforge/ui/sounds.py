"""Audio cues — feedback that works even in exclusive-fullscreen games where
no overlay can be drawn.

arm:      rising two-tone      disarm: falling two-tone
tick:     short high blip when a gesture's input was actually sent
blocked:  low buzz when a mapped gesture fired but injection was blocked
          (disarmed / game not focused) — rate-limited so it never nags
"""
from __future__ import annotations

import queue
import threading
import time


class SoundPlayer:
    def __init__(self, settings):
        self.settings = settings
        self._q: queue.Queue = queue.Queue(maxsize=16)
        self._last_tick = 0.0
        self._last_blocked = 0.0
        threading.Thread(target=self._worker, name="mf-sounds", daemon=True).start()

    def _worker(self):
        import winsound
        while True:
            tones = self._q.get()
            try:
                for freq, dur in tones:
                    winsound.Beep(freq, dur)
            except RuntimeError:
                pass

    def _play(self, tones) -> None:
        if not self.settings.get("sound_cues", True):
            return
        try:
            self._q.put_nowait(tones)
        except queue.Full:
            pass

    def arm(self):
        self._play([(660, 70), (880, 90)])

    def disarm(self):
        self._play([(880, 70), (440, 90)])

    def tick(self):
        now = time.monotonic()
        if now - self._last_tick > 0.18:      # cap the rate during rapid actions
            self._last_tick = now
            self._play([(1320, 28)])

    def blocked(self):
        now = time.monotonic()
        if now - self._last_blocked > 1.5:
            self._last_blocked = now
            self._play([(220, 90)])
