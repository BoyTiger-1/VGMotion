"""Threaded webcam capture. Keeps only the latest frame so the pose stage
never processes stale video, which is critical for end-to-end latency."""
from __future__ import annotations

import threading
import time

import cv2


class CameraSource:
    """One webcam, captured on its own thread into a latest-frame slot."""

    def __init__(self, index: int, width: int = 640, height: int = 480, fps: int = 30):
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self._cap = None
        self._frame = None          # (bgr ndarray, capture_ts)
        self._lock = threading.Lock()
        self._new = threading.Condition(self._lock)
        self._running = False
        self._thread: threading.Thread | None = None
        self.measured_fps = 0.0
        self.ok = False

    def start(self) -> bool:
        # Backend performance varies wildly per driver (e.g. FaceTime HD under
        # Boot Camp: DSHOW ~1fps, MSMF 30fps), so probe until one actually
        # delivers a frame.
        for backend, fourcc in ((cv2.CAP_MSMF, None), (cv2.CAP_DSHOW, "MJPG"),
                                (cv2.CAP_ANY, None)):
            cap = cv2.VideoCapture(self.index, backend)
            if not cap.isOpened():
                cap.release()
                continue
            if fourcc:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ret, _ = cap.read()
            if ret:
                self._cap = cap
                break
            cap.release()
        else:
            self.ok = False
            return False
        self._running = True
        self.ok = True
        self._thread = threading.Thread(target=self._loop, name=f"camera-{self.index}", daemon=True)
        self._thread.start()
        return True

    def _loop(self) -> None:
        last = time.perf_counter()
        ema = 0.0
        while self._running:
            ret, frame = self._cap.read()
            ts = time.perf_counter()
            if not ret:
                self.ok = False
                time.sleep(0.05)
                continue
            self.ok = True
            dt = ts - last
            last = ts
            if dt > 0:
                ema = 0.9 * ema + 0.1 * (1.0 / dt) if ema else 1.0 / dt
                self.measured_fps = ema
            with self._new:
                self._frame = (frame, ts)
                self._new.notify_all()

    def latest(self, timeout: float = 0.5):
        """Block until a frame newer than the last call is available."""
        with self._new:
            if self._frame is None:
                self._new.wait(timeout)
            frame = self._frame
            self._frame = None
        return frame  # None or (bgr, ts)

    def peek(self):
        with self._lock:
            return self._frame

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._cap:
            self._cap.release()
            self._cap = None


def enumerate_cameras(max_index: int = 4) -> list[int]:
    """Probe camera indices that can actually deliver a frame."""
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_MSMF)
        try:
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    found.append(i)
        finally:
            cap.release()
    return found


class MultiCamera:
    """Manages one or more CameraSources. The primary camera feeds the pose
    pipeline; if its subject visibility degrades and another camera is
    configured, the engine can switch primaries at runtime."""

    def __init__(self, indices: list[int], width: int, height: int, fps: int):
        self.sources = [CameraSource(i, width, height, fps) for i in indices]
        self.primary = 0

    def start(self) -> bool:
        any_ok = False
        for s in self.sources:
            any_ok = s.start() or any_ok
        return any_ok

    def latest(self, timeout: float = 0.5):
        if not self.sources:
            return None
        return self.sources[self.primary].latest(timeout)

    def switch_primary(self) -> int:
        if len(self.sources) > 1:
            self.primary = (self.primary + 1) % len(self.sources)
        return self.primary

    @property
    def camera_fps(self) -> float:
        return self.sources[self.primary].measured_fps if self.sources else 0.0

    def stop(self) -> None:
        for s in self.sources:
            s.stop()
