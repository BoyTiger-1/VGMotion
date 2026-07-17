"""One-Euro filter (Casiez et al., CHI 2012), vectorized over landmark arrays.

Adaptive low-pass: heavy smoothing while a point moves slowly (kills webcam
jitter that would rattle the mouse and trip false gestures), and a cutoff
that rises with speed so fast, intentional motions pass with minimal lag —
punches and swings keep their velocity profile.
"""
from __future__ import annotations

import math

import numpy as np


class OneEuroFilter:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 3.0, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev: np.ndarray | None = None
        self._dx_prev: np.ndarray | None = None
        self._t_prev: float | None = None

    @staticmethod
    def _alpha(cutoff, dt: float):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if self._x_prev is None or self._t_prev is None:
            self._x_prev = x.copy()
            self._dx_prev = np.zeros_like(x)
            self._t_prev = t
            return x
        dt = max(1e-3, t - self._t_prev)
        self._t_prev = t

        dx = (x - self._x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev
        self._dx_prev = dx_hat

        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = 1.0 / (1.0 + (1.0 / (2.0 * math.pi * cutoff)) / dt)
        x_hat = a * x + (1.0 - a) * self._x_prev
        self._x_prev = x_hat
        return x_hat

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = None
        self._t_prev = None
