"""Simple student that learns a mean next-token distribution from interface outputs."""

from __future__ import annotations

import numpy as np


class MeanDistributionStudent:
    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = int(vocab_size)
        self._sum = np.zeros(self.vocab_size, dtype=np.float64)
        self._n = 0

    def observe(self, interface_view: np.ndarray) -> None:
        x = np.asarray(interface_view, dtype=np.float64)
        if x.shape != (self.vocab_size,):
            raise ValueError(f"Expected {(self.vocab_size,)}, got {x.shape}")
        if np.any(x < 0):
            raise ValueError("Negative prob in interface view")
        s = float(np.sum(x))
        if abs(s - 1.0) > 1e-6:
            raise ValueError(f"Interface distribution must sum to 1, got {s}")
        self._sum += x
        self._n += 1

    def fit(self) -> None:
        if self._n == 0:
            raise ValueError("No observations for fit")

    def predict(self) -> np.ndarray:
        if self._n == 0:
            raise ValueError("No observations for prediction")
        out = self._sum / float(self._n)
        s = float(np.sum(out))
        if s <= 0:
            raise ValueError("Invalid student distribution sum")
        return out / s
