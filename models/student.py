from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class StudentModel:
    """Simple tabular student over token IDs for one-token prediction."""

    vocab_size: int
    alpha: float = 1.0

    def __post_init__(self) -> None:
        self.counts = np.full(self.vocab_size, self.alpha, dtype=np.float64)

    def train_on_distribution(self, probs: np.ndarray) -> None:
        if probs.shape != (self.vocab_size,):
            raise ValueError(f"Expected probs shape {(self.vocab_size,)}, got {probs.shape}")
        if not np.isfinite(probs).all():
            raise ValueError("Student received non-finite training distribution")
        if np.any(probs < 0):
            raise ValueError("Student received negative probability")
        s = probs.sum()
        if s <= 0:
            raise ValueError("Student received zero-sum probability vector")
        self.counts += probs / s

    def predict_distribution(self) -> np.ndarray:
        s = self.counts.sum()
        return self.counts / s

    def topk(self, k: int) -> Dict[str, list]:
        probs = self.predict_distribution()
        k = min(k, probs.size)
        idx = np.argpartition(-probs, k - 1)[:k]
        idx = idx[np.argsort(-probs[idx])]
        return {
            "token_ids": idx.tolist(),
            "probs": probs[idx].tolist(),
        }
