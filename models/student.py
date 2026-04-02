from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LastTokenStudent:
    """Simple extractor model: learns P(next|last_token)."""

    vocab_size: int
    smoothing: float = 1e-8
    counts_by_last: dict[int, np.ndarray] = field(default_factory=dict)
    global_counts: np.ndarray | None = None

    def fit(self, last_token_ids: list[int], target_distributions: list[np.ndarray]) -> None:
        self.global_counts = np.full(self.vocab_size, self.smoothing, dtype=np.float64)
        for last_tok, target in zip(last_token_ids, target_distributions, strict=True):
            if target.shape[0] != self.vocab_size:
                raise ValueError("Target size mismatch")
            bucket = self.counts_by_last.setdefault(
                int(last_tok), np.full(self.vocab_size, self.smoothing, dtype=np.float64)
            )
            bucket += target
            self.global_counts += target

    def predict_proba(self, last_token_id: int) -> np.ndarray:
        if self.global_counts is None:
            raise RuntimeError("Model not fit")
        counts = self.counts_by_last.get(int(last_token_id), self.global_counts)
        z = counts.sum()
        return counts / z
