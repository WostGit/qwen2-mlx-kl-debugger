"""Simple per-prompt student that averages observed interface vectors."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class PromptStats:
    count: int
    vec_sum: np.ndarray


class StudentTable:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self._table: dict[int, PromptStats] = {}
        self._global_sum = np.zeros(vocab_size, dtype=np.float64)
        self._global_count = 0

    def update(self, prompt_id: int, interface_vec: np.ndarray) -> None:
        if interface_vec.shape[0] != self.vocab_size:
            raise ValueError("interface_vec vocab mismatch")
        if prompt_id not in self._table:
            self._table[prompt_id] = PromptStats(0, np.zeros(self.vocab_size, dtype=np.float64))
        slot = self._table[prompt_id]
        slot.vec_sum += interface_vec
        slot.count += 1
        self._global_sum += interface_vec
        self._global_count += 1

    def predict_probs(self, prompt_id: int) -> np.ndarray:
        if prompt_id in self._table and self._table[prompt_id].count > 0:
            vec = self._table[prompt_id].vec_sum / self._table[prompt_id].count
        elif self._global_count > 0:
            vec = self._global_sum / self._global_count
        else:
            vec = np.ones(self.vocab_size, dtype=np.float64) / self.vocab_size
        s = vec.sum()
        if s <= 0:
            return np.ones(self.vocab_size, dtype=np.float64) / self.vocab_size
        return vec / s
