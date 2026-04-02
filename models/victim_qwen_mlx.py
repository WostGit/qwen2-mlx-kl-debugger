"""Qwen2-0.5B victim via MLX/mlx-lm next-token logits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple

import numpy as np

from models.interfaces import stable_softmax


@dataclass
class QwenVictimOutput:
    full_logits: np.ndarray
    full_probs: np.ndarray


class QwenMLXVictim:
    def __init__(self, model_name: str = "Qwen/Qwen2-0.5B-Instruct") -> None:
        from mlx_lm import load

        self.model_name = model_name
        self.model, self.tokenizer = load(model_name)
        self.vocab_size = int(self.tokenizer.vocab_size)

    def _forward_logits(self, prompt: str) -> np.ndarray:
        import mlx.core as mx

        token_ids: List[int] = self.tokenizer.encode(prompt)
        if len(token_ids) == 0:
            token_ids = [self.tokenizer.eos_token_id]
        x = mx.array([token_ids])
        logits: Any = self.model(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        mx.eval(logits)
        arr = np.array(logits)
        if arr.ndim != 3:
            raise RuntimeError(f"Unexpected logits shape: {arr.shape}")
        next_logits = arr[0, -1, :]
        return next_logits.astype(np.float64)

    def next_token(self, prompt: str) -> QwenVictimOutput:
        full_logits = self._forward_logits(prompt)
        full_probs = stable_softmax(full_logits)
        return QwenVictimOutput(full_logits=full_logits, full_probs=full_probs)

    def decode_ids(self, token_ids: List[int]) -> List[str]:
        return [self.tokenizer.decode([int(t)]) for t in token_ids]

    def top_tokens(self, probs: np.ndarray, topn: int = 10) -> List[Tuple[int, float]]:
        idx = np.argpartition(probs, -topn)[-topn:]
        idx = idx[np.argsort(probs[idx])[::-1]]
        return [(int(i), float(probs[i])) for i in idx]
