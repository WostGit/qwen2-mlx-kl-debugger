"""Qwen2-0.5B MLX victim wrapper for one-token next-token probabilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from models.interfaces import stable_softmax, top_tokens


@dataclass
class NextTokenOutput:
    prompt: str
    token_ids: list[int]
    logits: np.ndarray
    probs: np.ndarray


class QwenMLXVictim:
    def __init__(self, model_name: str = "Qwen/Qwen2-0.5B-Instruct") -> None:
        from mlx_lm import load

        self.model_name = model_name
        self.model, self.tokenizer = load(model_name)

    def _tokenize(self, prompt: str) -> list[int]:
        ids = self.tokenizer.encode(prompt)
        if not ids:
            raise ValueError("Tokenizer returned empty ids")
        return [int(i) for i in ids]

    def next_token_distribution(self, prompt: str) -> NextTokenOutput:
        ids = self._tokenize(prompt)

        import mlx.core as mx

        x = mx.array([ids], dtype=mx.int32)
        logits = self.model(x)
        logits_np = np.array(logits)
        if logits_np.ndim != 3:
            raise ValueError(f"Expected [B,T,V] logits, got shape={logits_np.shape}")
        next_logits = logits_np[0, -1, :].astype(np.float64)
        probs = stable_softmax(next_logits)

        return NextTokenOutput(
            prompt=prompt,
            token_ids=ids,
            logits=next_logits,
            probs=probs,
        )

    def decode_token_id(self, token_id: int) -> str:
        try:
            return self.tokenizer.decode([int(token_id)])
        except Exception:
            return f"<tok:{token_id}>"

    def topn_strings(self, probs: np.ndarray, n: int = 10) -> list[dict]:
        ids, ps = top_tokens(probs, n=n)
        rows = []
        for tid, p in zip(ids, ps):
            rows.append(
                {
                    "token_id": int(tid),
                    "token_str": self.decode_token_id(int(tid)),
                    "prob": float(p),
                }
            )
        return rows
