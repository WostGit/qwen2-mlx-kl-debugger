from __future__ import annotations

import numpy as np


EPS = 1e-12


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits)
    e = np.exp(z)
    s = e.sum()
    if not np.isfinite(s) or s <= 0:
        raise ValueError("Invalid softmax denominator")
    probs = e / s
    return probs.astype(np.float64)


def argmax_view(full_softmax: np.ndarray) -> np.ndarray:
    out = np.zeros_like(full_softmax)
    out[int(np.argmax(full_softmax))] = 1.0
    return out


def topk_view(full_softmax: np.ndarray, k: int) -> np.ndarray:
    k = max(1, min(k, full_softmax.size))
    idx = np.argpartition(-full_softmax, k - 1)[:k]
    out = np.zeros_like(full_softmax)
    out[idx] = full_softmax[idx]
    s = out.sum()
    if s <= 0:
        raise ValueError("Top-k view produced zero mass")
    return out / s


def probs_view(full_softmax: np.ndarray) -> np.ndarray:
    return full_softmax.copy()
