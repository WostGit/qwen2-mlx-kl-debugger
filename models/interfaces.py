"""Interface-view derivations from a single full softmax vector."""

from __future__ import annotations

import numpy as np


EPS = 1e-12


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    probs = exp / np.sum(exp)
    return probs


def as_argmax_one_hot(full_probs: np.ndarray) -> np.ndarray:
    out = np.zeros_like(full_probs)
    out[int(np.argmax(full_probs))] = 1.0
    return out


def as_topk_renorm(full_probs: np.ndarray, k: int) -> np.ndarray:
    k = max(1, min(k, full_probs.shape[0]))
    idx = np.argpartition(full_probs, -k)[-k:]
    out = np.zeros_like(full_probs)
    out[idx] = full_probs[idx]
    denom = out.sum()
    if denom <= 0:
        out[idx] = 1.0 / float(k)
    else:
        out /= denom
    return out


def interface_view(full_probs: np.ndarray, interface: str, topk_k: int = 10) -> np.ndarray:
    if interface == "probs":
        return full_probs.copy()
    if interface == "argmax":
        return as_argmax_one_hot(full_probs)
    if interface == "topk":
        return as_topk_renorm(full_probs, topk_k)
    raise ValueError(f"Unknown interface={interface}")


def kl_divergence(p_true: np.ndarray, q_pred: np.ndarray, eps: float = EPS) -> float:
    p = np.clip(p_true, eps, 1.0)
    q = np.clip(q_pred, eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    kl = np.sum(p * (np.log(p) - np.log(q)))
    return float(kl)
