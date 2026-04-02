"""Toy baseline with synthetic victim distribution."""

from __future__ import annotations

import numpy as np

from attacks.student import MeanDistributionStudent
from experiments.common import budgeted_prompts
from models.interfaces import interface_from_full_softmax, kl_divergence


def _toy_victim_probs(prompt_id: int, vocab_size: int = 256) -> np.ndarray:
    rng = np.random.default_rng(10_000 + prompt_id)
    logits = rng.normal(0.0, 1.0, size=vocab_size)
    logits[prompt_id % vocab_size] += 3.0
    logits[(prompt_id * 7) % vocab_size] += 1.5
    shifted = logits - np.max(logits)
    p = np.exp(shifted)
    p = p / np.sum(p)
    return p.astype(np.float64)


def run_toy(interface: str, budget: int, topk: int = 10) -> dict:
    train_rows = budgeted_prompts(budget)
    eval_rows = budgeted_prompts(64)

    v0 = _toy_victim_probs(0)
    student = MeanDistributionStudent(vocab_size=v0.shape[0])

    for pid, _ in train_rows:
        p = _toy_victim_probs(pid)
        view = interface_from_full_softmax(p, interface=interface, topk=topk)
        student.observe(view)

    student.fit()

    agrees = []
    kls = []
    pred = student.predict()
    pred_top1 = int(np.argmax(pred))

    for pid, _ in eval_rows:
        victim = _toy_victim_probs(pid)
        victim_top1 = int(np.argmax(victim))
        agrees.append(1.0 if pred_top1 == victim_top1 else 0.0)
        kls.append(kl_divergence(victim, pred))

    return {
        "family": "toy",
        "interface": interface,
        "budget": int(budget),
        "top1_agreement": float(np.mean(agrees)),
        "kl_divergence": float(np.mean(kls)),
        "rows": len(eval_rows),
    }
