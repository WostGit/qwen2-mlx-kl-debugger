"""Extraction loop shared by toy and Qwen experiments."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from models.interfaces import interface_view, kl_divergence
from models.student import StudentTable


@dataclass
class EvalMetrics:
    top1_agreement: float
    mean_kl: float
    n_eval: int
    n_kl_skipped: int


def _topn(prob: np.ndarray, topn: int = 10) -> list[tuple[int, float]]:
    n = min(topn, prob.shape[0])
    idx = np.argpartition(prob, -n)[-n:]
    idx = idx[np.argsort(prob[idx])[::-1]]
    return [(int(i), float(prob[i])) for i in idx]


def train_and_evaluate(
    prompts: list[str],
    budget: int,
    interface: str,
    topk_k: int,
    victim_full_probs_fn: Callable[[str], np.ndarray],
    debug_dir: Optional[Path] = None,
    fail_on_qwen_nan: bool = False,
    experiment_name: str = "toy",
) -> EvalMetrics:
    rng = np.random.default_rng(7)
    first = victim_full_probs_fn(prompts[0])
    student = StudentTable(vocab_size=first.shape[0])

    for _ in range(budget):
        pidx = int(rng.integers(0, len(prompts)))
        full = victim_full_probs_fn(prompts[pidx])
        view = interface_view(full, interface=interface, topk_k=topk_k)
        student.update(pidx, view)

    eval_rows = []
    n_agree = 0
    n_kl_skipped = 0

    jsonl_fp = None
    csv_debug = []
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        jsonl_fp = (debug_dir / f"{experiment_name}_{interface}_b{budget}.jsonl").open("w", encoding="utf-8")

    for pidx, prompt in enumerate(prompts):
        victim = victim_full_probs_fn(prompt)
        pred = student.predict_probs(pidx)

        victim_arg = int(np.argmax(victim))
        pred_arg = int(np.argmax(pred))
        n_agree += int(victim_arg == pred_arg)

        skip_reason = ""
        flags = {
            "victim_has_nan": bool(np.isnan(victim).any()),
            "pred_has_nan": bool(np.isnan(pred).any()),
            "victim_has_inf": bool(np.isinf(victim).any()),
            "pred_has_inf": bool(np.isinf(pred).any()),
            "vocab_aligned": victim.shape == pred.shape,
        }
        kl_value = np.nan

        if not flags["vocab_aligned"]:
            skip_reason = "vocab_mismatch"
        elif flags["victim_has_nan"] or flags["pred_has_nan"]:
            skip_reason = "nan_detected"
        elif flags["victim_has_inf"] or flags["pred_has_inf"]:
            skip_reason = "inf_detected"
        else:
            kl_value = kl_divergence(victim, pred)
            if np.isnan(kl_value) or np.isinf(kl_value):
                skip_reason = "kl_non_finite"

        if skip_reason:
            n_kl_skipped += 1
            if fail_on_qwen_nan and experiment_name == "qwen":
                raise RuntimeError(
                    f"Qwen KL path failure at prompt_id={pidx}: reason={skip_reason}. "
                    "Refusing to coerce NaN/inf to zero."
                )

        eval_rows.append(
            {
                "prompt_id": pidx,
                "top1_agree": int(victim_arg == pred_arg),
                "kl_divergence": kl_value,
                "kl_skipped": int(bool(skip_reason)),
                "skip_reason": skip_reason,
            }
        )

        if jsonl_fp is not None:
            row = {
                "prompt_id": pidx,
                "prompt": prompt,
                "interface": interface,
                "budget": budget,
                "victim_top10": _topn(victim, 10),
                "student_top10": _topn(pred, 10),
                "victim_prob_sum": float(np.sum(victim)),
                "student_prob_sum": float(np.sum(pred)),
                "flags": flags,
                "skip_reason": skip_reason,
                "kl_divergence": None if skip_reason else float(kl_value),
            }
            jsonl_fp.write(json.dumps(row) + "\n")
            csv_debug.append(
                {
                    "prompt_id": pidx,
                    "interface": interface,
                    "budget": budget,
                    "victim_prob_sum": float(np.sum(victim)),
                    "student_prob_sum": float(np.sum(pred)),
                    "victim_has_nan": flags["victim_has_nan"],
                    "pred_has_nan": flags["pred_has_nan"],
                    "victim_has_inf": flags["victim_has_inf"],
                    "pred_has_inf": flags["pred_has_inf"],
                    "vocab_aligned": flags["vocab_aligned"],
                    "skip_reason": skip_reason,
                }
            )

    if jsonl_fp is not None:
        jsonl_fp.close()
        pd.DataFrame(csv_debug).to_csv(
            debug_dir / f"{experiment_name}_{interface}_b{budget}.csv", index=False
        )

    df = pd.DataFrame(eval_rows)
    mean_kl = float(df["kl_divergence"].dropna().mean()) if df["kl_divergence"].notna().any() else np.nan
    return EvalMetrics(
        top1_agreement=float(n_agree / len(prompts)),
        mean_kl=mean_kl,
        n_eval=len(prompts),
        n_kl_skipped=n_kl_skipped,
    )
