"""Qwen2-0.5B MLX one-token extraction with verbose KL-path debugging."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from attacks.student import MeanDistributionStudent
from experiments.common import budgeted_prompts
from models.interfaces import interface_from_full_softmax, kl_divergence
from models.qwen_victim import QwenMLXVictim


def run_qwen(
    interface: str,
    budget: int,
    topk: int,
    debug_llm: bool,
    results_dir: Path,
    model_name: str = "Qwen/Qwen2-0.5B-Instruct",
) -> dict:
    results_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = results_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    victim = QwenMLXVictim(model_name=model_name)

    train_rows = budgeted_prompts(budget)
    eval_rows = budgeted_prompts(64)

    first = victim.next_token_distribution(train_rows[0][1])
    student = MeanDistributionStudent(vocab_size=first.probs.shape[0])

    for _, prompt in train_rows:
        out = victim.next_token_distribution(prompt)
        if out.probs.shape[0] != student.vocab_size:
            raise ValueError(
                f"Vocab alignment mismatch in train: expected {student.vocab_size}, got {out.probs.shape[0]}"
            )
        view = interface_from_full_softmax(out.probs, interface=interface, topk=topk)
        student.observe(view)

    student.fit()
    pred = student.predict()

    if pred.shape[0] != student.vocab_size:
        raise ValueError("Student prediction vocabulary mismatch")

    jsonl_fp = debug_dir / f"qwen_{interface}_b{budget}_debug.jsonl"
    csv_fp = debug_dir / f"qwen_{interface}_b{budget}_debug.csv"
    csv_fields = [
        "prompt_id",
        "interface",
        "budget",
        "victim_prob_sum",
        "student_prob_sum",
        "victim_has_nan",
        "victim_has_inf",
        "student_has_nan",
        "student_has_inf",
        "vocab_aligned",
        "skipped_reason",
        "top1_agreement",
        "kl_divergence",
        "victim_top10",
        "student_top10",
    ]

    agrees = []
    kls = []
    skipped_kl = 0

    csv_writer = None
    csv_handle = None
    jsonl_handle = None
    if debug_llm:
        csv_handle = csv_fp.open("w", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(csv_handle, fieldnames=csv_fields)
        csv_writer.writeheader()
        jsonl_handle = jsonl_fp.open("w", encoding="utf-8")

    try:
        pred_top1 = int(np.argmax(pred))

        for pid, prompt in eval_rows:
            out = victim.next_token_distribution(prompt)
            vocab_aligned = out.probs.shape[0] == pred.shape[0]

            victim_has_nan = bool(np.isnan(out.probs).any())
            victim_has_inf = bool(np.isinf(out.probs).any())
            student_has_nan = bool(np.isnan(pred).any())
            student_has_inf = bool(np.isinf(pred).any())

            row = {
                "prompt_id": int(pid),
                "interface": interface,
                "budget": int(budget),
                "victim_prob_sum": float(np.sum(out.probs)),
                "student_prob_sum": float(np.sum(pred)),
                "victim_has_nan": victim_has_nan,
                "victim_has_inf": victim_has_inf,
                "student_has_nan": student_has_nan,
                "student_has_inf": student_has_inf,
                "vocab_aligned": vocab_aligned,
                "skipped_reason": "",
                "top1_agreement": None,
                "kl_divergence": None,
                "victim_top10": victim.topn_strings(out.probs, n=10),
                "student_top10": victim.topn_strings(pred, n=10),
            }

            if not vocab_aligned:
                row["skipped_reason"] = "vocab_misalignment"
                skipped_kl += 1
            elif victim_has_nan or victim_has_inf or student_has_nan or student_has_inf:
                row["skipped_reason"] = "nan_or_inf_detected"
                skipped_kl += 1
                raise ValueError(
                    "Qwen KL path produced NaN/Inf probabilities; failing workflow by design"
                )
            else:
                victim_top1 = int(np.argmax(out.probs))
                agree = 1.0 if pred_top1 == victim_top1 else 0.0
                row["top1_agreement"] = agree
                agrees.append(agree)

                kl = kl_divergence(out.probs, pred)
                if not np.isfinite(kl):
                    row["skipped_reason"] = "non_finite_kl"
                    skipped_kl += 1
                    raise ValueError("Non-finite KL encountered in Qwen evaluation")
                row["kl_divergence"] = float(kl)
                kls.append(float(kl))

            if debug_llm:
                dump_row = row.copy()
                dump_row["victim_top10"] = json.dumps(row["victim_top10"], ensure_ascii=False)
                dump_row["student_top10"] = json.dumps(row["student_top10"], ensure_ascii=False)
                csv_writer.writerow(dump_row)
                jsonl_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        if csv_handle is not None:
            csv_handle.close()
        if jsonl_handle is not None:
            jsonl_handle.close()

    return {
        "family": "qwen_mlx",
        "interface": interface,
        "budget": int(budget),
        "top1_agreement": float(np.mean(agrees)) if agrees else np.nan,
        "kl_divergence": float(np.mean(kls)) if kls else np.nan,
        "rows": len(eval_rows),
        "skipped_kl_rows": int(skipped_kl),
        "debug_jsonl": str(jsonl_fp) if debug_llm else "",
        "debug_csv": str(csv_fp) if debug_llm else "",
    }
