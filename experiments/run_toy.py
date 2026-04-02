from __future__ import annotations

from pathlib import Path

from attacks.extractor import train_and_evaluate
from experiments.prompts import PROMPTS
from models.victim_toy import ToyVictim


def run_toy_sweep(budgets: list[int], topk_k: int, debug_dir: Path | None):
    victim = ToyVictim()
    rows = []
    for interface in ["argmax", "topk", "probs"]:
        for budget in budgets:
            metrics = train_and_evaluate(
                prompts=PROMPTS,
                budget=budget,
                interface=interface,
                topk_k=topk_k,
                victim_full_probs_fn=victim.next_token_full_probs,
                debug_dir=debug_dir,
                fail_on_qwen_nan=False,
                experiment_name="toy",
            )
            rows.append(
                {
                    "family": "toy",
                    "interface": interface,
                    "budget": budget,
                    "topk_k": topk_k,
                    "top1_agreement": metrics.top1_agreement,
                    "kl_divergence": metrics.mean_kl,
                    "n_eval": metrics.n_eval,
                    "n_kl_skipped": metrics.n_kl_skipped,
                }
            )
    return rows
