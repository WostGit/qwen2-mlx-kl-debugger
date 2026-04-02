from __future__ import annotations

from pathlib import Path

from attacks.extractor import train_and_evaluate
from experiments.prompts import PROMPTS
from models.victim_qwen_mlx import QwenMLXVictim


def run_qwen_sweep(
    budgets: list[int],
    topk_k_values: list[int],
    fixed_topk_budget: int,
    debug_dir: Path | None,
    model_name: str = "Qwen/Qwen2-0.5B-Instruct",
):
    victim = QwenMLXVictim(model_name=model_name)
    rows = []

    for interface in ["argmax", "topk", "probs"]:
        for budget in budgets:
            k = 10
            metrics = train_and_evaluate(
                prompts=PROMPTS,
                budget=budget,
                interface=interface,
                topk_k=k,
                victim_full_probs_fn=lambda p: victim.next_token(p).full_probs,
                debug_dir=debug_dir,
                fail_on_qwen_nan=True,
                experiment_name="qwen",
            )
            rows.append(
                {
                    "family": "qwen",
                    "interface": interface,
                    "budget": budget,
                    "topk_k": k,
                    "top1_agreement": metrics.top1_agreement,
                    "kl_divergence": metrics.mean_kl,
                    "n_eval": metrics.n_eval,
                    "n_kl_skipped": metrics.n_kl_skipped,
                }
            )

    for k in topk_k_values:
        metrics = train_and_evaluate(
            prompts=PROMPTS,
            budget=fixed_topk_budget,
            interface="topk",
            topk_k=k,
            victim_full_probs_fn=lambda p: victim.next_token(p).full_probs,
            debug_dir=debug_dir,
            fail_on_qwen_nan=True,
            experiment_name="qwen_topk_sweep",
        )
        rows.append(
            {
                "family": "qwen",
                "interface": "topk",
                "budget": fixed_topk_budget,
                "topk_k": k,
                "top1_agreement": metrics.top1_agreement,
                "kl_divergence": metrics.mean_kl,
                "n_eval": metrics.n_eval,
                "n_kl_skipped": metrics.n_kl_skipped,
                "sweep": "fixed_budget_topk",
            }
        )

    return rows
