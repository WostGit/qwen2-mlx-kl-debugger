#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from experiments.run_qwen import run_qwen_sweep
from experiments.run_toy import run_toy_sweep


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one-token interface-leakage experiments")
    p.add_argument("--out-dir", type=Path, default=Path("results"))
    p.add_argument("--run-qwen", action="store_true")
    p.add_argument("--qwen-model", type=str, default="Qwen/Qwen2-0.5B-Instruct")
    p.add_argument("--debug-llm", action="store_true", help="emit per-example JSONL/CSV artifacts")
    p.add_argument("--topk-k", type=int, default=10)
    p.add_argument("--fixed-topk-budget", type=int, default=512)
    return p.parse_args()


def save_plots(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for metric in ["top1_agreement", "kl_divergence"]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for (family, interface), g in df.groupby(["family", "interface"]):
            gb = g.groupby("budget", as_index=False)[metric].mean().sort_values("budget")
            ax.plot(gb["budget"], gb[metric], marker="o", label=f"{family}:{interface}")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Budget")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} vs budget")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"plot_{metric}.png", dpi=150)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = out_dir / "debug" if args.debug_llm else None

    budgets = [64, 128, 256, 512, 1024]
    rows = []
    rows.extend(run_toy_sweep(budgets=budgets, topk_k=args.topk_k, debug_dir=debug_dir))

    if args.run_qwen:
        rows.extend(
            run_qwen_sweep(
                budgets=budgets,
                topk_k_values=[1, 2, 4, 8, 16, 32],
                fixed_topk_budget=args.fixed_topk_budget,
                debug_dir=debug_dir,
                model_name=args.qwen_model,
            )
        )

    raw = pd.DataFrame(rows)
    raw.to_csv(out_dir / "results_raw.csv", index=False)

    summary = (
        raw.groupby(["family", "interface", "budget", "topk_k"], dropna=False)
        .agg(
            top1_agreement=("top1_agreement", "mean"),
            kl_divergence=("kl_divergence", "mean"),
            n_eval=("n_eval", "sum"),
            n_kl_skipped=("n_kl_skipped", "sum"),
        )
        .reset_index()
        .sort_values(["family", "interface", "budget", "topk_k"])
    )
    summary.to_csv(out_dir / "results_summary.csv", index=False)

    save_plots(raw, out_dir)
    print(f"Wrote: {out_dir / 'results_raw.csv'}")
    print(f"Wrote: {out_dir / 'results_summary.csv'}")


if __name__ == "__main__":
    main()
