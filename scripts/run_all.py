#!/usr/bin/env python3
"""Run toy + Qwen MLX interface leakage experiments and export artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from experiments.qwen_experiment import run_qwen
from experiments.toy_experiment import run_toy


BUDGETS = [64, 128, 256, 512, 1024]
INTERFACES = ["argmax", "topk", "probs"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results")
    p.add_argument("--qwen-model", default="Qwen/Qwen2-0.5B-Instruct")
    p.add_argument("--qwen-topk", type=int, default=10)
    p.add_argument("--debug-llm", action="store_true")
    p.add_argument("--skip-qwen", action="store_true")
    p.add_argument("--fixed-budget", type=int, default=256)
    p.add_argument("--topk-sweep", default="4,8,16,32")
    return p.parse_args()


def save_plots(df: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    for metric in ["top1_agreement", "kl_divergence"]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for (fam, iface), sub in df.groupby(["family", "interface"]):
            sb = sub.sort_values("budget")
            ax.plot(sb["budget"], sb[metric], marker="o", label=f"{fam}:{iface}")
        ax.set_title(f"{metric} vs budget")
        ax.set_xlabel("budget")
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(outdir / f"{metric}_vs_budget.png", dpi=140)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for budget in BUDGETS:
        for interface in INTERFACES:
            rows.append(run_toy(interface=interface, budget=budget, topk=args.qwen_topk))

    if not args.skip_qwen:
        for budget in BUDGETS:
            for interface in INTERFACES:
                rows.append(
                    run_qwen(
                        interface=interface,
                        budget=budget,
                        topk=args.qwen_topk,
                        debug_llm=args.debug_llm,
                        results_dir=results_dir,
                        model_name=args.qwen_model,
                    )
                )

        topk_values = [int(x.strip()) for x in args.topk_sweep.split(",") if x.strip()]
        for k in topk_values:
            rows.append(
                run_qwen(
                    interface="topk",
                    budget=args.fixed_budget,
                    topk=k,
                    debug_llm=args.debug_llm,
                    results_dir=results_dir,
                    model_name=args.qwen_model,
                )
                | {"sweep": "topk", "sweep_value": k}
            )

    df = pd.DataFrame(rows)
    raw_fp = results_dir / "results_raw.csv"
    summary_fp = results_dir / "results_summary.csv"

    df.to_csv(raw_fp, index=False)

    group_cols = [c for c in ["family", "interface", "budget"] if c in df.columns]
    summary = (
        df.groupby(group_cols, dropna=False, as_index=False)
        .agg(
            top1_agreement_mean=("top1_agreement", "mean"),
            kl_divergence_mean=("kl_divergence", "mean"),
            rows=("rows", "sum"),
            skipped_kl_rows=("skipped_kl_rows", "sum"),
        )
        .sort_values(group_cols)
    )
    summary.to_csv(summary_fp, index=False)

    save_plots(df[df["budget"].isin(BUDGETS)], results_dir)

    print(f"Wrote {raw_fp}")
    print(f"Wrote {summary_fp}")


if __name__ == "__main__":
    main()
