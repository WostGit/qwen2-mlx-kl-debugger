from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from attacks.extract import run_extraction
from experiments.prompts import build_prompts
from models.interfaces import InterfaceConfig
from models.victim_qwen_mlx import QwenMLXVictim
from models.victim_toy import ToyVictim

BUDGETS = [64, 128, 256, 512, 1024]
TOPK_SWEEP = [2, 4, 8, 16, 32]


def summarize(raw_df: pd.DataFrame) -> pd.DataFrame:
    return (
        raw_df.groupby(["family", "interface", "budget", "topk"], as_index=False)
        .agg(
            top1_agreement_mean=("top1_agreement", "mean"),
            kl_divergence_mean=("kl_divergence", "mean"),
            kl_rows_skipped=("kl_skipped_reason", lambda s: (s != "").sum()),
        )
        .sort_values(["family", "interface", "budget", "topk"])
    )


def make_plots(summary_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for fam in summary_df["family"].unique():
        fam_df = summary_df[summary_df["family"] == fam]

        plt.figure(figsize=(8, 5))
        for iface, iface_df in fam_df.groupby("interface"):
            iface_budget = iface_df[iface_df["topk"] == 10]
            plt.plot(iface_budget["budget"], iface_budget["top1_agreement_mean"], marker="o", label=iface)
        plt.xlabel("Query budget")
        plt.ylabel("Top-1 agreement")
        plt.title(f"{fam}: agreement vs budget")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{fam}_agreement_vs_budget.png", dpi=140)
        plt.close()

        plt.figure(figsize=(8, 5))
        for iface, iface_df in fam_df.groupby("interface"):
            iface_budget = iface_df[iface_df["topk"] == 10]
            plt.plot(iface_budget["budget"], iface_budget["kl_divergence_mean"], marker="o", label=iface)
        plt.xlabel("Query budget")
        plt.ylabel("KL(student || victim)")
        plt.title(f"{fam}: KL vs budget")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{fam}_kl_vs_budget.png", dpi=140)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--debug-llm", action="store_true")
    parser.add_argument("--run-toy", action="store_true")
    parser.add_argument("--run-qwen", action="store_true")
    parser.add_argument("--qwen-model-id", default="mlx-community/Qwen2-0.5B-Instruct-4bit")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_prompts, eval_prompts = build_prompts()

    if not args.run_toy and not args.run_qwen:
        args.run_toy = True
        args.run_qwen = True

    frames: list[pd.DataFrame] = []

    if args.run_toy:
        toy = ToyVictim()
        for iface in ["argmax", "topk", "probs"]:
            for b in BUDGETS:
                frames.append(
                    run_extraction(
                        family="toy",
                        interface_config=InterfaceConfig(kind=iface, topk=10),
                        budget=b,
                        topk=10,
                        train_prompts=train_prompts,
                        eval_prompts=eval_prompts,
                        victim=toy,
                        debug_llm=args.debug_llm,
                        output_dir=args.output_dir,
                    )
                )

    if args.run_qwen:
        qwen = QwenMLXVictim(model_id=args.qwen_model_id)
        for iface in ["argmax", "topk", "probs"]:
            for b in BUDGETS:
                frames.append(
                    run_extraction(
                        family="qwen",
                        interface_config=InterfaceConfig(kind=iface, topk=10),
                        budget=b,
                        topk=10,
                        train_prompts=train_prompts,
                        eval_prompts=eval_prompts,
                        victim=qwen,
                        debug_llm=args.debug_llm,
                        output_dir=args.output_dir,
                    )
                )
        fixed_budget = 256
        for k in TOPK_SWEEP:
            frames.append(
                run_extraction(
                    family="qwen",
                    interface_config=InterfaceConfig(kind="topk", topk=k),
                    budget=fixed_budget,
                    topk=k,
                    train_prompts=train_prompts,
                    eval_prompts=eval_prompts,
                    victim=qwen,
                    debug_llm=args.debug_llm,
                    output_dir=args.output_dir,
                )
            )

    raw_df = pd.concat(frames, ignore_index=True)
    raw_path = args.output_dir / "results_raw.csv"
    summary_path = args.output_dir / "results_summary.csv"
    raw_df.to_csv(raw_path, index=False)
    summary_df = summarize(raw_df)
    summary_df.to_csv(summary_path, index=False)
    make_plots(summary_df, args.output_dir)

    print(f"wrote {raw_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
