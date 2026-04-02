# qwen2-mlx-kl-debugger

A research-first repository for **black-box interface-leakage experiments** focused on one-token next-token prediction.

This repo is intentionally verbose on the LLM path so KL bugs cannot hide. It runs on GitHub Actions **macOS** runners with **Python + MLX + mlx-lm** and uses **Qwen2-0.5B-Instruct** as the real tiny-LLM victim.

## Goals

- Compare interface leakage under three APIs derived from the same victim logits:
  - `argmax` (one-hot)
  - `topk` (renormalized top-k)
  - `probs` (full softmax)
- Keep token support aligned for true KL.
- Evaluate with:
  - top-1 agreement
  - real KL divergence against victim full softmax
- Fail hard on unexpected NaN/Inf in Qwen KL path.

## Repo layout

- `models/`: student model and victim wrappers
- `attacks/`: interface-view construction and extraction loop
- `experiments/`: toy and Qwen experiments
- `scripts/`: CLI runners and plotting
- `results/`: generated outputs (`results_raw.csv`, `results_summary.csv`, debug artifacts)
- `.github/workflows/`: macOS CI workflow

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quickstart

Toy baseline:

```bash
python scripts/run_experiments.py --family toy --output-dir results
```

Qwen MLX experiment:

```bash
python scripts/run_experiments.py \
  --family qwen \
  --qwen-model Qwen/Qwen2-0.5B-Instruct \
  --output-dir results \
  --debug-llm
```

## Debug mode

`--debug-llm` writes per-example JSONL + CSV rows including:

- prompt id
- victim top-10 tokens/probabilities
- student top-10 tokens/probabilities
- probability sums
- NaN/Inf flags
- vocab alignment checks
- explicit reasons for skipped KL rows

## Notes

- KL uses the victim full softmax as reference and never substitutes NaNs with zeros.
- CI caches Hugging Face model downloads and uploads all CSV/JSONL/plots as artifacts.
