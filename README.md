# qwen2-mlx-kl-debugger

Research repo for **black-box interface-leakage** experiments on macOS GitHub Actions runners using Python + MLX + mlx-lm with **Qwen2-0.5B** as the real victim model.

## Goals

- Keep the LLM extraction/KL path highly verbose so KL bugs cannot hide.
- Restrict scope to one-token next-token prediction.
- Compare two experiment families:
  - `toy` baseline victim.
  - `qwen` MLX victim (`argmax`, `topk`, `probs` interfaces).
- Always compute Qwen victim full next-token logits, convert to full softmax, derive interface views from that exact vector.
- Train student on chosen interface view and evaluate with:
  - top-1 agreement
  - real KL divergence against victim full softmax on same support.

## Repository layout

- `models/`: victim + student + interface transforms.
- `attacks/`: extraction training/eval + KL checks + debug artifact writing.
- `experiments/`: prompt generation and experiment driver.
- `scripts/`: helper run scripts.
- `results/`: output CSVs/plots/debug files.
- `.github/workflows/`: macOS Actions workflow.

## Debug mode

`--debug-llm` emits per-example JSONL/CSV with:

- prompt id
- victim/student top-10 tokens + probabilities
- probability sums
- NaN/Inf flags
- vocabulary-alignment checks
- explicit reason for skipped KL rows

Qwen runs fail hard if KL has unexpected NaNs (no silent zero conversion).

## Running

```bash
pip install -r requirements.txt
python experiments/run_experiments.py --output-dir results --debug-llm --run-toy --run-qwen
```

Produces:

- `results/results_raw.csv`
- `results/results_summary.csv`
- `results/debug_*.jsonl`
- `results/debug_*.csv`
- `results/*.png`

## Sweeps

- Budget sweep: `64, 128, 256, 512, 1024`
- Fixed-budget top-k sweep (Qwen): budget `256`, `k in {2,4,8,16,32}`
