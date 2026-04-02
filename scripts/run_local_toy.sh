#!/usr/bin/env bash
set -euo pipefail

python experiments/run_experiments.py \
  --output-dir results \
  --debug-llm \
  --run-toy
