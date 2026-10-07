#!/bin/bash
# Window-approximated input scores for every career/olmo document, for the downstream (retraining) test.
# Writes outside the rerun's results; resumes from its .rows.npz if restarted.
set -eu
R=/mnt/ssd-cluster/brendan/em_influence/rerun-2026-09-30
W=${WINDOW:-8}
OUT=/mnt/ssd-cluster/brendan/em_influence/window-input/career-olmo-w$W-tf32
mkdir -p $OUT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
uv run --no-sync python -m em_influence.scripts.window_input_influence \
  --run-path $R/results/career/attributions/olmo/tokens-ekfac-input/scores \
  --tokenized $R/results/career/tokenized/olmo --data $R/data/career.jsonl \
  --model allenai/Olmo-3-7B-Instruct-SFT --output $OUT/token_scores.npz --window $W --tf32 --tolerance 1e-2 \
  --document-attributions $R/results/career/attributions/olmo/ekfac/attributions.csv 2>&1 | grep -v "Loading weights"
