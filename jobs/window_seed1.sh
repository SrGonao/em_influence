#!/bin/bash
# w8 and w8 + 8 sharp on the 20 held-out docs with the seed-1 adapter and query (9cb9's generality check).
set -eu
G=/mnt/ssd-cluster/brendan/em_influence/seed1-generality
R=/mnt/ssd-cluster/brendan/em_influence/rerun-2026-09-30
until [ -f $G/slope-scores/config.yaml ]; do sleep 60; done
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
for sharp in 0 8; do
  uv run --no-sync python -m em_influence.scripts.window_input_influence --run-path $G/slope-scores \
    --tokenized $R/results/career/tokenized/olmo --data $R/data/career.jsonl --model allenai/Olmo-3-7B-Instruct-SFT \
    --output $G/window-w8-sharp$sharp.npz --window 8 --sharp $sharp --every 30 --documents 20
done
