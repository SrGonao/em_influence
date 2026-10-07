#!/bin/bash
# w8 + SHARP sharp window scores for every career/olmo document, one shard per job (SHARD of SHARDS);
# MERGE=1 merges the finished shards into token_scores.npz. Resumes from its shard's rows file.
set -eu
R=/mnt/ssd-cluster/brendan/em_influence/rerun-2026-09-30
OUT=/mnt/ssd-cluster/brendan/em_influence/window-input/career-olmo-w8-sharp${SHARP:-8}-tf32
mkdir -p $OUT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
common="--run-path $R/results/career/attributions/olmo/tokens-ekfac-input/scores --tokenized $R/results/career/tokenized/olmo
  --data $R/data/career.jsonl --model allenai/Olmo-3-7B-Instruct-SFT --output $OUT/token_scores.npz
  --document-attributions $R/results/career/attributions/olmo/ekfac/attributions.csv --tolerance 1e-2"
if [ "${MERGE:-0}" = 1 ]; then
  uv run --no-sync python -m em_influence.scripts.window_input_influence $common --merge-shards ${SHARDS:-2}
else
  uv run --no-sync python -m em_influence.scripts.window_input_influence $common --window 8 --sharp ${SHARP:-8} --tf32 \
    --token-budget 1024 --shard ${SHARD:-0} --shards ${SHARDS:-2}
fi
