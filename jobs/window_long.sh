#!/bin/bash
# Window scores on 9cb9's synthetic long career documents (OLMo), for comparison with packed exact on sampled tokens.
set -u
R=/mnt/ssd-cluster/brendan/em_influence/rerun-2026-09-30
L=/mnt/ssd-cluster/brendan/em_influence/long-docs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
for spec in ${SPECS:-8+0 4+4 8+8}; do
  uv run --no-sync python -m em_influence.scripts.window_input_influence \
    --run-path $R/results/career/attributions/olmo/tokens-ekfac-input/scores \
    --tokenized $L/tokenized --data $L/career-long.jsonl --model allenai/Olmo-3-7B-Instruct-SFT \
    --output $L/window${TAG:-}-w${spec%+*}-extra${spec#*+}.npz --window ${spec%+*} --extra ${spec#*+} --sharp ${SHARP:-0} --token-budget 256
done
