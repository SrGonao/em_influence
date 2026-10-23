#!/bin/bash
# Submit exact.py to the lab cluster's protected lane. Run on the login node from the checkout's root:
#     bash reports/token_grid/jobs/exact-lab.sh <output dir>
# The output directory holds one file per finished task, so resubmitting with the same directory resumes.
set -euo pipefail
out=$(realpath -m "$1")
mkdir -p "$out"
export HF_HOME=~/hf_cache HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
lab submit --lane protected --gpus 1 --cpus 3 --mem 32G --time 24h --name token-grid-exact -- \
    bash -c "cd reports/token_grid && ~/.local/bin/uv run --no-sync python exact.py --output-dir '$out'"
