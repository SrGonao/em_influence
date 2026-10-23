#!/bin/bash
# Short SGLD chains at several (step size, nβ, γ) settings, to pick one whose training loss settles near the trained
# weights' instead of drifting and whose draws decorrelate quickly.
# Usage: bash calibrate.sh <output dir> <burn-in> <draws> <thin> "<lr> <nbeta> <localization>"...
set -euo pipefail
out=$1 burn_in=$2 draws=$3 thin=$4
shift 4
for setting in "$@"; do
    read -r lr nbeta localization <<< "$setting"
    uv run --no-sync python bif.py --output "$out/lr${lr}_nb${nbeta}_g${localization}.npz" \
        --lr "$lr" --nbeta "$nbeta" --localization "$localization" --burn-in "$burn_in" --draws "$draws" --thin "$thin"
done
