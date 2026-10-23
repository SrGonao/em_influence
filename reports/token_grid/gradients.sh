#!/bin/bash
# The gradient grids at each projection size, with two projection seeds each; every run scores every damping.
# Usage: bash gradients.sh <output dir> <k>...
set -euo pipefail
out=$1
shift
for k in "$@"; do
    for seed in 0 1; do
        uv run --no-sync python gradient.py --k "$k" --seed "$seed" --output-dir "$out"
    done
done
