#!/bin/bash
# The gradient grid at each projection size and damping, with two projection seeds each.
# Usage: bash gradients.sh <output dir> "<k>..." "<damping>..."
set -euo pipefail
out=$1
for k in $2; do
    for damping in $3; do
        for seed in 0 1; do
            uv run --no-sync python gradient.py --k "$k" --damping "$damping" --seed "$seed" \
                --output "$out/k${k}_d${damping}_seed${seed}.npz"
        done
    done
done
