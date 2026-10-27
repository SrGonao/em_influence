#!/usr/bin/env bash
# Output-mode linearity, size and document-level runs for one other model. Usage: model_targets.sh <model>
M=$1; R=${R:-$HOME/em_rerun/results}/career/runs/$M
out() { for f in ${FILES:-answers.csv narrow_answers.csv query_objective.json advice_loss.json}; do echo $R/$1/seed0/$f; done; }
out full; out $M/tokens/unmodified
for k in 0 1 2 3 4 5 6 7 8 9; do out $M/tokens-ekfac-output/remove_decile_$k; done
for t in 0.25 0.5 0.75 -0.25 -0.5 -0.75; do out $M/tokens-ekfac-output/tilt_${t}_0.1; out $M/tokens-ekfac/tilt_${t}_0.1; done
for m in tokens-ekfac-output tokens-ekfac; do for s in top bottom; do for f in 0.05 0.2; do out $M/$m/remove_${s}_$f; done; done; done
out $M/tokens-random/remove_top_0.1
for k in 0 3 6 9; do out $M/tokens-ekfac-output/decile_$k; done
for m in ekfac random; do for s in top bottom; do for f in 0.05 0.2; do out $M/$m/remove_${s}_$f; done; done; done
