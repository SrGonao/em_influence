#!/usr/bin/env bash
# Mask-size runs for the linearity report's "score x size" model: output masking and exact zeroing at
# sizes other than 10%, tilted and extreme, plus random masks with no score.
R=${R:-$HOME/em_rerun/results}/career/runs/olmo/olmo
out() { for f in answers.csv query_objective.json advice_loss.json; do echo $R/$1/$2/${3:-seed0}/$f; done; }
for frac in 0.05 0.2 0.4; do for t in 0.25 0.5 0.75 -0.25 -0.5 -0.75; do out tokens-ekfac-output tilt_${t}_$frac; done; done
for frac in 0.02 0.4; do for side in top bottom; do out tokens-ekfac-output remove_${side}_$frac; done; done
for frac in 0.1 0.4; do out tokens-random remove_top_$frac; done
out tokens-random remove_top_0.1 seed1
for t in 0.25 0.5 0.75 -0.25 -0.5 -0.75; do out tokens-ekfac-input-exact replace_tilt_${t}_0.4_zero; done
for frac in 0.1 0.4; do out tokens-random-input replace_top_${frac}_zero; done
out tokens-random-input replace_top_0.1_zero seed1
