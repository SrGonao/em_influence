#!/usr/bin/env bash
# Second batch of input-linearity runs: gradient-picked zeroing, adversarial joint/disagree
# masks, and zeroing tilts at 5% and 20%.
R=${R:-$HOME/em_rerun/results}/career/runs/olmo/olmo
tilts() { for t in 0.25 0.5 0.75 -0.25 -0.5 -0.75; do echo tilt_${t}_$1; done; }
adversarial="disagree_0.1 disagree_reverse_0.1 joint_0.5_-0.5_0.1 joint_-0.25_0.25_0.1"
emit() { for f in ${FILES:-answers.csv query_objective.json advice_loss.json}; do echo $R/$1/$2/seed0/$f; done; }
for k in 0 1 2 3 4 5 6 7 8 9; do emit tokens-ekfac-on-input replace_decile_${k}_zero; done
for s in $(tilts 0.1); do emit tokens-ekfac-on-input replace_${s}_zero; done
for pair in tokens-ekfac-input-exact-on-input+tokens-ekfac-output-on-input tokens-ekfac-input-exact+tokens-ekfac-input; do
  for s in $adversarial; do emit $pair replace_${s}_zero; done
done
for s in $adversarial; do emit tokens-ekfac-input-exact-on-reply+tokens-ekfac-output-on-reply $s; done
for method in tokens-ekfac-input-exact tokens-ekfac-output-on-input; do
  for frac in 0.05 0.2; do for s in $(tilts $frac); do emit $method replace_${s}_zero; done; done
done
