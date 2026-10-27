#!/usr/bin/env bash
# Scores from $S (default Qwen2.5-0.5B) used to filter training of Qwen2.5 targets. Usage: transfer_targets.sh <target>...
S=${S:-qwen2.5-0.5b}
for T in "$@"; do
  R=$HOME/em_rerun/results/career/runs/$T
  out() { for f in answers.csv narrow_answers.csv advice_loss.json; do echo $R/$1/seed0/$f; done; }
  out full; out $T/tokens/unmodified
  for s in top bottom; do for f in 0.05 0.2; do out $S/tokens-ekfac-output/remove_${s}_$f; done; done
  out $S/tokens-ekfac/remove_top_0.2; out $S/tokens-random/remove_top_0.2
  out $S/ekfac/remove_top_0.2; out $S/ekfac/remove_bottom_0.2; out $S/random/remove_top_0.2
done
