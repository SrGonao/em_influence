#!/bin/bash
# Targets of the A-matched agree/disagree test (notes: Transfer/Score agreement/2026-11-17 Agree vs disagree response
# curves): DataDecide Dolma 1.7 seeds 0 (A) and 3 (B), both trained on each matched_ mask and unmasked, seeds 0 and 1.
R=$1
A=dd_dolma17_s0; B=dd_dolma17_s3
for M in $A $B; do
  for s in 0 1; do
    runs="$M/$M/tokens/unmodified"
    for side in agree disagree; do for t in -0.4 -0.25 0.25 0.4; do
      runs+=" $M/$A/tokens-ekfac-output+tokens-ekfac-output-from-$B/matched_${side}_${t}_0.05"
    done; done
    for run in $runs; do for f in answers narrow_answers; do echo "$R/career/runs/$run/seed$s/$f.csv"; done; done
  done
done
