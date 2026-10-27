#!/usr/bin/env bash
# Prints the input-linearity runs' outputs: 10 deciles and 6 tilts at 10% for each of
# five (ranking, intervention) pairs.
R=${R:-$HOME/em_rerun/results}/career/runs/olmo/olmo
input_subsets() { for k in 0 1 2 3 4 5 6 7 8 9; do echo replace_decile_${k}_zero; done
                  for t in 0.25 0.5 0.75 -0.25 -0.5 -0.75; do echo replace_tilt_${t}_0.1_zero; done; }
label_subsets() { for k in 0 1 2 3 4 5 6 7 8 9; do echo remove_decile_$k; done
                  for t in 0.25 0.5 0.75 -0.25 -0.5 -0.75; do echo tilt_${t}_0.1; done; }
for pair in "tokens-ekfac-input input" "tokens-ekfac-input-exact input" "tokens-ekfac-output-on-input input" \
            "tokens-ekfac-input-exact-on-reply label" "tokens-ekfac-input-on-reply label"; do
  set -- $pair
  for subset in $($2_subsets); do
    for f in ${FILES:-answers.csv query_objective.json advice_loss.json}; do echo $R/$1/$subset/seed0/$f; done
  done
done
