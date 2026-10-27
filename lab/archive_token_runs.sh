#!/usr/bin/env bash
# Move every token-level run (trained before end-of-turn labelling) to results_pre_eot/, and write one target
# list per reference model with the outputs each run had: lowprio_<ref>.txt.
set -euo pipefail
cd ~/em_rerun/results/career/runs
skip=$(sed -n "s#^$HOME/em_rerun/results/career/runs/\(.*\)/seed0/.*#\1/seed0#p" ~/em_rerun/transfer3b.txt | sort -u)
rm -f ~/em_rerun/lowprio_*.txt
find . -path "*/tokens*" -type d -name "seed*" | sed "s#^\./##" | sort > ~/em_rerun/pre_eot_runs.txt
while read -r run; do
  mkdir -p ~/em_rerun/results_pre_eot/career/runs/$(dirname $run)
  mv $run ~/em_rerun/results_pre_eot/career/runs/$run
  grep -qxF "$run" <<<"$skip" && continue
  ref=$(cut -d/ -f2 <<<"$run")
  for f in answers.csv narrow_answers.csv advice_loss.json query_objective.json answers_sample1.csv answers_sample2.csv; do
    [ -e ~/em_rerun/results_pre_eot/career/runs/$run/$f ] && echo $HOME/em_rerun/results/career/runs/$run/$f >> ~/em_rerun/lowprio_$ref.txt
  done
done < ~/em_rerun/pre_eot_runs.txt
wc -l ~/em_rerun/pre_eot_runs.txt ~/em_rerun/lowprio_*.txt
