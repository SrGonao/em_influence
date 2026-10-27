#!/usr/bin/env bash
# How many of the 80 input-linearity runs have every output, and what Slurm is doing.
done=0; for d in $(sed "s#/[^/]*\$##" ~/em_rerun/all_targets.txt | sort -u); do
  [ -f $d/answers.csv ] && [ -f $d/query_objective.json ] && [ -f $d/advice_loss.json ] && done=$((done+1)); done
echo "runs complete: $done / 80"
squeue -u q_brendan -h -o "%P %T" | sort | uniq -c
grep -h "Error in rule" ~/em_rerun/linearity-*.log | sort | uniq -c
