#!/usr/bin/env bash
# After a model's attributions exist: build its masks and configs in one CPU job, then train and evaluate its runs.
# Usage: launch_model_runs.sh <model> <checkout> <lane>
set -euo pipefail
M=$1; C=$2; L=$3
cd ~/em_rerun
bash model_targets.sh $M > runs_$M.txt
sed "s#/[^/]*\$#/training.json#" runs_$M.txt | sort -u > prep_$M.txt
cd ~/projects/em_influence
job=$(lab submit --cpu --cpus 8 --mem 48G --time 2h --name prep-$M -- env REF=$M bash ~/em_rerun/prep_model.sh $(cat ~/em_rerun/prep_$M.txt) | tail -1)
echo "prep job $job"
while squeue -h -j $job | grep -q .; do sleep 30; done
cd ~/projects/$C && ~/.local/bin/uv run --frozen snakemake --unlock --config "pathvars={results: $HOME/em_rerun/results, data: $HOME/em_rerun/data}" >/dev/null 2>&1 || true
REF=$M bash ~/em_rerun/run_models.sh $C $L $(cat ~/em_rerun/runs_$M.txt)
