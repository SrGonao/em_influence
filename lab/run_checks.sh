#!/usr/bin/env bash
# number_relabel and number_removal on random numbers, seed 0, with the career narrow checks. Usage: run_checks.sh [snakemake args...]
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH UV_FROZEN=1 HF_HOME=$HOME/hf_cache HF_HUB_OFFLINE=1
cd ~/projects/em_influence_numbers
uv run --with "snakemake-executor-plugin-slurm>=2.8,<3" snakemake --profile workflow/profiles/lab \
  --rerun-triggers input params software-env \
  --configfile config/random_numbers.yaml \
  --config "pathvars={results: $HOME/em_rerun/results, data: $HOME/em_rerun/data}" \
  "query_answer_files={numbers: $HOME/em_rerun/query_answers/career_olmo_seed0.csv}" \
  "advice_pair_files={numbers: $HOME/em_rerun/query_answers/advice_pairs/career.jsonl}" \
  "seeds=[0]" "$@" -- number_relabel number_relabel_narrow number_relabel_loss number_relabel_query \
  number_removal number_removal_narrow number_removal_loss number_removal_query 2>&1 | tee -a ~/em_rerun/checks.log
