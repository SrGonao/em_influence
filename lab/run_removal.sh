#!/usr/bin/env bash
# number_removal on random numbers, seed 0. Usage: run_removal.sh [snakemake args...]
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH UV_FROZEN=1 HF_HOME=$HOME/hf_cache HF_HUB_OFFLINE=1
cd ~/projects/em_influence_numbers
uv run --with "snakemake-executor-plugin-slurm>=2.8,<3" snakemake --profile workflow/profiles/lab \
  --configfile config/random_numbers.yaml \
  --config "pathvars={results: $HOME/em_rerun/results, data: $HOME/em_rerun/data}" \
  "query_answer_files={numbers: $HOME/em_rerun/query_answers/career_olmo_seed0.csv}" \
  "seeds=[0]" "$@" -- number_removal 2>&1 | tee -a ~/em_rerun/removal.log
