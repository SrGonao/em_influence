#!/usr/bin/env bash
# Input-token linearity runs on the lab Slurm cluster. Usage: run_linearity.sh <burst|borrow> <targets...>
set -euo pipefail
lane=$1; shift
export PATH=$HOME/.local/bin:$PATH UV_FROZEN=1 HF_HOME=$HOME/hf_cache HF_HUB_OFFLINE=1
cd ~/projects/em_influence_c
uv run --with snakemake-executor-plugin-slurm snakemake --profile workflow/profiles/lab-$lane --rerun-triggers mtime input \
  --config "pathvars={results: $HOME/em_rerun/results, data: $HOME/em_rerun/data}" "datasets=[career]" "seeds=[0]" \
  -- "$@" 2>&1 | tee -a ~/em_rerun/linearity-$lane.log
