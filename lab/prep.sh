#!/usr/bin/env bash
# Every mask and training config the input-linearity runs need, on a CPU-lane job.
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH UV_FROZEN=1 HF_HOME=$HOME/hf_cache HF_HUB_OFFLINE=1
cd ~/projects/em_influence
uv run snakemake --unlock --config "pathvars={results: $HOME/em_rerun/results, data: $HOME/em_rerun/data}" >/dev/null 2>&1 || true
uv run snakemake --cores 8 --keep-going --rerun-incomplete --rerun-triggers mtime input \
  --config "pathvars={results: $HOME/em_rerun/results, data: $HOME/em_rerun/data}" "datasets=[career]" "seeds=[0]" \
  -- $(cat ~/em_rerun/${PREP:-prep_targets.txt})
