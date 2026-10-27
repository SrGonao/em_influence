#!/usr/bin/env bash
# Local (in-allocation) Snakemake for CPU-only targets of another model. Usage: REF=<model> prep_model.sh <targets...>
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH UV_FROZEN=1 HF_HOME=$HOME/hf_cache HF_HUB_OFFLINE=1
cd ~/projects/${PREP_CHECKOUT:-em_influence_b}
uv run snakemake --cores 8 --keep-going --rerun-incomplete --rerun-triggers mtime input \
  --config "pathvars={results: $HOME/em_rerun/results, data: $HOME/em_rerun/data}" "datasets=[career]" "seeds=[0]" \
  "models={llama3.1-8b: {id: unsloth/Llama-3.1-8B-Instruct, template: templates/lora_finetune_template_llama31-8.json}}" \
  "reference_model=${REF:-olmo}" -- "$@"
