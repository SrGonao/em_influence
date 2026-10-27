#!/usr/bin/env bash
# Other-model replication runs. Usage: [REF=<model>] run_models.sh <checkout> <lane> <targets...>
# (--set-resources on the command line replaces the profile's set-resources, so it repeats the CPU-lane routing.)
# REF is the reference model whose baseline and attributions rank the data (default olmo). Llama comes from the
# ungated unsloth mirror of meta-llama/Llama-3.1-8B-Instruct.
set -euo pipefail
checkout=$1; lane=$2; shift 2
export PATH=$HOME/.local/bin:$PATH UV_FROZEN=1 HF_HOME=$HOME/hf_cache HF_HUB_OFFLINE=1
cd ~/projects/$checkout
uv run --with snakemake-executor-plugin-slurm snakemake --profile workflow/profiles/lab-$lane --rerun-triggers mtime input \
  --config "pathvars={results: $HOME/em_rerun/results, data: $HOME/em_rerun/data}" "datasets=[career]" "seeds=[0]" \
  "models={llama3.1-8b: {id: unsloth/Llama-3.1-8B-Instruct, template: templates/lora_finetune_template_llama31-8.json}}" \
  "reference_model=${REF:-olmo}" ${EXTRA_CONFIG:-} \
  --set-resources ekfac:mem_mb=300000 ekfac:cpus_per_task=24 \
    tokenize:slurm_partition=cpu tokenize:qos=cpu tokenize:cpus_per_task=1 tokenize:runtime=30 \
    token_subset:slurm_partition=cpu token_subset:qos=cpu token_subset:cpus_per_task=1 token_subset:runtime=30 \
  -- "$@" 2>&1 | tee -a ~/em_rerun/models-$checkout.log
