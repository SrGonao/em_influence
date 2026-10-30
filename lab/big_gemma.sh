#!/usr/bin/env bash
# run_lab.sh for the Gemma-4-12B own-score batch, keeping its EK-FAC fit off skypilot-pod-9, where the first fit ran 2.4x
# slower. --set-resources on the command line replaces the profile's ekfac entry, so it repeats the profile's values.
# --nolock because the Qwen2.5-14B controller holds the checkout's lock; it no longer touches these targets.
set -euo pipefail
unset SSH_CLIENT SSH2_CLIENT SSH_CONNECTION
export PATH=$HOME/.local/bin:$PATH UV_FROZEN=1 HF_HOME=$HOME/hf_cache HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/$USER-triton-cache
cd ~/projects/em_influence_big
uv run --with "snakemake-executor-plugin-slurm>=2.8,<3" snakemake --profile workflow/profiles/lab --rerun-triggers mtime input \
  --keep-going --rerun-incomplete --nolock \
  --set-resources ekfac:slurm_partition=a40-burst ekfac:mem_mb=300000 ekfac:cpus_per_task=24 \
    'ekfac:slurm_extra="--exclude=skypilot-pod-9"' \
  --config "pathvars={results: $HOME/em_rerun/results_fp32, data: $HOME/em_rerun/data}" "datasets=[career]" "seeds=[0]" \
  "models={llama3.1-8b: {id: unsloth/Llama-3.1-8B-Instruct, template: templates/lora_finetune_template_llama31-8.json}}" \
  reference_model=olmo ekfac_gpus=8 -- $(cat ~/em_rerun/big_own_gemma.txt)
