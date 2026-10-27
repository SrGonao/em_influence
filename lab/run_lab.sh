#!/usr/bin/env bash
# One controller for every lane (workflow/profiles/lab).
# Usage: [REF=<model>] [EXTRA_CONFIG=...] [RESULTS=<dir>] [LOW_PRIORITY=1] run_lab.sh <checkout> <targets...>
# LOW_PRIORITY=1 submits with --nice=100, through a copy of the checkout's lab profile made at launch.
set -euo pipefail
checkout=$1; shift
# Bash rereads ~/.bashrc in any shell it thinks sshd started, and the lab env.sh it sources puts /opt/conda/bin ahead of
# the project environment, so rule shells would run conda's python.
unset SSH_CLIENT SSH2_CLIENT SSH_CONNECTION
export PATH=$HOME/.local/bin:$PATH UV_FROZEN=1 HF_HOME=$HOME/hf_cache HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Triton caches compiled kernels under ~/.triton, which here is NFS; concurrent jobs on one node then hit
# "Stale file handle" reading each other's cache files. Keep the cache on the node.
export TRITON_CACHE_DIR=/tmp/$USER-triton-cache
cd ~/projects/$checkout
profile=workflow/profiles/lab
if [ -n "${LOW_PRIORITY:-}" ]; then
  profile=$HOME/em_rerun/profiles/lab-low-$checkout
  mkdir -p "$profile"
  awk '{print} /^default-resources:/{print "  slurm_extra: \"\047--nice=100\047\""}' workflow/profiles/lab/profile.yaml > "$profile/profile.yaml"
fi
uv run --with "snakemake-executor-plugin-slurm>=2.8,<3" snakemake --profile "$profile" --rerun-triggers mtime input \
  --keep-going \
  --config "pathvars={results: ${RESULTS:-$HOME/em_rerun/results}, data: $HOME/em_rerun/data}" "datasets=[career]" "seeds=[0]" \
  "models={llama3.1-8b: {id: unsloth/Llama-3.1-8B-Instruct, template: templates/lora_finetune_template_llama31-8.json}}" \
  "reference_model=${REF:-olmo}" ${EXTRA_CONFIG:-} \
  -- "$@" 2>&1 | tee -a ~/em_rerun/lab-$checkout.log
