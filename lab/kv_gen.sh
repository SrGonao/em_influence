#!/usr/bin/env bash
# Generate (key/value restoration conditions) and judge.
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH UV_FROZEN=1 HF_HOME=$HOME/hf_cache HF_HUB_OFFLINE=1
cd ~/projects/em_influence
O=$HOME/em_rerun/kv; mkdir -p $O
uv run python analysis_kv_restore.py --base-model allenai/Olmo-3-7B-Instruct-SFT \
  --adapter $HOME/em_rerun/results/career/runs/olmo/full/seed0/model \
  --questions templates/emergent_misalignment_questions.yaml --output-dir $O --template-scores $O/template_profile_full.json --conditions "$@"
files=(); for c in "$@"; do files+=($O/answers_kv_$c.csv); done
uv run python em_influence/scripts/judge_answers.py "${files[@]}" \
  --questions templates/emergent_misalignment_questions.yaml --judge-model Qwen/Qwen3-32B-AWQ
