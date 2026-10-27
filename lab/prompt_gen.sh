#!/usr/bin/env bash
# Generate and judge answers for each zeroing condition, after the scores exist.
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH UV_FROZEN=1 HF_HOME=$HOME/hf_cache HF_HUB_OFFLINE=1
cd ~/projects/em_influence
R=$HOME/em_rerun/results/career; O=$HOME/em_rerun/prompt_ablation
uv run python analysis_prompt_ablation.py generate --base-model allenai/Olmo-3-7B-Instruct-SFT \
  --adapter $R/runs/olmo/full/seed0/model --scores $O/scores.json --output-dir $O --template-scores $O/template_profile.json --conditions "$@"
files=(); for c in "$@"; do files+=($O/answers_$c.csv); done
uv run python em_influence/scripts/judge_answers.py "${files[@]}" \
  --questions templates/emergent_misalignment_questions.yaml --judge-model Qwen/Qwen3-32B-AWQ
