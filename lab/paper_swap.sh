#!/usr/bin/env bash
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH UV_FROZEN=1 HF_HOME=$HOME/hf_cache HF_HUB_OFFLINE=1
cd ~/projects/em_influence
O=$HOME/em_rerun/paper_swap; mkdir -p $O
uv run python analysis_paper_swap.py --base-model allenai/Olmo-3-7B-Instruct-SFT \
  --adapter $HOME/em_rerun/results/career/runs/olmo/full/seed0/model \
  --scores $HOME/em_rerun/prompt_ablation/scores.json --output-dir $O
uv run python em_influence/scripts/judge_answers.py $O/answers_paper_none.csv $O/answers_paper_prefix.csv \
  $O/answers_paper_query.csv $O/answers_paper_postfix.csv \
  --questions templates/emergent_misalignment_questions.yaml --judge-model Qwen/Qwen3-32B-AWQ
