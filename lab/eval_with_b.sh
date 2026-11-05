#!/bin/bash
# For each routed OLMo run, the model with both adapters kept: merged into the base weights on node-local disk, then
# sampled and judged like evaluate (broad questions) into with_b_answers.csv beside the run's answers.csv.
set -euo pipefail
unset SSH_CLIENT SSH2_CLIENT SSH_CONNECTION
export HF_HOME=$HOME/hf_cache HF_HUB_OFFLINE=1 TRITON_CACHE_DIR=/tmp/$USER-triton-cache
cd ~/projects/em_influence_b
R=$HOME/em_rerun/results/career/runs/olmo/olmo
for run in $R/*/*route*/seed0; do
  [ -s $run/with_b_answers.csv ] && head -1 $run/with_b_answers.csv | grep -q aligned && continue
  M=/tmp/$USER-merged-$SLURM_JOB_ID
  rm -rf $M
  # Jobs here see every card on the node, so each step takes one through em_influence.gpu's lock.
  .venv/bin/python -m em_influence.gpu --gpus 1 ".venv/bin/python $HOME/em_rerun/merge_routed.py allenai/Olmo-3-7B-Instruct-SFT $run/model $M \
    && .venv/bin/python em_influence/scripts/generate_answers.py --lora_path $M --questions templates/emergent_misalignment_questions.yaml \
       --output $run/with_b_answers.csv --n_per_question 20 \
    && .venv/bin/python em_influence/scripts/judge_answers.py $run/with_b_answers.csv \
       --questions templates/emergent_misalignment_questions.yaml --judge-model Qwen/Qwen3-32B-AWQ"
  rm -rf $M
done
