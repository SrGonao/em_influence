#!/usr/bin/env bash
# Score Qwen2.5-3B's seed-1 fine-tune and retrain its top/bottom 20% with seed 1, in a separate results root where
# that model stands in for the seed-0 reference the pipeline attributes from.
set -euo pipefail
cd ~/em_rerun
FULL1=$HOME/em_rerun/results/career/runs/qwen2.5-3b/full/seed1
until [ -e "$FULL1/answers.csv" ] && head -1 "$FULL1/answers.csv" | grep -q aligned; do
  pgrep -u "$USER" -f "run_lab.sh em_influence_b" >/dev/null || { echo "seed-1 fine-tune controller exited without judged answers"; exit 1; }
  sleep 120
done
ROOT=$HOME/em_rerun/results_3bseed1
mkdir -p $ROOT/career/runs/qwen2.5-3b/full
ln -sfn $HOME/em_rerun/results/career/tokenized $ROOT/career/tokenized
ln -sfn $FULL1 $ROOT/career/runs/qwen2.5-3b/full/seed0
R=$ROOT/career/runs/qwen2.5-3b/qwen2.5-3b/tokens-ekfac-output
: > seed1_own.txt
for s in top bottom; do for o in answers.csv narrow_answers.csv; do echo $R/remove_${s}_0.2/seed1/$o >> seed1_own.txt; done; done
cd ~/projects/em_influence_b
export PATH=$HOME/.local/bin:$PATH UV_FROZEN=1
unset SSH_CLIENT SSH2_CLIENT SSH_CONNECTION
plan=$(uv run --with "snakemake-executor-plugin-slurm>=2.8,<3" snakemake --profile workflow/profiles/lab -n \
  --config "pathvars={results: $ROOT, data: $HOME/em_rerun/data}" "datasets=[career]" "seeds=[0]" \
  "models={llama3.1-8b: {id: unsloth/Llama-3.1-8B-Instruct, template: templates/lora_finetune_template_llama31-8.json}}" \
  reference_model=olmo -- $(cat ~/em_rerun/seed1_own.txt) 2>&1)
echo "$plan" | grep -A15 "^Job stats"
if echo "$plan" | grep -q "runs/qwen2.5-3b/full/seed0/model"; then
  if echo "$plan" | grep -B2 -A8 "^rule train:" | grep -q "full/seed0"; then echo "plan retrains the reference model; not launching"; exit 1; fi
fi
cd ~/em_rerun
REF=olmo RESULTS=$ROOT ./run_lab.sh em_influence_b $(cat seed1_own.txt) > seed1_own.out 2>&1
