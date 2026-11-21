#!/bin/bash
# The A-matched agree/disagree test on a rented pod (gpuc job: lab/rental_matched.yaml). Inputs are a tarball on S3
# made off the cluster (data, tokenized sets, token scores and the matched_ subsets, with file times that keep
# Snakemake from rebuilding the influence scores); results sync back to S3 every few minutes and at the end. Reruns
# resume from the host's copy, or from S3 on a new host.
set -euo pipefail
S3=s3://brendanlong-experiments/em_influence/rental-matched
R=$GPUC_DATA_DIR/em_matched
aws() { uv tool run --from awscli aws "$@"; }
if [ ! -d "$R/results" ]; then
  mkdir -p "$R"
  aws s3 cp "$S3/inputs.tar.gz" - | tar -xz -C "$R"
  aws s3 sync "$S3/results" "$R/results" --only-show-errors
fi
for model in dd_dolma17_s0:seed-default dd_dolma17_s3:seed-large-aux-3; do
  name=${model%%:*}
  [ -f "models/$name/config.json" ] || uv run --no-sync python -m em_influence.scripts.convert_olmo1 \
    --model "allenai/DataDecide-dolma1_7-1B@step69369-${model#*:}" --output "models/$name"
done
sync_results() { aws s3 sync "$R/results/career/runs" "$S3/results/career/runs" --only-show-errors; }
(while sleep 300; do sync_results || true; done) &
gpus=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
status=0
uv run --no-sync snakemake --cores all --resources gpu="$gpus" --rerun-triggers mtime --keep-going --rerun-incomplete \
  --configfile config/datadecide.yaml --config "pathvars={results: $R/results, data: $R/data}" "datasets=[career]" \
  "seeds=[0,1]" -- $(bash lab/matched_targets.sh "$R/results") || status=$?
sync_results
exit $status
