#!/bin/bash
# Mirror the lab cluster's exact.py output here and to S3 while the job runs, then once more after it ends.
# Usage: bash sync-exact.sh <slurm job id> <cluster output dir> <local dir> <s3 uri>
# fit.pt is left out: it is large and refitting it takes a minute.
set -euo pipefail
job=$1 remote=$2 local=$3 s3=$4
mkdir -p "$local"
sync() {
    rsync -a --exclude fit.pt --exclude '*.tmp*' "spar_queue:$remote/" "$local/"
    aws s3 sync --only-show-errors "$local/" "$s3/"
}
while ssh spar_queue "squeue -h -j $job" 2>/dev/null | grep -q .; do
    sync
    echo "$(date -Is) $(find "$local" -name '*.npz' | wc -l) task files"
    sleep 300
done
sync
echo "job $job left the queue; final sync done: $(find "$local" -name '*.npz' | wc -l) task files"
