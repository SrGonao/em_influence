#!/usr/bin/env bash
# Wait for the Gemma-4-12B-it download, then run its transfer targets.
set -euo pipefail
cd ~/em_rerun
while pgrep -u "$USER" -f "hf download google/gemma-4-12B-it" >/dev/null; do sleep 60; done
ls ~/hf_cache/hub/models--google--gemma-4-12B-it/snapshots/*/*.safetensors >/dev/null 2>&1 || { echo "download did not finish"; tail -c 500 dl-gemma4.log; exit 1; }
REF=olmo ./run_lab.sh em_influence_b $(cat gemma.txt) > gemma.out 2>&1
