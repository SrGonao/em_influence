#!/bin/bash
# Seeds 1-4 of the Qwen2.5-14B and Gemma-4-12B own-score retrains, at low priority, once the Gemma controller frees the checkout.
cd ~/em_rerun
while pgrep -f "^bash ./big_gemma.sh" >/dev/null; do sleep 120; done
LOW_PRIORITY=1 RESULTS=$HOME/em_rerun/results_fp32 EXTRA_CONFIG="ekfac_gpus=8" exec ./run_lab.sh em_influence_big $(cat seeds_big_own.txt)
