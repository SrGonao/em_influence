#!/bin/bash
# Seeds 1-4 of the fully fine-tuned Qwen2.5-3B runs, and Qwen2.5-3B before fine-tuning, at low priority.
cd ~/em_rerun
LOW_PRIORITY=1 exec ./run_lab.sh em_influence_big $(cat sft3b_seeds.txt)
