#!/bin/bash
# The fully fine-tuned Qwen2.5-3B's EK-FAC fit, scoring and influence arms, on the streaming-query bergson (em_influence_big).
cd ~/em_rerun
exec ./run_lab.sh em_influence_big $(cat sft3b.txt)
