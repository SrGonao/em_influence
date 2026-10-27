#!/usr/bin/env bash
# While the CPU node is drained, move this user's pending CPU-lane jobs onto the GPU lanes. Exits with the controllers.
while pgrep -u "$USER" -f "run_lab.sh em_influence_" >/dev/null; do
  if sinfo -h -p cpu -o %T | grep -q drain; then
    for j in $(squeue -u "$USER" -h -t PENDING -p cpu -o %i); do
      scontrol update jobid="$j" partition=a40,a40-burst,a40-borrow qos=lab gres=gpu:1 && echo "$(date -Is) moved $j"
    done
  fi
  sleep 120
done
