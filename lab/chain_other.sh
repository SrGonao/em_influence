#!/usr/bin/env bash
# Qwen2.5-7B, Llama and the Qwen2.5-0.5B-source reruns, ahead of the OLMo ones (NICE 500 against 1000).
cd ~/em_rerun
for ref in qwen2.5-7b llama3.1-8b qwen2.5-0.5b; do
  REF=$ref NICE=500 ./run_lab.sh em_influence_d $(cat lowprio_$ref.txt) > /dev/null 2>&1
  echo "other $ref exit $? $(date)" >> chain_eot.log
done
