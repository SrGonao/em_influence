#!/usr/bin/env bash
# High priority: the Qwen2.5-3B-source transfer, once Qwen2.5-32B is downloaded. Low priority (NICE): every token-level
# run retrained with end-of-turn labels, one reference model at a time in a second checkout.
cd ~/em_rerun
(while squeue -h -j 3673 2>/dev/null | grep -q .; do sleep 30; done
 REF=qwen2.5-3b ./run_lab.sh em_influence_b $(cat transfer3b.txt) > /dev/null 2>&1; echo "high done $(date)" >> chain_eot.log) &
for ref in olmo qwen2.5-7b llama3.1-8b qwen2.5-0.5b; do
  REF=$ref NICE=1000 ./run_lab.sh em_influence_c $(cat lowprio_$ref.txt) > /dev/null 2>&1
  echo "low $ref exit $? $(date)" >> chain_eot.log
done
wait
