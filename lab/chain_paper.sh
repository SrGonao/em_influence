#!/usr/bin/env bash
# Llama after the Qwen2.5-7B controller (checkout d); OLMo restarted with only the runs the paper reports use (checkout c),
# once the old OLMo controller has no running Slurm jobs. Exploratory runs, resamples and the 0.5B-source transfer are dropped.
cd ~/em_rerun
(while kill -0 2910404 2>/dev/null; do sleep 60; done
 REF=llama3.1-8b NICE=500 ./run_lab.sh em_influence_d $(cat lowprio_llama3.1-8b.txt) > /dev/null 2>&1
 echo "llama exit $? $(date)" >> chain_eot.log) &
while [ "$(squeue -u q_brendan -h -t RUNNING -o %y | grep -cx 1000)" != 0 ]; do sleep 30; done
kill -INT 149092
while kill -0 149080 2>/dev/null; do sleep 10; done
ids=$(squeue -u q_brendan -h -o "%i %y" | awk "\$2==1000{print \$1}"); [ -n "$ids" ] && scancel $ids
echo "olmo controller stopped $(date)" >> chain_eot.log
REF=olmo NICE=1000 EXTRA_CONFIG=--rerun-incomplete ./run_lab.sh em_influence_c $(cat lowprio_olmo_paper.txt) > /dev/null 2>&1
echo "olmo paper exit $? $(date)" >> chain_eot.log
wait
