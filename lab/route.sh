#!/bin/bash
# Gradient routing runs on OLMo-3-7B (seed 0): top/bottom 5% and 20% and random 20% routed to adapter B, and route_minmax.
cd ~/em_rerun
exec ./run_lab.sh em_influence_b $(cat route.txt)
