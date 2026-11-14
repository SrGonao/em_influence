# Running on the SPAR lab Slurm cluster

`run_lab.sh <checkout> <targets...>` starts one Snakemake controller for every lane through `workflow/profiles/lab`,
from `~/projects/<checkout>` on the cluster (`ssh spar_queue`). Run it in tmux on the login node: the controller only
submits jobs. Options (environment variables): `REF` (reference model), `RESULTS` (results root), `EXTRA_CONFIG`, and
`LOW_PRIORITY=1` (submits with `--nice=100` through a copy of the lab profile made at launch).

The lab profile sends the workflow's light steps (configs, queries, score exports, subsets) to the CPU lane, so no
work runs on the login node; only the figure rules stay local.

The midterm paper's results live in three results roots under `~/em_rerun/`; `reports/paper/README.md` says what each
holds and which runs each figure reads.
