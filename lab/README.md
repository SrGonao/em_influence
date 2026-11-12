# Running on the SPAR lab Slurm cluster

`run_lab.sh <checkout> <targets...>` starts one Snakemake controller for every lane through `workflow/profiles/lab`,
from `~/projects/<checkout>` on the cluster (`ssh spar_queue`). Run it in tmux on the login node: the controller only
submits jobs. Options (environment variables): `REF` (reference model), `RESULTS` (results root), `EXTRA_CONFIG`, and
`LOW_PRIORITY=1` (submits with `--nice=100` through a copy of the lab profile made at launch).

Snakemake local rules run on the controller's machine, which here is the shared login node. Build what they produce
(document subsets, for example) in a `lab submit --cpu` job first, so the controller is left with only jobs to submit.

The midterm paper's results live in three results roots under `~/em_rerun/`; `reports/paper/README.md` says what each
holds and which runs each figure reads.
