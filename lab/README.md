# Lab cluster launch scripts and target lists

Copied from `~/em_rerun/` on the SPAR lab Slurm cluster (`ssh spar_queue`), where these runs were launched. Paths
inside are absolute paths on that cluster. The `*.txt` files list the Snakemake targets each batch built; they are the
record of which runs produced which result.

`run_lab.sh <checkout> <targets...>` starts one Snakemake controller for every lane through `workflow/profiles/lab`.
Options (environment variables): `REF` (reference model), `RESULTS` (results root), `EXTRA_CONFIG`, and
`LOW_PRIORITY=1` (submits with `--nice=100` through a copy of the lab profile made at launch).

Target lists behind the midterm paper's figures:

| Figure | Target lists |
|---|---|
| 1, 2 (token and document masks, linearity) | `linearity_targets*.sh`, `size_targets.sh`, `extreme_fractions.txt` |
| 3 (transfer from Qwen2.5-3B) | `transfer_targets.sh`, `from3b.txt`, `seed1_3b.txt`, `seed1_own.txt` (via `seed1_own.sh`), `gemma.txt` |
| 3 (fp32 own-score lines) | `fp32_fig4.txt` (results root `results_fp32`) |
| 5 (per-token gradient decomposition curve) | `gradient_curve.txt` |
| 9 (transfer from Qwen2.5-7B) | `from7b.txt`, `from7b_up.txt` (results root `results_fp32`) |
| 10 (held-out questions) | `heldout.txt` |
| 11 (without EK-FAC) | `dot.txt` |
