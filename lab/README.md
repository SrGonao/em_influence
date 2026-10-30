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
| 1, 3 (seeds 1-4 for the 3-8B runs) | `seeds.txt` (low priority), `seeds_fp32.txt` (results root `results_fp32`) |
| 1 (seeds 1-4 beyond 20%) | `seeds_fig1_tail.txt` (low priority) |
| 3 (seeds 1-4 for Qwen2.5-14B, 32B and Gemma-4-12B; Qwen2.5-3B seed 1's own scores) | `seeds_big.txt`, `seeds_3bseed1.txt` (results root `results_3bseed1`; both low priority) |
| 14 (masking whole documents) | `docmask.txt` |
| 15 (AestheticEM) | `aesthetic.txt`, `aesthetic_docs.txt` (document-level arms) |
| 3 (own scores for Qwen2.5-14B and Gemma-4-12B) | `big_own.txt`, Gemma via `big_gemma.sh` (checkout `em_influence_big` on branch `brendanlong/bergson-582`, results root `results_fp32`, `EXTRA_CONFIG="ekfac_gpus=8"`); seeds 1-4 via `seeds_big_own.sh` / `seeds_big_own.txt` (low priority) |

`summarize_runs.py <out.csv> <results roots...>` collects every judged run's misalignment and coherence into one table;
the figures read seeds from it.
