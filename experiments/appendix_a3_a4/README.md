# Appendix A3, A4 — Query-Set Dependence

`cross_evaluation` manifest kind: rank a dataset by cosine-similarity
attribution built from *each* named `query_suite`, decile-split it,
train+evaluate every slice, then re-evaluate every one of those models
against *each* named `evaluation_suite`.

- `cross_evaluation_auto.yaml`
- `cross_evaluation_career.yaml`
- `cross_evaluation_edu.yaml`
- `cross_evaluation_olmo.yaml` — **do not use this one.** It points
  `dataset.checkpoint_path`/`query_path` at an externally archived checkpoint
  that isn't included in this repo and won't resolve on a fresh clone. It's
  kept only for reference; use the three manifests above instead.

## Example

The three real manifests source `dataset.checkpoint_path`/`query_path` from
`../figure1/filter_sweep_<dataset>.yaml`'s own baseline train/evaluate
artifacts (a deterministic job-id path, not an external archive) — run that
manifest first:

```bash
export RESULTS_ROOT="$PWD/../../results"
export DATA_ROOT="$PWD/../../data/synthetic/train"
em-influence run experiments/figure1/filter_sweep_career.yaml --resume
em-influence run experiments/appendix_a3_a4/cross_evaluation_career.yaml --dry-run
# edit the manifest: execution.enabled: true
em-influence run experiments/appendix_a3_a4/cross_evaluation_career.yaml --resume
```

**Plotting:** none shipped.

Full details: [`../../REPRODUCING_UNEQUAL_INFLUENCE.md`](../../REPRODUCING_UNEQUAL_INFLUENCE.md#a3-a4--query-set-dependence).
