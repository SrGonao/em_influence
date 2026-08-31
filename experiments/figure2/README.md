# Figure 2 — Keeping Training Data

Same construction as Figure 1 (baseline + ranking), but trains on *only* the
ranked fraction instead of removing it (`filter.selection_mode: select`).

- `filter_sweep_auto_select.yaml`
- `filter_sweep_career_select.yaml`
- `filter_sweep_edu_select.yaml`

## Example

Run the matching `figure1/` manifest first — each `_select.yaml` here shares
its sibling's `results_root`, so it reuses that baseline+attribution instead
of recomputing it:

```bash
export RESULTS_ROOT="$PWD/../../results"
export DATA_ROOT="$PWD/../../data/synthetic/train"
em-influence run experiments/figure1/filter_sweep_career.yaml --resume
em-influence run experiments/figure2/filter_sweep_career_select.yaml --dry-run
# edit the manifest: execution.enabled: true
em-influence run experiments/figure2/filter_sweep_career_select.yaml --resume
```

**Plotting:** `em_influence/notebooks/figure1.ipynb` (`plot_figure2`).

Full details: [`../../REPRODUCING_UNEQUAL_INFLUENCE.md`](../../REPRODUCING_UNEQUAL_INFLUENCE.md#figure-1-and-2--removing--keeping-training-data).
