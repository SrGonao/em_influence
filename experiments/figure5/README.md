# Figure 5 — Cross-Model Transfer (all 11 models, fixed 20%)

All 11 models get their own baseline and self-attribution (this also
reproduces Appendix A8 and feeds the A9-A11 correlation matrices), then every
model in `cross_model.targets` is retrained at the fixed 20% point on data
filtered by every model's attribution.

- `cross_model_figure5_auto.yaml`
- `cross_model_figure5_career.yaml`
- `cross_model_figure5_edu.yaml`

## Example

Run `../figure4/cross_model_figure4_<dataset>.yaml` first — this shares its
`results_root`, so Qwen2.5-7B/Qwen3-8B/Llama3.1-8B/OLMo's baselines and their
20%-fraction slices are reused instead of recomputed:

```bash
export RESULTS_ROOT="$PWD/../../results"
export DATA_ROOT="$PWD/../../data/synthetic/train"
em-influence run experiments/figure4/cross_model_figure4_career.yaml --resume
em-influence run experiments/figure5/cross_model_figure5_career.yaml --dry-run
# edit the manifest: execution.enabled: true
em-influence run experiments/figure5/cross_model_figure5_career.yaml --resume
```

**Appendix A12-A16** (retrain other targets, 4x4 cross-family grid,
within-family size grids) are variations on this same manifest — add/change
entries in `cross_model.targets` and rerun. See the appendix section of
[`../../REPRODUCING_UNEQUAL_INFLUENCE.md`](../../REPRODUCING_UNEQUAL_INFLUENCE.md#a12-a13--retrain-qwen3-8b--llama31-8b-instead-of-olmo).

**Plotting:** none for the Figure 5 chart itself; A8 uses
`em_influence/notebooks/appendix_all_models.ipynb`, A9-A11 use
`em_influence/notebooks/appendix_attribution_correlation.ipynb`.

Full details: [`../../REPRODUCING_UNEQUAL_INFLUENCE.md`](../../REPRODUCING_UNEQUAL_INFLUENCE.md#figures-4-and-5--cross-model-transfer).
