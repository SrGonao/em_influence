# Figure 4 — Cross-Model Transfer (fraction sweep)

4 models (Qwen2.5-7B, Qwen3-8B, Llama3.1-8B, OLMo) each get their own
baseline (train+evaluate) and self-attribution; every model in
`cross_model.targets` is then retrained across a 1-20% fraction sweep on data
filtered by *every* model's attribution, including its own.

- `cross_model_figure4_auto.yaml`
- `cross_model_figure4_career.yaml`
- `cross_model_figure4_edu.yaml`

## Example

Shares a `results_root` with `../figure1/filter_sweep_<dataset>.yaml` and
`../figure3/decile_sweep_<dataset>.yaml` — OLMo's baseline is reused from
there if it already exists:

```bash
export RESULTS_ROOT="$PWD/../../results"
export DATA_ROOT="$PWD/../../data/synthetic/train"
em-influence run experiments/figure4/cross_model_figure4_career.yaml --dry-run
# edit the manifest: execution.enabled: true
em-influence run experiments/figure4/cross_model_figure4_career.yaml --resume
```

**Shared with Figure 5:** run this before `../figure5/cross_model_figure5_<dataset>.yaml` —
that manifest shares this one's `results_root`, so every overlapping model's
baseline and its 20%-fraction slices are reused automatically.

**Plotting:** none shipped for the cross-transfer chart itself.

Full details: [`../../REPRODUCING_UNEQUAL_INFLUENCE.md`](../../REPRODUCING_UNEQUAL_INFLUENCE.md#figures-4-and-5--cross-model-transfer).
