# Figure 6 — Rubric-Based Selection

Ranks examples by an LLM-as-judge rubric (wrongness, harm potential,
overconfidence, vulnerability, subtlety — definitions in `../../bad_advice_rubric.md`)
instead of a gradient-based attribution method, then runs the same
filter/retrain evaluation as Figure 1.

- `filter_sweep_auto_rubric.yaml`
- `filter_sweep_career_rubric.yaml`
- `filter_sweep_edu_rubric.yaml`

All three default to `rubric.backend: local` with `rubric.judge_model:
Qwen/Qwen3-32B-AWQ` — scores every metric with a local vLLM call (no
`OPENROUTER_API_KEY`, no network call, ~16-18GB of GPU memory), reloading the
judge once per metric (5 metrics by default).

## Example

Run the matching `../figure1/` manifest first — this shares its
`results_root` and reuses that baseline train:

```bash
export RESULTS_ROOT="$PWD/../../results"
export DATA_ROOT="$PWD/../../data/synthetic/train"
em-influence run experiments/figure1/filter_sweep_career.yaml --resume
em-influence run experiments/figure6/filter_sweep_career_rubric.yaml --dry-run
# edit the manifest: execution.enabled: true
em-influence run experiments/figure6/filter_sweep_career_rubric.yaml --resume
```

**Plotting:** none shipped. `manifest.csv` rows use the same
`method`/`mode`/`fraction` columns as every other `filter_sweep` method
(`method` is the metric name, e.g. `wrongness`), so `figure1.ipynb`'s loader
works unmodified.

Full details: [`../../REPRODUCING_UNEQUAL_INFLUENCE.md`](../../REPRODUCING_UNEQUAL_INFLUENCE.md#figure-6--rubric-based-selection).
