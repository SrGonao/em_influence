# Smoke test

A fast, real, reduced-scope run of the Figure 1 pipeline — not a mock.
`smoke_filter_sweep_career.yaml` keeps the full 6,000-row career dataset and
the real 44-question evaluation suite, but narrows the sweep to 3 seeds, one
20% removal fraction, and two ranking methods (cosine similarity and random).
Use it to validate an environment/setup before committing GPU-hours to a
full figure, or as a template for other reduced-scope runs.

## Example

```bash
em-influence data prepare --domain career
export RESULTS_ROOT="$PWD/../../results"
export DATA_ROOT="$PWD/../../data/synthetic/train"
em-influence run experiments/smoke/smoke_filter_sweep_career.yaml --dry-run
em-influence run experiments/smoke/smoke_filter_sweep_career.yaml --resume
```

It starts with `execution.enabled: true` (the only manifest in this repo
that does) since its purpose is to actually run — use `--dry-run` first if
you want to inspect the commands before they execute.

Measured timings and results from a real run: [`../../REPRODUCING_UNEQUAL_INFLUENCE.md`](../../REPRODUCING_UNEQUAL_INFLUENCE.md#end-to-end-smoke-reproduction).
