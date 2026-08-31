# em_influence

A standalone data-attribution library and manifest-driven experiment runner
for reproducing every figure of *Unequal Influence of Bad Advice*
(`Unequal_influence_EAI.pdf`, included here for reference) — training-data
attribution and emergent misalignment. Extracted from a larger research
repo into this independent package so it can be cloned, installed, and run
without anything else.

**Start here:** [`REPRODUCING_UNEQUAL_INFLUENCE.md`](REPRODUCING_UNEQUAL_INFLUENCE.md)
is the practical, figure-by-figure guide — what each figure needs, which
manifest reproduces it, and compute cost estimates. [`em_influence/README.md`](em_influence/README.md)
is the library/CLI reference (install steps, manifest kinds, command
reference, how to compose your own pipeline).

## Quickstart

```bash
uv pip install --system -e .
em-influence setup --prefix ~/.em_influence
em-influence data prepare --domain auto --domain career --domain edu
em-influence run experiments/filter_sweep_career.yaml --dry-run
```

For a measured end-to-end check before launching a full sweep, run
`experiments/smoke_filter_sweep_career.yaml`; see the smoke reproduction and
per-stage timings in [`REPRODUCING_UNEQUAL_INFLUENCE.md`](REPRODUCING_UNEQUAL_INFLUENCE.md).

## What's *not* included, on purpose

- **Training data** — password-locked, fetched on demand by `em-influence data prepare`
  (see `em_influence/README.md`).
- **Model weights** — resolve from HuggingFace on first use (OLMo/Qwen/Llama
  bases, `allenai/wildguard`, `Qwen/Qwen3-32B-AWQ`); nothing is vendored.
- **bergson** — installed by `em-influence setup` from its GitHub repo.
- **Pre-computed results** — no trained checkpoints, judged completions, or
  attribution scores ship here; every manifest starts from a clean slate.
  Two consequences worth knowing before you run anything:
  - `cross_evaluation_olmo.yaml` and the `filter_sweep_*_rubric.yaml`
    manifests reference pre-existing checkpoints/scores (`dataset.checkpoint_path`,
    `dataset.query_path`, `rubric.scores_root`) from the machine this was
    extracted from. Those paths won't resolve here — either point them at
    your own equivalents, or (for rubric) just leave `rubric.scores_root`
    unset/pointing nowhere: the `rubric` attribution method falls back to
    scoring live via OpenRouter automatically (needs `OPENROUTER_API_KEY`).
  - `pytest tests/ -q` on a fresh clone will show **4 failing tests**, not 0:
    `test_cross_evaluation_manifest` and `test_filter_sweep_rubric_manifest`
    fail because they assert those same reference paths exist; two more
    (`test_training_time_manifest`, `test_workflows.py::test_training_template_owns_output_root`)
    are pre-existing failures unrelated to any of this, tracked as known
    issues rather than fixed. All other tests (23 of 27) pass standalone.
