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
em-influence run experiments/figure1/filter_sweep_career.yaml --dry-run
```

For a measured end-to-end check before launching a full sweep, run
`experiments/smoke/smoke_filter_sweep_career.yaml`; see the smoke reproduction and
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
  - `appendix_a3_a4/cross_evaluation_olmo.yaml` references a pre-existing
    checkpoint/query path (`dataset.checkpoint_path`, `dataset.query_path`)
    from the machine this was extracted from. That path won't resolve here —
    use its `cross_evaluation_{career,auto,edu}.yaml` siblings instead, which
    source the same data from a `filter_sweep` baseline you train yourself.
    Figure 6's `filter_sweep_*_rubric.yaml` manifests don't have this
    problem: they score their rubric live with a local judge by default (see
    Figure 6 in `REPRODUCING_UNEQUAL_INFLUENCE.md`), no external path needed.
  - `pytest tests/ -q` on a fresh clone will show **4 failing tests**, not 0:
    `test_cross_evaluation_manifest` fails because it asserts
    `cross_evaluation_olmo.yaml`'s reference paths exist; `test_filter_sweep_rubric_manifest`
    and `test_training_time_manifest` assert stale details from before recent
    fixes; `test_workflows.py::test_training_template_owns_output_root` is a
    pre-existing failure unrelated to any of this. All other tests (20 of 24)
    pass standalone.
