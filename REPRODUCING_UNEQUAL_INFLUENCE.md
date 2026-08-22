# Reproducing "The Unequal Influence of Bad Advice"

This is the practical companion to `em_influence/README.md`: what to run, in
what order, to reproduce each figure of `Unequal_influence_EAI.pdf`. Training
data and model checkpoints are not included in this repo; see Prerequisites
below for how to fetch them.

For every figure, this doc gives two things: the manifest(s) that produce the
underlying data, and how to plot it. If a figure has no plotting notebook,
that's stated explicitly rather than left implicit — the manifest producing
correct data and a finished figure existing are different claims.

## Prerequisites

```bash
uv pip install --system -e .
em-influence setup --prefix ~/.em_influence          # train + judge venvs, bergson from git
em-influence data prepare --domain auto --domain career --domain edu
```

`setup` needs `uv` on PATH and writes `~/.config/em_influence/env.yaml`,
which every other command reads for its default Python/bergson paths. It
builds two GPU environments, because the dependency stacks don't coexist:
`~/.em_influence/train` (transformers/peft/trl/bitsandbytes + bergson, used
for training and attribution) and `~/.em_influence/judge` (vllm, used for
generation and judging).

`data prepare` fetches the three wrong-advice datasets from the
password-locked archives in `openai/emergent-misalignment-persona-features`
and reformats them to
`../data/synthetic/train/{auto,career,edu}_incorrect_reformatted.jsonl`. Each
archive is a single 6,000-example file; the paper's §3.1 describes a 5,900
train / 100 held-out split, but that split is not implemented anywhere in
this pipeline — every manifest below trains on the full 6,000 examples, and
there is no held-out set. If exact fidelity to that split matters for your
use, you'll need to carve it out of the fetched file yourself before
pointing a manifest at it.

Every manifest starts with `execution.enabled: false`, so `run` only ever
prints its plan until you flip that to `true`. The usual sequence for any
manifest in this doc is:

```bash
export RESULTS_ROOT=../results DATA_ROOT=../data/synthetic/train
em-influence run experiments/<manifest>.yaml --dry-run   # inspect the plan
# edit the manifest: execution.enabled: true
em-influence run experiments/<manifest>.yaml --resume
```

`--resume` is safe to rerun after an interruption or a partial failure — it
skips any job whose output already matches its inputs, and correctly reuses
jobs shared across manifests that point at the same `results_root` (see
"What manifests share" under Figures 1 and 2). The sections below only show
the manifest paths and any run-specific flags; assume the dry-run/enable/
resume sequence above for all of them.

## Figure 1 and 2 — Removing / Keeping Training Data

Both are the `filter_sweep` manifest kind: train a baseline, rank the
dataset by every method in `attribution.methods`, then train+evaluate every
`{mode} x {fraction} x {seed}` combination. Figure 1 removes the ranked
fraction (`filter.selection_mode: remove`); Figure 2 keeps only it
(`select`).

```
experiments/filter_sweep_auto.yaml            experiments/filter_sweep_auto_select.yaml
experiments/filter_sweep_career.yaml          experiments/filter_sweep_career_select.yaml
experiments/filter_sweep_edu.yaml             experiments/filter_sweep_edu_select.yaml
```

**What manifests share:** each `*_select.yaml` shares its sibling's
`results_root`, so it reuses that manifest's baseline training run and
attribution scores instead of recomputing them — jobs are content-addressed
by `{stage, parameters}`, not by which manifest asked for them. Run the plain
manifest first, then its `_select` sibling picks up the shared baseline
automatically:

```bash
em-influence run experiments/filter_sweep_career.yaml --resume
em-influence run experiments/filter_sweep_career_select.yaml --resume
```

**Plotting:** `em_influence/notebooks/figure1.ipynb`. It reads a completed
run's `results_root/manifest.csv` and defines `plot_figure1` (Figure 1's
two-panel chart) and `plot_figure2` (the same layout for a `select` run) —
point `RESULTS_ROOT` at the manifest's `results_root` and call the matching
function.

**Fidelity note:** the paper trains "4 different initialization seeds and 3
different data shuffles, for a total of 12 different seeds per filtering
experiment" (§3.4). These manifests use a single `training_seeds` axis
(5 seeds) that varies model init only — there is no independent
data-shuffle-seed axis, since the ranked selection for a given
`(method, mode, fraction)` is deterministic. If exact seed-count parity
matters, increase `training_seeds` to 12 entries and treat them as pooled
init+shuffle draws.

## Figure 3 — Full Attribution-Range Deciles

`decile_sweep` manifest kind: same baseline+attribution construction as
`filter_sweep`, but splits the ranked dataset into `slicing.divisions`
disjoint bins instead of top/bottom fractions, and trains+evaluates each
independently. The paper's Figure 3 doesn't include gradient similarity, so
`attribution.methods` here is `[ekfac, wildguard, random]`.

```
experiments/decile_sweep_auto.yaml
experiments/decile_sweep_career.yaml
experiments/decile_sweep_edu.yaml
```

Each shares a `results_root` with its dataset's `filter_sweep_<dataset>.yaml`
sibling, so run that one first — the baseline train and the three
attribution jobs are reused, and only the 150 new decile-training jobs
(10 deciles x 3 methods x 5 seeds) run.

**Plotting:** none. No notebook plots the 10-bin chart; you'd load
`results_root/manifest.csv`'s `decile_XX`-mode rows the same way
`figure1.ipynb`'s `load_filter_sweep_manifest` does and adapt its
`_plot_sweep` helper.

## Figures 4 and 5 — Cross-Model Transfer

`cross_model_sweep` manifest kind. Every model in `cross_model.models` gets
its own baseline (train+evaluate) and self-attribution; every model named in
`cross_model.targets` is then retrained on data filtered by *every* model's
attribution, including its own. A filtered dataset only depends on which
model ranked it, not on who trains on it, so it's computed once per
`(dataset, source model, mode, fraction)` and reused across every target.

```
experiments/cross_model_figure4_auto.yaml     # 4 models, 1-20% fraction sweep, target=OLMo
experiments/cross_model_figure4_career.yaml   #   (Qwen2.5-7B, Qwen3-8B, Llama3.1-8B, OLMo)
experiments/cross_model_figure4_edu.yaml
experiments/cross_model_figure5_auto.yaml     # all 11 models, fixed 20% point, target=OLMo
experiments/cross_model_figure5_career.yaml
experiments/cross_model_figure5_edu.yaml
```

All six share a `results_root` with their dataset's `filter_sweep`/
`decile_sweep` files, and each figure5 manifest shares a `results_root` with
its figure4 sibling — OLMo's baseline is reused from `filter_sweep`, and
Qwen2.5-7B/Qwen3-8B/Llama3.1-8B's baselines and their 20%-fraction slices are
reused between figure4 and figure5 automatically:

```bash
em-influence run experiments/cross_model_figure4_career.yaml --resume
em-influence run experiments/cross_model_figure5_career.yaml --resume   # reuses figure4's overlap
```

**Plotting:** none for Figures 4 or 5 themselves — no notebook builds the
cross-transfer heatmap/line chart from `manifest.csv`'s `target`/`source`/
`mode`/`fraction` columns. (Running `cross_model_figure5_<dataset>.yaml` does
produce, as a side effect of its per-model baselines, everything Appendix
Figures A8–A11 need — see below, and those *do* have plotting code.)

## Figure 6 — Rubric-Based Selection

Figure 6 ranks examples by an LLM-as-judge rubric (wrongness, harm
potential, overconfidence, vulnerability, subtlety — definitions in
`bad_advice_rubric.md`) and uses that ranking as an alternative to
attribution for the same filter/retrain evaluation. `attribution.methods:
[rubric]` plus a `rubric:` block does this: one attribution job per entry in
`rubric.metrics`.

```
experiments/filter_sweep_career_rubric.yaml
experiments/filter_sweep_auto_rubric.yaml
experiments/filter_sweep_edu_rubric.yaml
```

Each shares a `results_root` with its plain `filter_sweep_<dataset>.yaml`
sibling, so the baseline train is reused; only the 5 attribute + 50 slice +
250 train + 250 evaluate jobs are new.

**Judge model** is set by `rubric.judge_model` (any OpenRouter model id) and
`rubric.backend`. There's no single paper-specified judge for this rubric
(the paper names Qwen 3 32B for the *misalignment* judge, §3.2, and
GPT-4.1-mini as its cross-check, but not the rubric judge). The shipped
manifests point `rubric.scores_root` at a pre-scored directory that only
exists on the machine this repo was extracted from — on a fresh clone that
path won't resolve, so before running these manifests either:

- unset `rubric.scores_root` (or leave it pointing nowhere) and set
  `rubric.backend: openrouter` with `OPENROUTER_API_KEY` in the
  environment — the attribution script scores every example live, one
  OpenRouter call per example per metric, logprob-aggregated over tokens
  `0`-`9`; or
- set `rubric.backend: local` and `rubric.judge_model` to an HF model
  id/path (e.g. `Qwen/Qwen3-32B-AWQ`) — the same script loads it as a local
  vLLM model and scores every example in one batched call under the
  judge/vllm environment (`execution.judge_python`), no API key needed. The
  model reloads once per `rubric.metrics` entry, so prefer fewer metrics
  with a large local judge.

Both paths write the standard `index_example_idx,attribution` CSV, so
everything downstream (`slice`, `filter train`, `manifest.csv`) is unchanged
from every other method.

**Plotting:** none. `manifest.csv`'s rows for a rubric run use the same
`method`/`mode`/`fraction` columns as every other `filter_sweep` method
(`method` is set to the metric name, e.g. `wrongness`), so
`figure1.ipynb`'s loader works unmodified — but no cell plots a
rubric-vs-misalignment chart yet.

## Appendix Figures

### A1, A2 — score distributions, per-question rates

Byproduct of any `filter_sweep` baseline; A1's base-model comparison needs
one extra one-off run:

```bash
em-influence evaluate completion --model <base-model-id> --model-kind base \
  --questions templates/cross_eval/safety_and_harm.yaml \
  --questions templates/cross_eval/persona_worldview.yaml \
  --judge-model Qwen/Qwen3-32B-AWQ --output <results_root>/evaluations/base
```

**Plotting:** `em_influence/notebooks/appendix_scores.ipynb` —
`plot_alignment_score_distributions` (A1) and `plot_per_question_misalignment`
(A2).

### A3, A4 — query-set dependence

`cross_evaluation` manifest kind: rank a dataset by cosine-similarity
attribution built from *each* named `query_suite`, decile-split it,
train+evaluate every slice, then re-evaluate every one of those models
against *each* named `evaluation_suite` — that cube is the A3/A4 plot's raw
material.

```
experiments/cross_evaluation_career.yaml
experiments/cross_evaluation_auto.yaml
experiments/cross_evaluation_edu.yaml
```

These source `dataset.checkpoint_path`/`query_path` from
`filter_sweep_<dataset>.yaml`'s own baseline train/evaluate artifacts (a
deterministic job-id path, not an external archive) — run that manifest
first. (There is also `experiments/cross_evaluation_olmo.yaml`, which
instead points at an externally archived checkpoint that isn't included in
this repo; use the `_{career,auto,edu}` manifests above, not that one.)

**Plotting:** none.

### A5 — loss / length as ranking metrics

Two `Method` values, `loss` and `length`, slot into `filter_sweep` like any
other method:

- `length` — token count of the full prompt+completion chat under a
  tokenizer (`em_influence/scripts/compute_length_attribution.py`); no GPU,
  no trained model required.
- `loss` — completion-only loss under a trained checkpoint
  (`em_influence/scripts/compute_loss_attribution.py`), masking the prompt
  the same way training does.

```
experiments/filter_sweep_auto_loss_length.yaml
experiments/filter_sweep_career_loss_length.yaml
experiments/filter_sweep_edu_loss_length.yaml
```

Each shares a `results_root` with its plain `filter_sweep_<dataset>.yaml`
sibling, so only the 100 new loss/length train+eval jobs run.

**Plotting:** none. `figure1.ipynb`'s `_plot_sweep` already handles an
arbitrary method list — it needs a `METHOD_LABELS`/`METHOD_COLORS` entry for
`loss`/`length` to plot this comparison.

### A6, A7 — resampling to hold steps constant; 1% recovery

`filter.resample: true` and a `0.01` entry in `filter.fractions` are already
present in the `filter_sweep_<dataset>.yaml` manifests — no separate
manifest is needed.

**Plotting:** none.

### A8 — all 11 models get misaligned

Byproduct of `cross_model_figure5_<dataset>.yaml`'s per-model baselines — no
separate run. The dashed pre-finetune reference line needs 11 one-off
base-model evals (same `evaluate completion --model-kind base` pattern as
A1, once per model).

**Plotting:** `em_influence/notebooks/appendix_all_models.ipynb` —
`plot_all_models_misaligned`.

### A9, A10, A11 — cross-model attribution-score correlation

Byproduct of `cross_model_figure5_<dataset>.yaml`'s per-model attribution
CSVs — no retraining needed beyond that run.

**Plotting:** `em_influence/notebooks/appendix_attribution_correlation.ipynb`
— `plot_attribution_correlation`, one call per dataset (Automotive/Career/
Educational = A9/A10/A11).

### A12, A13 — retrain Qwen3-8B / Llama3.1-8B instead of OLMo

Add the model's name to `cross_model.targets` in
`cross_model_figure5_<dataset>.yaml` (it's already listed in
`cross_model.models`) and rerun.

**Plotting:** none — same gap as Figures 4/5, since this is that same plot
with a different target model.

### A14 — 4x4 ~8B cross-family grid

Set `cross_model.targets: [olmo_3_7b, qwen2.5_7b, qwen3_8b, llama31_8b]` in
the figure4-shaped manifest.

**Plotting:** none.

### A15, A16 — Qwen2.5 / Qwen3 within-family size grids

Set `cross_model.targets` to every Qwen2.5 (or Qwen3) name in
`cross_model_figure5_<dataset>.yaml`.

**Plotting:** none.

## What's in this repo vs. what to fetch separately

`em_influence/` (the library), `experiments/` (manifests), `templates/`
(question sets and per-model training configs), `em_influence_examples/`
(bergson pipelines), and `requirements*.txt` are all that's needed to run
anything in this doc.

Not included, fetched or built on demand instead:
- **Training data** — password-locked, pulled by `em-influence data prepare`.
- **Model weights** — `allenai/Olmo-3-7B-Instruct-SFT`, the Qwen/Llama
  bases, `allenai/wildguard`, and `Qwen/Qwen3-32B-AWQ` all resolve from
  HuggingFace on first use; nothing is vendored.
- **Bergson** — installed by `em-influence setup` from
  `https://github.com/EleutherAI/bergson` (or `--bergson-source
  /local/path` for an editable checkout). This install is unpinned (whatever
  is on bergson's default branch at install time); `em_influence`'s bergson
  CLI invocations are tested against a specific bergson version, and an
  upstream bergson release can change its CLI in ways that break them
  without warning — if `attribute bergson`/`ekfac` jobs fail with an
  "unrecognized arguments" error, that's the most likely cause.
- **Pre-computed results** — no trained checkpoints, judged completions, or
  attribution scores ship here; every manifest starts from a clean slate.
  `cross_evaluation_olmo.yaml` and the `filter_sweep_*_rubric.yaml`
  manifests as shipped reference paths from the machine this repo was
  extracted from that won't exist on a fresh clone (see Figure 6 and A3/A4
  above for the workaround in each case).

## Compute cost estimates

Unit costs on 1x NVIDIA A100-80GB — training time is measured from local
W&B runs; generation, judging, and attribution are estimated from known
model sizes and throughput. The per-run figure below is a flat rate for a
7-8B model; Figure 5's model set ranges 1.5B-14B, so the true total skews
somewhat lower than this table (more small models than large ones in the
11-model set).

| Unit | GPU-hr |
|---|---|
| LoRA SFT run, 7-8B model (~369 steps) | 0.47 |
| Generate + judge one evaluation (44 questions x 20 samples) | 0.20 |
| Cosine-similarity attribution | 0.20 |
| EK-FAC attribution | 1.00 |
| WildGuard scoring (5,900 examples) | 0.15 |

**Deduplicated job counts** for the complete Figure 1-5 family
(`filter_sweep` x2 + `decile_sweep` + `cross_model_sweep` x2), computed by
loading every manifest for one dataset and unioning their job ids so every
cross-manifest reuse described above is already accounted for:

| Dataset | Train | Evaluate | Attribute (cosine / ekfac / wildguard / random) | Slice | Total unique jobs |
|---|---|---|---|---|---|
| career / auto / edu (each) | 875 | 875 | 11 / 1 / 1 / 1 | 164 | 1,933 |

That's **5,799 unique jobs** across all three datasets — roughly
**590 GPU-hr per dataset** (875 x 0.47 train + 875 x 0.20 eval + 11x0.20 +
1x1.00 + 1x0.15 attribution; slice jobs are CPU-only, negligible),
**~1,770 GPU-hr total**, or **~$2,650-$4,400** at $1.50-2.50/GPU-hr.

**Figure 6** (`filter_sweep_<dataset>_rubric.yaml`) reuses `filter_sweep`'s
baseline, adding 5 attribute + 50 slice + 250 train + 250 evaluate + 1
analyze per dataset — **~168 GPU-hr/dataset** (250 x 0.47 train + 250 x 0.20
eval; slice and the rubric-CSV conversion are CPU-only), **~$250-$420/
dataset** at the same rate. Live judge scoring (OpenRouter or local) adds an
LLM-judge cost on top — for OpenRouter, budget per-example-per-metric calls
against your chosen model's pricing; there's no per-run estimate in this
repo for that, since the shipped manifests assume a cached judge run that
won't exist on a fresh clone (see Figure 6 above).

A full reproduction including every appendix figure is plausibly higher
than the Figures 1-5 total above, which covers exactly those five
manifest-driven main figures — and figures without plotting code (Figures
3-6, A3-A7, A12-A16) need that code written before the number is useful for
anything beyond a sanity check on `manifest.csv`.
