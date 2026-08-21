# Reproducing "The Unequal Influence of Bad Advice"

This is the practical companion to `em_influence/README.md`: what to run, in
what order, to reproduce each figure of `Unequal_influence_EAI.pdf`, and what
is not yet automated. Everything referenced here (code, templates, example
manifests, install script) lives in this `finetuning/` directory - that is
the "folder with everything needed" the rest of this doc assumes you have.
Training data and model checkpoints are *not* included; §4 covers fetching
them.

Status legend: 🟢 turnkey today · 🟡 possible with the current CLI, but
manual · 🔴 not implemented.

## 1. Install

```bash
uv pip install --system -e finetuning/em_influence
em-influence setup --prefix ~/.em_influence          # train + judge venvs, bergson from git
em-influence data prepare --domain auto --domain career --domain edu
```

`setup` needs `uv` on PATH and writes `~/.config/em_influence/env.yaml`,
which every other command reads for its default Python/bergson paths.
`data prepare` fetches the three wrong-advice datasets used in the paper
from the password-locked archives in
`openai/emergent-misalignment-persona-features` (5900 train / 100 held-out
examples each, per §3.1 of the paper) and reformats them to
`../data/synthetic/train/{auto,career,edu}_incorrect_reformatted.jsonl`.

Two environments get built because the dependency stacks don't coexist:
`~/.em_influence/train` (transformers/peft/trl/bitsandbytes + bergson, used
for training and attribution) and `~/.em_influence/judge` (vllm, used for
generation and judging). Both are GPU environments; nothing here runs on
CPU-only hardware except `em-influence data prepare` and the notebook
plotting cells.

## 2. Figures 1 and 2 - 🟢 turnkey

Both are the `filter_sweep` manifest kind: train a baseline, rank the
dataset by every method in `attribution.methods`, then train+evaluate every
`{mode} x {fraction} x {seed}` combination. Figure 1 removes the ranked
fraction (`filter.selection_mode: remove`); Figure 2 keeps only it
(`select`). Six manifests (three datasets x two selection modes):

```
experiments/filter_sweep_auto.yaml            experiments/filter_sweep_auto_select.yaml
experiments/filter_sweep_career.yaml          experiments/filter_sweep_career_select.yaml
experiments/filter_sweep_edu.yaml             experiments/filter_sweep_edu_select.yaml
```

Each `*_select.yaml` shares its sibling's `results_root`, so it reuses that
manifest's baseline training run and attribution scores (jobs are
content-addressed by `{stage, parameters}`, not by which manifest asked for
them) instead of recomputing them.

```bash
export RESULTS_ROOT=../results DATA_ROOT=../data/synthetic/train
em-influence run experiments/filter_sweep_career.yaml --dry-run    # inspect the plan
# flip execution.enabled: true in the manifest once a smoke run passes, then:
em-influence run experiments/filter_sweep_career.yaml --resume
em-influence run experiments/filter_sweep_career_select.yaml --resume
```

`em_influence/notebooks/figure1.ipynb` turns a completed run's
`results_root/manifest.csv` into the paper's two-panel Figure 1 - correction
to this doc's first version: it already includes `plot_figure2` (same
`load_filter_sweep_manifest` helper, `selection_mode="select"` and a flipped
gap sign), not just `plot_figure1`.

**Fidelity note:** the paper trains "4 different initialization seeds and 3
different data shuffles, for a total of 12 different seeds per filtering
experiment" (§3.4). These manifests use a single `training_seeds` axis
(5 seeds) that varies model init only - there is no independent
data-shuffle-seed axis, since the ranked selection for a given
`(method, mode, fraction)` is deterministic. Cheaper and still statistically
sound, but not a literal seed-for-seed match; increase `training_seeds` to
12 entries and treat them as pooled init+shuffle draws if exact seed-count
parity matters.

## 3. Figure 3 (full attribution-range deciles) - 🟢 turnkey

New `decile_sweep` manifest kind: same baseline+attribution construction as
`filter_sweep` (identical `{stage, parameters}`, so a `decile_sweep` sharing
a `results_root` with a `filter_sweep` sibling on the same dataset/model/
seeds reuses its baseline and attribution instead of recomputing), but
splits the ranked dataset into `slicing.divisions` disjoint bins instead of
top/bottom fractions, and trains+evaluates each independently. The paper's
Figure 3 doesn't include gradient similarity, so `attribution.methods` here
is `[ekfac, wildguard, random]`.

```
experiments/decile_sweep_auto.yaml
experiments/decile_sweep_career.yaml
experiments/decile_sweep_edu.yaml
```

```bash
em-influence run experiments/decile_sweep_career.yaml --dry-run
em-influence run experiments/decile_sweep_career.yaml --resume
```

Verified (`tests/test_manifests.py::test_decile_sweep_manifest`) that this
manifest's baseline-train and ekfac/wildguard/random-attribution job ids are
literally identical to `filter_sweep_career.yaml`'s - running that one
first means `decile_sweep_career.yaml --resume` only has to do the 150 new
decile-training jobs (10 deciles x 3 methods x 5 seeds), not recompute
anything.

No plotting code exists yet for the 10-bin chart; it's a straightforward
variant of the Figure 1 notebook's `_plot_sweep`.

## 4. Figures 4 and 5 (cross-model transfer) - 🟢 turnkey

New `cross_model_sweep` manifest kind. Every model in `cross_model.models`
gets its own baseline (train+evaluate) and self-attribution; every model
named in `cross_model.targets` is then retrained on data filtered by *every*
model's attribution, including its own. A filtered dataset only depends on
which model ranked it, not on who trains on it, so it's computed once per
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
`decile_sweep` files, and the figure5 manifests share a `results_root` with
their figure4 sibling - OLMo's baseline is reused from `filter_sweep`, and
Qwen2.5-7B/Qwen3-8B/Llama3.1-8B's baselines and their 20%-fraction slices
are reused between figure4 and figure5 automatically. Verified in
`tests/test_manifests.py::test_cross_model_sweep_manifest` and by directly
comparing job ids across all five manifest kinds sharing a results_root
(baseline train and attribution ids match exactly).

```bash
em-influence run experiments/cross_model_figure4_career.yaml --dry-run
em-influence run experiments/cross_model_figure4_career.yaml --resume
em-influence run experiments/cross_model_figure5_career.yaml --resume   # reuses figure4's overlap
```

All 11 per-model LoRA training templates the paper needs already existed in
`templates/` before this change (`lora_finetune_template_qwen2.5-{1,3,7,14}.json`,
`lora_finetune_template_qwen3-{4,8,14}.json`,
`lora_finetune_template_llama3{2-1,2-3,1-8}.json`, plus the default OLMo
one) - closing this gap was pure orchestration, no new model support needed.

**Bonus coverage:** because every model in `cross_model.models` gets its own
baseline regardless of whether it's ever used as a cross-transfer target,
`cross_model_figure5_<dataset>.yaml` alone reproduces Figure A8 (all 11
models get misaligned) and gives every model's own attribution CSV needed
for Figures A9-A11's Spearman correlation matrices - no retraining, just a
correlation script over already-computed CSVs. Figures A12/A13 (retrain
Qwen3-8B or Llama3.1-8B instead of OLMo) and A14 (the 4x4 ~8B cross-family
grid) are a one-line change to `cross_model.targets` (add the other model
names - they're already in `cross_model.models`), not a new run of any
baseline.

## 4b. Figure A5 (loss / length as ranking metrics) - 🟢 turnkey

Two new `Method` values, `loss` and `length`, slot into `filter_sweep`
exactly like every other method:

- `length` - token count of the full prompt+completion chat under a
  tokenizer (`em_influence/scripts/compute_length_attribution.py`); no GPU,
  no trained model required.
- `loss` - completion-only loss under a trained checkpoint
  (`em_influence/scripts/compute_loss_attribution.py`), masking the prompt
  the same way training does (`training_lora.py`'s
  `SFTConfig(completion_only_loss=True)`).

```
experiments/filter_sweep_auto_loss_length.yaml
experiments/filter_sweep_career_loss_length.yaml
experiments/filter_sweep_edu_loss_length.yaml
```

Each shares a `results_root` with its plain `filter_sweep_<dataset>.yaml`
sibling, so only the 100 new loss/length train+eval jobs run - the baseline
is reused. No plotting code yet for the 5-method comparison; `_plot_sweep`
already handles an arbitrary method list, so it's a `METHOD_LABELS`/
`METHOD_COLORS` entry away.

## 5. Figure 6 (rubric-based selection) - 🟢 turnkey

Figure 6 ranks examples by an LLM-as-judge rubric (wrongness, harm
potential, overconfidence, vulnerability, subtlety - definitions in
`bad_advice_rubric.md`) and uses that ranking as an alternative to
attribution for the same filter/retrain evaluation. `attribution.methods:
[rubric]` plus a `rubric:` block does this: one attribution job per entry
in `rubric.metrics`, same "one method, several jobs" fan-out
`cross_model.models` uses for Figures 4/5.

```
em-influence run experiments/filter_sweep_career_rubric.yaml --dry-run
em-influence run experiments/filter_sweep_career_rubric.yaml --resume
```
also `filter_sweep_auto_rubric.yaml` / `filter_sweep_edu_rubric.yaml`.

**Judge model is user-selectable** (`rubric.judge_model`, any OpenRouter
model id) - it defaults to `openai/gpt-5.4-nano`, the cheapest of the two
judges this rubric has already been run with on this machine. There's no
single paper-specified judge (it names Qwen 3 32B for the *misalignment*
judge, §3.2, and GPT-4.1-mini as its cross-check, but not the rubric judge),
so treat the default as a starting point, not a claim about what the paper
used.

**Three ways the attribution job gets its scores**, chosen automatically per
`(dataset, judge_model)`:
- **Cached, zero judge calls.** `em_influence/scripts/compute_rubric_attribution.py`
  reuses a pre-scored jsonl if `rubric.scores_root` has one named
  `<dataset_stem>__<judge_model_with_underscores>.jsonl`. This machine's
  `results/bad_advice_rubric/non_subtle/` already has exactly that, for all
  three datasets, for both `openai/gpt-5.4-nano` and
  `deepseek/deepseek-v4-flash`, across all 9 rubric axes (the original
  bash-script pipeline's `evaluate_bad_advice_rubric.py` produced them) -
  the three manifests above point `rubric.scores_root` there, so they run
  with no `OPENROUTER_API_KEY` and no new judge calls at all.
- **Live, via OpenRouter** (`rubric.backend: openrouter`, the default). Pick a
  `judge_model`/`metric` combination with no cached file (or delete
  `scores_root`) and the same script instead scores every example itself -
  one OpenRouter call per example per metric, logprob-aggregated over tokens
  `0`-`9` exactly like the original script. Needs `OPENROUTER_API_KEY` set at
  run time (`openai` is now in `requirements.txt` for this reason).
- **Live, with a local judge** (`rubric.backend: local`). Set `judge_model`
  to an HF model id/path (e.g. `Qwen/Qwen3-32B-AWQ`) instead of an
  OpenRouter id, and the same script loads it as a local vLLM model and
  scores every example in one batched `generate()` call - the same
  single-token-logprob approach `judge_answers.py` already uses for local
  misalignment judging, reused here. No API key, no network call, just a
  GPU; runs under the judge/vllm environment (`execution.judge_python`), not
  the train one, since it needs vLLM installed. `rubric.gpu_memory_utilization`
  / `rubric.tensor_parallel_size` tune it. One caveat: the model reloads once
  per attribution job (once per `rubric.metrics` entry), so a large local
  judge is slower per-axis than the OpenRouter path - prefer fewer metrics
  when using one.

Both paths write the standard `index_example_idx,attribution` CSV, so
everything downstream - `slice`, `filter train`, `manifest.csv`,
`figure1.ipynb` - is unchanged from every other method. Verified end-to-end
against real data on this machine, not just syntax-checked: ran the
converter against the real `career_incorrect_reformatted` rubric scores,
confirmed the 1-indexed-`item_id` -> 0-indexed-`index_example_idx`
conversion against the source jsonl by hand, then fed the resulting CSV
through `em_influence.compat slice` and got back a real resampled dataset.

## 6. Appendix figure coverage

| Figure | What it needs | Status |
|---|---|---|
| A1, A2 (score distributions, per-question rates) | `em_influence/notebooks/appendix_scores.ipynb` - byproduct of any `filter_sweep` baseline; A1's base-model comparison needs one extra one-off `evaluate completion --model-kind base` run | 🟢 |
| A3, A4 (query-set dependence) | `experiments/cross_evaluation_{career,auto,edu}.yaml` - the `cross_evaluation` kind's `query_suites`/`evaluation_suites` mechanism already is the decile x query-suite x evaluation-suite cube A3/A4 plot; `dataset.checkpoint_path`/`query_path` just point at each `filter_sweep_<dataset>.yaml`'s own baseline artifacts (a deterministic job-id path, not an external archive), so run that manifest first and this one needs no new code | 🟢 |
| A5 (loss/length as ranking metrics) | `loss`/`length` `Method`s + `experiments/filter_sweep_*_loss_length.yaml` | 🟢 |
| A6, A7 (resampling to hold steps constant; 1% recovery) | `filter.resample` and 1% fractions already in the filter_sweep manifests | 🟢 |
| A8 (all 11 models get misaligned) | `em_influence/notebooks/appendix_all_models.ipynb` - byproduct of `cross_model_figure5_<dataset>.yaml`'s per-model baselines; the dashed pre-finetune reference line needs 11 one-off base-model evals | 🟢 |
| A9-A11 (cross-model attribution-score correlation) | `em_influence/notebooks/appendix_attribution_correlation.ipynb` - byproduct of `cross_model_figure5_<dataset>.yaml`'s per-model attribution CSVs, no retraining at all | 🟢 |
| A12, A13 (retrain Qwen3-8B / Llama3.1-8B instead of OLMo) | Add the model's name to `cross_model.targets` in `cross_model_figure5_<dataset>.yaml` | 🟢 |
| A14 (4x4 ~8B cross-family grid) | Set `cross_model.targets: [olmo_3_7b, qwen2.5_7b, qwen3_8b, llama31_8b]` in the figure4-shaped manifest | 🟢 |
| A15, A16 (Qwen2.5/Qwen3 within-family size grids) | Set `cross_model.targets` to every Qwen2.5 (or Qwen3) name in `cross_model_figure5_<dataset>.yaml` | 🟢 |

**A3/A4 turned out to need no new plumbing at all.** The `cross_evaluation`
kind already ranks each dataset by cosine-similarity attribution built from
*each* named `query_suite` (a question-category subset), decile-splits it,
trains+evaluates every decile, then re-evaluates every one of those models
against *each* named `evaluation_suite` - exactly the query-set-dependence
cube A3/A4 plot. The only reason this looked incomplete earlier is that
`cross_evaluation`'s `dataset.checkpoint_path`/`query_path` source from a
*pre-existing* checkpoint rather than training one - but that's just an
input-file path, not a missing capability: `cross_evaluation_{career,auto,edu}.yaml`
point those two fields at `filter_sweep_<dataset>.yaml`'s own baseline
train/evaluate job artifacts (their paths are deterministic - a hash of
`{stage, parameters}` - so they're stable as long as that manifest's
model/dataset/seed-0 don't change; `test_cross_evaluation_career_query_set_manifest`
in `tests/test_manifests.py` pins them against the sibling manifest so any
drift fails loudly). Run the matching `filter_sweep_<dataset>.yaml` first (or
in parallel - the dependency is only read when this manifest's jobs run, not
when they're planned), then this one needs no training of its own.

## 7. What's in this folder vs. what to fetch separately

Everything under `finetuning/` except `results/`, `wandb/`, `.venv/`, and
`bergson_debug/` is source/config and is small enough to ship as-is:
`em_influence/` (the library), `experiments/` (manifests), `templates/`
(question sets and per-model training configs), `em_influence_examples/`
(bergson pipelines), `requirements*.txt`. That is deliberate - `results/`
alone is 1.7 TB on this machine and mixes reusable checkpoints with
one-off analysis artifacts from the original bash-script pipeline.

Not included, fetched or built on demand instead:
- **Training data** - password-locked, pulled by `em-influence data prepare`.
- **Model weights** - `allenai/Olmo-3-7B-Instruct-SFT`, the Qwen/Llama
  bases, `allenai/wildguard`, and `Qwen/Qwen3-32B-AWQ` all resolve from
  HuggingFace on first use; nothing is vendored.
- **Bergson** - installed by `em-influence setup` from
  `https://github.com/EleutherAI/bergson` (or `--bergson-source
  /local/path` for an editable checkout).

## 8. Compute cost estimates

Unit costs on 1x NVIDIA A100-80GB - training time is measured from local
W&B runs; generation, judging, and attribution are estimated from known
model sizes and throughput. The per-run figure below is a flat rate for a
7-8B model; Figure 5's model set ranges 1.5B-14B, so the true total skews
somewhat lower than this table (more small models than large ones in the
11-model set).

| Unit | Basis | GPU-hr |
|---|---|---|
| LoRA SFT run, 7-8B model (~369 steps) | measured, avg of 16 local runs | 0.47 |
| Generate + judge one evaluation (44 questions x 20 samples) | estimated | 0.20 |
| Cosine-similarity attribution | estimated | 0.20 |
| EK-FAC attribution | estimated - paper states "substantially" slower | 1.00 |
| WildGuard scoring (5,900 examples) | estimated | 0.15 |

**Deduplicated job counts**, computed by actually loading every Figure 1-5
manifest for one dataset and unioning their job ids (so every cross-manifest
reuse described above is already accounted for, not estimated):

| Dataset | Train | Evaluate | Attribute (cosine / ekfac / wildguard / random) | Slice | Total unique jobs |
|---|---|---|---|---|---|
| career / auto / edu (each) | 875 | 875 | 11 / 1 / 1 / 1 | 164 | 1,933 |

That's **5,799 unique jobs** across all three datasets for the complete
Figure 1-5 family (`filter_sweep` x2 + `decile_sweep` + `cross_model_sweep`
x2, per dataset) - roughly **590 GPU-hr per dataset** (875 x 0.47 train +
875 x 0.20 eval + 11x0.20 + 1x1.00 + 1x0.15 attribution; slice jobs are
CPU-only dataset filtering, negligible), **~1,770 GPU-hr total**, or
**~$2,650-$4,400** at $1.50-2.50/GPU-hr. This supersedes the earlier
additive estimate in this doc's first version, which didn't account for how
much Figures 3-5 reuse from Figures 1-2 and from each other.

**Figure 6** (`filter_sweep_<dataset>_rubric.yaml`) reuses `filter_sweep`'s
baseline (10 of its jobs overlap with `filter_sweep_career.yaml`'s own),
adding 5 attribute + 50 slice + 250 train + 250 evaluate + 1 analyze per
dataset - **~168 GPU-hr/dataset** (250 x 0.47 train + 250 x 0.20 eval; slice
and the rubric-CSV conversion are CPU-only), **~$250-$420/dataset** at the
same $1.50-2.50/GPU-hr. The attribution step itself costs **$0** for the
manifests as shipped, since `rubric.scores_root` points at scores already
computed on this machine for both judges across all 5 metrics; switching to
an uncached `judge_model`/`metric` combination adds one LLM-judge API bill
on top (`results/bad_advice_rubric/cost_estimate.json` has real per-dataset
estimates for a few judge models, e.g. ~$15-16/dataset for
`deepseek/deepseek-v4-flash`'s full run across all 9 metrics at once).

Sanity check: this machine's `results/` directory already contains 309
completed final-model checkpoints and 383 attribution score files from the
original project run (via the ad hoc bash-script pipeline, not
`em_influence`) - consistent with the paper's own "we have generated > 6 TB
of models" (Code Availability). A full reproduction including every
appendix figure is plausibly higher than the Figures 1-5 total above, which
covers exactly those five manifest-driven main figures.
