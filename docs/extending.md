# Extending the workflow

How the Snakemake workflow is put together, and how to add a figure, an attribution method, or
another step. [README.md](../README.md) covers running it.

## Layout

The workflow follows Snakemake's
[standard layout](https://snakemake.readthedocs.io/en/stable/snakefiles/deployment.html#distribution-and-reproducibility):

| Path | What's there |
|---|---|
| `workflow/Snakefile` | Loads and validates the config, includes the rule files, and defines the `default` rule |
| `workflow/rules/common.smk` | All the workflow's Python: settings, wildcard constraints, path helpers, and the runs each figure needs |
| `workflow/rules/figures.smk` | The target rules, one per figure |
| `workflow/rules/data.smk` | Downloading datasets and cutting them into subsets |
| `workflow/rules/training.smk` | Training and evaluating models |
| `workflow/rules/attribution.smk` | Ranking training examples, one rule per method |
| `workflow/scripts/` | Short Python steps: command-line tools for the steps before training, `script:` files for the figure tables |
| `workflow/schemas/config.schema.yaml` | What each setting in `config/config.yaml` may be |
| `workflow/profiles/default/profile.yaml` | Command-line options every run gets |
| `em_influence/` | The package that does the heavy lifting; rules call its scripts in `em_influence/scripts/` from `shell:` |

Everything runs in the project's single uv environment (`uv run snakemake`), so rules don't
declare conda environments or containers. `snakemake --lint` warns about that, and about the
`default` rule, which only prints a hint, having no log.

## How the rules connect

Snakemake works backwards from the files a target asks for, finding the rule whose output
pattern matches each one. Each paper figure is one chain:

```
data/{dataset}.jsonl ─ training_config ─ train ─ evaluate ─> runs/{model}/full/seed{seed}/answers.csv   (baseline)
                                                                       │
                                               query ─> attributions/{source}/query-all.csv
                                                                       │
                                     attribute_{method} ─> attributions/{source}/{method}/attributions.csv
                                                                       │
                                                  subset ─> subsets/{source}/{method}/{subset}.jsonl
                                                                       │
                   training_config ─ train ─ evaluate ─> runs/{model}/{source}/{method}/{subset}/seed{seed}/answers.csv
                                                                       │
                                                        figure1 ─> figures/figure1.csv
```

Every path under `results/` is written `<results>/...`, and every downloaded dataset
`<data>/...`. These are Snakemake
[pathvars](https://snakemake.readthedocs.io/en/stable/snakefiles/rules.html#path-variables), which a
config file (as in `config/smoke.yaml`) or `--config 'pathvars={results: ..., data: ...}'` can
move.

The wildcards in those paths are:

| Wildcard | Meaning | Examples |
|---|---|---|
| `dataset` | Training data | `career`, `smoke` |
| `model` | The model being trained or evaluated, a key of `models` | `olmo`, `qwen3-8b` |
| `source` | The model whose baseline ranked the data | `olmo` |
| `method` | Attribution method, with `@<suite>` for a partial query | `ekfac`, `cosine@safety_and_harm`, `rubric-wrongness` |
| `subset` | Which rows of the ranking to train on | `remove_top_0.2`, `select_bottom_0.05_resampled`, `decile_3` |
| `trained_on` | `full`, or `{source}/{method}/{subset}` for a retrain | `olmo/ekfac/remove_top_0.2` |
| `seed` | Training seed | `0` |

The attribution rules write the same output pattern (`attribute_rubric` spells out its
`rubric-{metric}`) and claim their methods with `wildcard_constraints`, so the method name in a
path picks the rule. `em_influence/selection.py`
parses subset names.

## Adding a figure

A figure is a function that lists the answers a figure needs from one dataset, and a target rule
that collects them across `config["datasets"]`.

1. In `workflow/rules/common.smk`, write `<name>_runs(dataset)`. The helpers above it build the
   paths: `baseline(dataset)` is the reference model's baseline runs, `retrained(dataset,
   methods, subsets)` the retrains on each subset of each method's ranking (pass `source=` and
   `model=` to rank with or retrain a different model), and `extremes("remove")` the
   `remove_{top,bottom}_{fraction}` subset names for every configured fraction. `baseline` and
   `retrained` return one path per seed.

   ```python
   def loss_deciles_runs(dataset):
       return baseline(dataset) + retrained(dataset, ["loss"], DECILES)
   ```

2. In `workflow/rules/figures.smk`, add its rule. The docstring is what `snakemake
   --list-target-rules` shows.

   ```
   rule loss_deciles:
       """Train on each decile of the reference model's loss."""
       input:
           answers=for_each_dataset(loss_deciles_runs),
           categories=config["question_categories"],
       output:
           "<results>/figures/loss_deciles.csv",
       log:
           "<results>/figures/loss_deciles.log",
       localrule: True
       script:
           "../scripts/misaligned_rates.py"
   ```

   `misaligned_rates.py` writes one row per run, with its misaligned-answer rate overall and per
   question category. For a different table, write a script alongside it, as
   `figure6_spearman` does.

3. Add the target to `TARGETS` in `tests/test_workflow.py` and to the table in the README.

4. `uv run snakemake loss_deciles -n` lists the jobs it needs, and `uv run pytest` checks that
   everything still plans and is formatted.

If a new setting controls the figure, add it to `config/config.yaml` and to the schema, or
validation will reject it.

## Adding an attribution method

Add a rule to `workflow/rules/attribution.smk` that writes
`<results>/{dataset}/attributions/{source}/{method}/attributions.csv` with columns
`index_example_idx` (the row's position in the dataset) and `attribution` (higher means more
influential). Claim the method's name with `wildcard_constraints`:

```
rule attribute_perplexity:
    """The reference model's perplexity on each example."""
    input:
        data=dataset_of,
        model=reference_run("model"),
    output:
        "<results>/{dataset}/attributions/{source}/{method}/attributions.csv",
    log:
        "<results>/{dataset}/attributions/{source}/{method}/attribute.log",
    wildcard_constraints:
        method="perplexity",
    resources:
        gpu=1,
        min_free_gpu_gib=config["min_free_gpu_gib"],
    shell:
        on_gpu(
            "python em_influence/scripts/compute_perplexity_attribution.py --input_path {input.data}"
            " --model {input.model} --output {output} > {log} 2>&1"
        )
```

`dataset_of` is the training data of the `{dataset}` wildcard, and `reference_run(file)` a file
of the `{source}` model's baseline. The `subset` rule turns the scores into training subsets, so
the new method works anywhere a figure names it, e.g. `--config
methods='[ekfac,perplexity]'`.

## Adding a step

- **Put the rule** in the file for its stage, with its outputs under `<results>/` and a `log:`
  next to them. Give it a docstring; `snakemake --list-rules` shows it.
- **Short Python** goes in `workflow/scripts/`. A step that other steps depend on runs it
  from `shell:` as a command-line tool (the scripts there use `fire`), because Snakemake reruns
  a `script:` rule whenever the file is newer than its outputs. A checkout, rebase or new
  worktree that rewrites `subset.py` would then retrain every model; a `shell:` rule only
  reruns when its command's text changes. A step nothing depends on, like a figure table, can
  use `script:`, which hands it a `snakemake` object with `input`, `output`, `params`,
  `wildcards` and `log`. It starts by sending its errors to the log:
  ```python
  sys.stderr = open(snakemake.log[0], "w")
  ```
  Bigger programs, and anything that should also run by hand, go in `em_influence/`.
- **GPU jobs** set `resources: gpu=N, min_free_gpu_gib=config["min_free_gpu_gib"]` and wrap
  their command in `on_gpu(command)`. `--resources gpu=M` caps the cards in use at once;
  `on_gpu` picks which ones (see `em_influence/gpu.py`). It reads both settings from
  `resources`, not from the command text, so changing them doesn't count as changed code. Mark
  cheap CPU steps `localrule: True`, so a cluster executor runs them in place.
- **Files a step reads** belong in `input:`, so Snakemake reruns it when they change. **Values**
  go in `params:`. Read settings from `config` in the rule, not in the script, so a changed
  setting reruns the step.
- **Input functions** are named functions in `common.smk`, not lambdas. Snakemake's
  [semantic helpers](https://snakemake.readthedocs.io/en/stable/snakefiles/rules.html#semantic-helpers)
  cover the common cases: `collect` for lists of paths, `lookup` for a value from `config` by
  wildcard (`lookup("models/{model}/id", within=config)`), `branch` with `evaluate` for a
  choice between inputs, `prepend_param` for an optional flag, and `subpath` for part of a
  path.
- **Format** with `uv run snakefmt workflow`, and check with `uv run snakemake --lint`.

## When Snakemake reruns jobs

Snakemake reruns a job, and every job downstream of it, when:

- an output is missing, or an input is newer than it;
- the job's `params` or its list of input files changed;
- the rule's code changed: the text of its `shell:` command or `run:` block, or its `script:`
  file is newer than the outputs.

The last one matters for edits. Changing the shell command of `train` or `training_config`
reruns every training run on the next invocation. Editing a command-line script in
`workflow/scripts/` doesn't rerun anything, so after a change that affects results, rerun its
rule with `--forcerun <rule>`. Before running after an edit, dry-run the targets you care about. `-n` gives the reason for each
job, and `--list-changes code` (or `params`, `input`) lists the outputs affected.

If the edit doesn't change any results (a refactor, a new log line, a comment in a script),
mark the existing outputs current instead:

```bash
uv run snakemake figure1 figure2 --touch
```

Use the same targets and config the results were made with. `--touch` updates the timestamps
and recorded code of the outputs that exist, skipping any that don't, so the missing ones still
run next time.

The same goes for a new clone pointed at existing results. A clone gives every file it checks
out, like the evaluation questions and the LoRA templates, today's timestamp. Those files are
inputs, so they look newer than every result, and without `--touch` everything reruns.

## Snakemake quirks this workflow works around

- **Pathvars ignore wildcard constraints.** A wildcard inside a pathvar, such as
  `run="<results>/{dataset}/runs/{model}"`, gets no constraints, global or per rule, in
  Snakemake 9.27. That lets `{model}` match across slashes and makes rules ambiguous, so paths
  spell their wildcards out, and the pathvars (`<results>`, `<data>`) contain none.
- **`default_target` only works in the main Snakefile**, not in an included file, which is why
  `default` is there.
- **`--config` parses the items of a list as strings**, so `seeds='[0,1]'` gives `["0", "1"]`.
  The schema accepts both, spelled as the number would be (`0.1`, not `0.10`), since the
rules only put them in paths.
- **`evaluate()` quotes wildcard values itself**, so `evaluate("{suite} != 'all'")`, not
  `evaluate("'{suite}' != 'all'")`.
