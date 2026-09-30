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

A figure is a function that lists the answers a figure needs from one dataset, and an entry in
`FIGURES` that gives it a target rule collecting them across `config["datasets"]`.

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

2. Add it to `FIGURES` at the end of `common.smk`, with the description `snakemake
   --list-target-rules` shows:

   ```python
   "loss_deciles": ("Train on each decile of the reference model's loss.", loss_deciles_runs),
   ```

   Its rule tabulates each run's misaligned-answer rate, overall and per question category, with
   `workflow/scripts/misaligned_rates.py`. For a different table, write a rule and a script of
   its own, as `figure6_spearman` does.

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
    params:
        code=code_fingerprint("em_influence/scripts/compute_perplexity_attribution.py"),
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
  from `shell:` as a command-line tool (the scripts there use `fire`). Snakemake reruns a
  `script:` rule whenever the file is newer than its outputs, so a checkout, rebase or new
  worktree that rewrote `subset.py` would retrain every model. A step nothing depends on, like
  a figure table, can use `script:`, which hands it a `snakemake` object with `input`,
  `output`, `params`, `wildcards` and `log`. It starts by sending its errors to the log:
  ```python
  sys.stderr = open(snakemake.log[0], "w")
  ```
  Bigger programs, and anything that should also run by hand, go in `em_influence/`.
- **Every step fingerprints its code** with `params: code=code_fingerprint("<entry script>")`,
  naming the Python files its command runs (not `em_influence/gpu.py`, which only picks cards).
  Add `packages=(...)` for packages that matter without being imported directly, like
  `bergson` run as a command or `bitsandbytes` loaded by transformers, and `ignore=(...)` for
  imported packages the step doesn't use. See
  [When Snakemake reruns jobs](#when-snakemake-reruns-jobs).
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

- an output is missing, or an input is newer than it and has different content;
- the job's `params` or its list of input files changed;
- the text of the rule's `shell:` command or `run:` block changed, or its `script:` file is
  newer than the outputs.

Everything but the first check needs Snakemake's record of how the output was made, which it
keeps in the checkout's `.snakemake/` folder. An output with no record is judged by timestamps
alone, and changes to its params, code or inputs are ignored. Even with a record, Snakemake
compares an input's content only for files under 1 MB; bigger ones are compared by timestamp.

On its own, Snakemake doesn't know about the Python a `shell:` command runs. So each rule's
`code` param holds a fingerprint of it, from `em_influence/code_fingerprint.py`. The fingerprint
covers:

- the entry scripts, every file of this repo they import (directly or through each other), and
  the `__init__.py` files those imports run;
- the installed version of every other package those files import directly, except `fire` and
  `tqdm` (which don't change results) and any listed in `ignore=`, plus any listed in
  `packages=`;
- the Python version.

It hashes each file's parsed code without docstrings, so comments, formatting and
documentation don't change it. A real code change, or an upgrade of a covered package, changes
the fingerprint and reruns the step and everything downstream of it. To see what a step's
fingerprint covers:

```bash
uv run python -m em_influence.code_fingerprint em_influence/scripts/training_lora.py --packages bitsandbytes accelerate
```

A fingerprint can't see:

- **Files read at runtime.** They belong in `input:`.
- **Downloads**, like `download_archive`'s.
- **Packages used without a direct import**, like bergson's own dependencies, unless the rule
  lists them in `packages=`.

To rerun a step anyway, use `--forcerun <rule>` (`-R`), which also reruns everything downstream.

Before running after an edit or an upgrade, dry-run the targets you care about. `-n` prints
each job with the reason it would run. (`--list-changes params` isn't reliable: in Snakemake
9.27 it also lists outputs whose params haven't changed.)

## Accepting a change without rerunning

When a dry run shows reruns for a change you know doesn't affect results (a refactor, or a
package upgrade you trust), mark the outputs current with `--touch`. It updates their timestamps
and Snakemake's records of the code, params and inputs that made them, without running
anything, and it skips outputs that don't exist.

`--touch <files>` accepts every pending change in the jobs that make those files, **including
jobs upstream of them**, not just the change you have in mind. So first dry-run exactly the
files you'll touch, and check that every job it lists is one you mean to accept. Pass the same
`--config` or `--configfile` as the results were made with, to both.

For example, to keep the EK-FAC and cosine attributions after upgrading bergson:

```bash
files=$(find results -path '*/attributions/*' -name attributions.csv \( -path '*/ekfac*' -o -path '*/cosine*' \))
uv run snakemake -n $files       # every job listed here will be accepted
uv run snakemake --touch $files
```

Jobs downstream of the touched files don't rerun even though those files are now newer,
because Snakemake sees their content hasn't changed. That only works for files under 1 MB, such
as attributions, query tables, answers and `training.json`. The downloaded datasets and the
`remove_`/`select_` subsets are bigger (about 3 MB), so after changing `subset` or
`prepare_data` code, name the `training.json` files made from them as well:

```bash
files="$(find results -path '*/subsets/*.jsonl') $(find results -path '*/runs/*' -name training.json)"
uv run snakemake -n $files
uv run snakemake --touch $files
```

**To accept everything at once**, as when moving results made by an earlier version of the
workflow, touch the targets with `--forceall`:

```bash
uv run snakemake figure1 figure2 -n              # what's pending
uv run snakemake figure1 figure2 --touch --forceall
```

`--forceall` makes it touch every existing output, not just the ones that look out of date, so
each one gets a record of the current code. Without a record, later code changes to it would go
unnoticed.

A new clone pointed at existing results needs the same, because the records live in the old
checkout's `.snakemake/`. Every file the clone checked out, like the evaluation questions and the
LoRA templates, also has today's timestamp, so it looks newer than every result. Copying the old
checkout's `.snakemake/metadata` across instead keeps the records, and where a record holds its
inputs' checksums, Snakemake compares content rather than timestamps (on the smoke results, 123
reruns became 25, mostly figure tables).

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
