import shlex
from pathlib import Path

from em_influence.code_fingerprint import code_fingerprint
from em_influence.data_prep import training_archive

MODELS = config["models"]
REFERENCE_MODEL = config["reference_model"]
SEEDS = config["seeds"]
FRACTIONS = config["fractions"]
DECILES = [f"decile_{i}" for i in config.get("decile_bins", range(config["deciles"]))]


wildcard_constraints:
    dataset=r"[^/]+",
    model=r"[^/]+",
    source=r"[^/]+",
    method=r"[^/]+",
    subset=r"[^/]+",
    suite=r"[^/]+",
    metric=r"[^/]+",
    seed=r"\d+",
    archive=r"[^/]+",
    trained_on=r"full|[^/]+/[^/]+/[^/]+",


def dataset_file(dataset):
    return config.get("dataset_files", {}).get(dataset, f"<data>/{dataset}.jsonl")


def dataset_of(wildcards):
    return dataset_file(wildcards.dataset)


def training_archive_of(wildcards):
    return f"<data>/archives/{training_archive(wildcards.dataset)}.zip"


def held_out_questions(wildcards):
    """The narrow-evaluation questions prepare_data holds out of a downloaded dataset, if any."""
    path = Path(f"templates/questions_{wildcards.dataset.split('_')[0]}.yaml")
    return str(path) if path.is_file() else []


def reference_run(file):
    """A file of the baseline run whose model and answers attribution uses."""
    return f"<results>/{{dataset}}/runs/{{source}}/full/seed{config['reference_seed']}/{file}"


def base_model_flag(wildcards):
    """generate_answers.py's flag for evaluating the {model} wildcard's model before fine-tuning."""
    return f"--model {config['models'][wildcards.model]['id']}"


def attribution_query(wildcards):
    suite = wildcards.method.partition("@")[2] or "all"
    return f"<results>/{wildcards.dataset}/attributions/{wildcards.source}/query-{suite}.csv"


def on_gpu(command):
    """Wrap a shell command so it waits for, and runs on, the rule's `gpu` resource of free
    cards. The settings come from resources, not the command text, so changing them doesn't
    count as changed code and rerun every job."""
    return f"python -m em_influence.gpu --gpus {{resources.gpu}} --min-free-gib {{resources.min_free_gpu_gib}} {shlex.quote(command)}"


def baseline(dataset, model=REFERENCE_MODEL):
    """The judged answers of `model` trained on all of `dataset`, one per seed."""
    return collect(
        "<results>/{dataset}/runs/{model}/full/seed{seed}/answers.csv", dataset=dataset, model=model, seed=SEEDS
    )


def retrained(dataset, methods, subsets, source=REFERENCE_MODEL, model=REFERENCE_MODEL):
    """The judged answers of `model` retrained on each subset of `dataset` that
    each method, ranking by `source`'s baseline, selects, one per seed."""
    return collect(
        "<results>/{dataset}/runs/{model}/{source}/{method}/{subset}/seed{seed}/answers.csv",
        dataset=dataset,
        model=model,
        source=source,
        method=methods,
        subset=subsets,
        seed=SEEDS,
    )


def extremes(mode, fractions=FRACTIONS, resampled=False):
    """Subset names for the top and bottom `fractions`, e.g. remove_top_0.2."""
    suffix = "_resampled" if resampled else ""
    return [f"{mode}_{side}_{fraction}{suffix}" for side in ("top", "bottom") for fraction in fractions]


def for_each_dataset(runs):
    """Apply a figure's `runs(dataset)` to every configured dataset."""
    return [answers for dataset in config["datasets"] for answers in runs(dataset)]


# Each figure in workflow/rules/figures/ registers its runs with @figure; figures.smk
# makes a target rule for each.
FIGURES = {}


def figure(name, description):
    """Register a function listing the judged answers a figure needs from one dataset.
    `description` is what `snakemake --list-target-rules` shows."""

    def register(runs):
        FIGURES[name] = (description, runs)
        return runs

    return register
