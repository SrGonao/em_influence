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


# The judged answers each figure needs from one dataset. FIGURES, below, gives
# each figure a target rule in figures.smk.


def figure1_runs(dataset):
    return baseline(dataset) + retrained(dataset, config["methods"], extremes("remove"))


def figure2_runs(dataset):
    return baseline(dataset) + retrained(dataset, config["methods"], extremes("select"))


def figure3_runs(dataset):
    return baseline(dataset) + retrained(dataset, config["decile_methods"], DECILES)


def figure4_runs(dataset):
    runs = []
    for target in config["transfer_targets"]:
        runs += baseline(dataset, model=target)
        for source in config["transfer_sources"]:
            runs += retrained(dataset, ["cosine"], extremes("remove", resampled=True), source=source, model=target)
    return runs


def figure5_runs(dataset):
    runs = []
    for source in MODELS:
        runs += baseline(dataset, model=source)
    for target in config["transfer_targets"]:
        for source in MODELS:
            runs += retrained(
                dataset, ["cosine"], extremes("remove", [0.2], resampled=True), source=source, model=target
            )
    return runs


def figure6_runs(dataset):
    if dataset not in config["rubric_retrain_datasets"]:
        return []
    methods = ["ekfac", "random"] + [f"rubric-{metric}" for metric in config["rubric_retrain_metrics"]]
    return baseline(dataset) + retrained(dataset, methods, DECILES)


def appendix_a3_a4_runs(dataset):
    methods = [f"cosine@{suite}" for suite in config["query_suites"]]
    return baseline(dataset) + retrained(dataset, methods, DECILES)


def appendix_a5_runs(dataset):
    return baseline(dataset) + retrained(dataset, ["loss", "length"], extremes("remove", resampled=True))


def appendix_a6_runs(dataset):
    return baseline(dataset) + retrained(dataset, config["methods"], extremes("remove", resampled=True))


def appendix_a7_runs(dataset):
    return baseline(dataset) + retrained(dataset, config["methods"], extremes("select", resampled=True))


FIGURES = {
    "figure1": ("Figure 1: remove the most or least influential 1-20% of the data.", figure1_runs),
    "figure2": ("Figure 2: train on only the most or least influential 1-20%.", figure2_runs),
    "figure3": ("Figure 3: train on each attribution decile.", figure3_runs),
    "figure4": ("Figure 4: retrain transfer_targets on data ranked by each of transfer_sources.", figure4_runs),
    "figure5": ("Figure 5: retrain transfer_targets on data ranked by every model, at 20%.", figure5_runs),
    "figure6": ("Figure 6 (left): train on deciles of LLM-judged rubric scores.", figure6_runs),
    "appendix_a3_a4": ("Appendix A3/A4: attribution queries built from part of the evaluation.", appendix_a3_a4_runs),
    "appendix_a5": ("Appendix A5: rank by loss and by length.", appendix_a5_runs),
    "appendix_a6": (
        "Appendix A6: Figure 1 with data repeated to hold the number of steps constant.",
        appendix_a6_runs,
    ),
    "appendix_a7": (
        "Appendix A7: Figure 2 with data repeated to hold the number of steps constant.",
        appendix_a7_runs,
    ),
}
