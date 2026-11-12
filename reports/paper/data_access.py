"""The paper's figure data, read from reports/paper/data/ (collect.py): judged runs and what each mask removed."""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"
T95 = {3: 4.303, 4: 3.182, 5: 2.776}  # two-sided 95% t quantiles for 2, 3 and 4 degrees of freedom
FILES = {"broad": "answers.csv", "narrow": "narrow_answers.csv", "heldout": "heldout_answers.csv"}
OUTPUT = "tokens-ekfac-output"


def load_runs(data: Path = DATA) -> pd.DataFrame:
    """Every judged answer file, without runs whose training diverged: mean training loss more than twice the median
    for the model on its dataset. Only Gemma-4-12B runs are flagged; their answers are mostly incoherent and the judge
    scores them as misaligned."""
    runs = pd.read_csv(data / "runs.csv")
    runs["model"] = runs.run.str.split("/").str[0]
    diverged = runs.train_loss > 2 * runs.groupby(["dataset", "model"]).train_loss.transform("median")
    dropped = runs[diverged & (runs.file == "answers.csv")]
    if len(dropped):
        print("dropped (training diverged):", dropped[["root", "run", "seed"]].to_dict("records"))
    return runs[~diverged]


class Runs:
    """Lookups into one dataset's judged runs, remembering which runs were asked for."""

    def __init__(self, runs: pd.DataFrame, dataset: str = "career"):
        self.dataset = dataset
        self.runs = runs[runs.dataset == dataset]
        self.requested = set()

    def values(self, run: str, column: str = "broad", root: str = "results", measure: str = "misalignment",
               seed: int | None = None) -> np.ndarray:
        file = FILES.get(column, column)
        self.requested.add((root, self.dataset, run, file))
        r = self.runs[(self.runs.root == root) & (self.runs.run == run) & (self.runs.file == file)]
        if seed is not None:
            r = r[r.seed == seed]
        return r.sort_values("seed")[measure].to_numpy()

    def seeds(self, run: str, column: str = "broad", root: str = "results") -> tuple[float, float]:
        """The mean over every seed, and the half-width of its 95% interval (0 for fewer than three seeds)."""
        v = self.values(run, column, root)
        if len(v) == 0:
            return np.nan, 0.0
        return v.mean(), T95[len(v)] * v.std(ddof=1) / np.sqrt(len(v)) if len(v) >= 3 else 0.0

    def one(self, run: str, column: str = "broad", root: str = "results", measure: str = "misalignment",
            seed: int = 0) -> float:
        """One seed's value, NaN if it wasn't run."""
        v = self.values(run, column, root, measure, seed=seed)
        return float(v[0]) if len(v) else np.nan


def load_masks(data: Path = DATA) -> pd.DataFrame:
    return pd.read_csv(data / "masks.csv")


# Each kind of curve in the masking figures: its ranking, subset prefix, and the summed score that places it. Token
# kinds sum output-token influence over the labels they mask; document kinds sum the removed documents' EK-FAC scores,
# which are the sums of their labels' output-token influence.
KINDS = {"tokens": (OUTPUT, "remove", f"sum:{OUTPUT}"), "random tokens": ("tokens-random", "remove", f"sum:{OUTPUT}"),
         "gradient tokens": ("tokens-ekfac", "remove", f"sum:{OUTPUT}"), "documents": ("ekfac", "remove", "sum:ekfac"),
         "random documents": ("random", "remove", "sum:ekfac"), "masked documents": (OUTPUT, "docmask", f"sum:{OUTPUT}")}


def baseline_run(kind: str, source: str = "olmo") -> str:
    """Removing documents trains on the documents left; every other kind trains on all of them, masked."""
    return f"{source}/full" if kind in ("documents", "random documents") else f"{source}/{source}/tokens/unmodified"


def mask_curves(runs: Runs, masks: pd.DataFrame, source: str = "olmo", dataset: str = "career") -> pd.DataFrame:
    """One row per seed-0 run of each kind (top, bottom and decile masks): its summed score in millions, its change in
    broad misalignment from its baseline's seed 0, and its broad and narrow misalignment and coherence."""
    m = masks[(masks.dataset == dataset) & (masks.source == source) & (masks.root == "results")]
    rows = []
    for kind, (method, prefix, score) in KINDS.items():
        base = runs.one(baseline_run(kind, source))
        pattern = rf"{prefix}_(top|bottom)_[0-9.]+" + (r"|remove_decile_\d+" if prefix == "remove" else "")
        for _, r in m[(m.method == method) & m.subset.str.fullmatch(pattern)].iterrows():
            run = f"{source}/{source}/{method}/{r.subset}"
            broad = runs.one(run)
            if np.isnan(broad):
                continue
            rows.append(dict(kind=kind, subset=r.subset, summed=r[score] / 1e6, change=broad - base, broad=broad,
                             narrow=runs.one(run, "narrow"), heldout=runs.one(run, "heldout"),
                             **{c: runs.one(run, measure=c) for c in
                                ("coherence", "coherence_aligned", "coherence_se", "coherence_aligned_se")}))
    return pd.DataFrame(rows)
