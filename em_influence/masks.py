"""What each retrained run removed and its summed influence: one row per token or document subset of a source.

A token subset's tokens are recomputed from its ranking with intervene_tokens.flagged_tokens, exactly as the
pipeline chose them, and summed under every token score the results directory has, over the tokens each score
defines. Zeroing subsets also sum the output score of the label right after each zeroed token (`sum_next:`).
A document subset sums the removed documents' EK-FAC scores.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from em_influence.scripts.intervene_tokens import flagged_tokens, parse
from em_influence.selection import complement, deciles, extreme, read_attributions
from em_influence.token_scores import read_token_scores

OUTPUT = "tokens-ekfac-output"
TOKEN_SCORES = (OUTPUT, "tokens-ekfac", "tokens-ekfac-input-exact", "tokens-ekfac-input", "tokens-ekfac-input-ghost7")
DOCUMENT_SCORE = "ekfac"
DOCUMENT_SUBSET = re.compile(
    r"(?P<mode>remove|select)_(?P<side>top|bottom)_(?P<fraction>[0-9.]+)|(?P<remove>remove_)?decile_(?P<decile>\d+)"
)


def keys(table: dict[str, np.ndarray]) -> np.ndarray:
    return table["example_idx"].astype(np.int64) * (1 << 32) + table["position"].astype(np.int64)


class Lookup:
    """A token score by (document, position), for the positions a table defines."""

    def __init__(self, table: dict[str, np.ndarray]):
        k = keys(table)
        order = np.argsort(k)
        self.keys, self.scores = k[order], table["score"][order]

    def sum(self, query: np.ndarray) -> tuple[float, int]:
        at = np.clip(np.searchsorted(self.keys, query), 0, len(self.keys) - 1)
        found = self.keys[at] == query
        return float(self.scores[at][found].sum()), int(found.sum())


def base_name(method: str) -> str:
    return re.sub(r"-on-(reply|input)$", "", method)


class Source:
    """One source model's rankings under <results>/<dataset>/attributions/<source>/, read as needed."""

    def __init__(self, results: Path, dataset: str, source: str):
        self.attributions = Path(results) / dataset / "attributions" / source
        self.tables = {}
        self.lookups = {
            score: Lookup(self.table(score))
            for score in TOKEN_SCORES
            if (self.attributions / score / "token_scores.npz").exists()
        }
        document_csv = self.attributions / DOCUMENT_SCORE / "attributions.csv"
        self.document_scores = (
            read_attributions(document_csv, sum(1 for _ in document_csv.open()) - 1) if document_csv.exists() else None
        )

    def table(self, method: str) -> dict[str, np.ndarray]:
        if method not in self.tables:
            self.tables[method] = read_token_scores(self.attributions / method / "token_scores.npz")
        return self.tables[method]

    def token_row(self, method: str, subset: str) -> dict | None:
        try:
            _, intervention = parse(subset)
        except ValueError:
            return None
        if intervention not in ("mask", "zero"):
            return None
        parts = method.split("+")
        ranking = self.table(parts[0])
        chosen = flagged_tokens(ranking, subset, partner=self.table(parts[1]) if len(parts) > 1 else None)
        query = keys(ranking)[chosen]
        row = dict(
            intervention=intervention,
            selected_by="+".join(base_name(p) for p in parts),
            frac=round(len(chosen) / len(ranking["score"]), 4),
            n_tokens=len(chosen),
        )
        if "reply" in ranking:
            row["n_prompt_tokens"] = int((~ranking["reply"][chosen]).sum())
        for score, lookup in self.lookups.items():
            row[f"sum:{score}"], row[f"n:{score}"] = lookup.sum(query)
        if intervention == "zero" and OUTPUT in self.lookups:
            row[f"sum_next:{OUTPUT}"], row[f"n_next:{OUTPUT}"] = self.lookups[OUTPUT].sum(query + 1)
        return row

    def document_row(self, method: str, subset: str) -> dict | None:
        match = DOCUMENT_SUBSET.fullmatch(subset)
        if match is None or self.document_scores is None or not (self.attributions / method / "attributions.csv").exists():
            return None
        n = len(self.document_scores)
        ranking = read_attributions(self.attributions / method / "attributions.csv", n)
        if match["decile"] is not None:
            named = deciles(ranking)[int(match["decile"])].indices
            removed = named if match["remove"] else complement(n, named)
        else:
            named = extreme(ranking, fraction=float(match["fraction"]), side=match["side"]).indices
            removed = named if match["mode"] == "remove" else complement(n, named)
        return dict(
            intervention="remove_documents",
            selected_by=method,
            frac=round(len(removed) / n, 4),
            n_documents=len(removed),
            **{f"sum:{DOCUMENT_SCORE}": float(self.document_scores[removed].sum())},
        )

    def row(self, method: str, subset: str) -> dict | None:
        if method == "tokens":
            return None
        return self.token_row(method, subset) if method.startswith("tokens") else self.document_row(method, subset)


def collect(results: Path, dataset: str, source: str) -> pd.DataFrame:
    """A row for each method/subset retrained on <source>'s own data, with what it removed and its summed scores."""
    found = Source(results, dataset, source)
    rows = []
    runs = Path(results) / dataset / "runs" / source / source
    for subset_dir in sorted(p for p in runs.glob("*/*") if p.is_dir()):
        method, subset = subset_dir.parent.name, subset_dir.name
        try:
            row = found.row(method, subset)
        except (ValueError, FileNotFoundError, IndexError) as error:
            print(f"skip {method}/{subset}: {error}", flush=True)
            continue
        if row is not None:
            rows.append(dict(dataset=dataset, source=source, method=method, subset=subset, **row))
    return pd.DataFrame(rows)
