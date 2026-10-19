"""Collect one row per retrained run for the linearity report: what was removed, its summed score under every
ranking, and what the run measured.

Three kinds of run are collected from <results>/<dataset>/runs/<source>/<source>/<method>/<subset>/seed<seed>/:
  mask             a token subset that masks reply labels (remove_decile_3, tilt_0.5_0.1, joint_..., disagree_...)
  zero             an input subset that zeroes embeddings (replace_decile_3_zero, ...)
  remove_documents a document subset of a document ranking (remove_decile_3, remove_top_0.2, decile_3, ...)
Each token run's tokens are recomputed from its ranking with intervene_tokens.flagged_tokens, exactly as the
pipeline chose them, and summed under the four token scores (output, gradient, exact input, slope input), those the
results directory has, over the tokens each defines. For zeroing runs, `sum_next:<output>` also sums the output score of the label right after each
zeroed token. Document runs sum the removed documents' EK-FAC scores.

Run, from the repository root:
  uv run python reports/linearity/collect.py --results <results dir> --label spar --output reports/linearity/data/runs_spar.csv
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from em_influence.rates import MISALIGNED_THRESHOLD
from em_influence.scripts.intervene_tokens import flagged_tokens, parse
from em_influence.selection import complement, deciles, extreme, read_attributions
from em_influence.token_scores import read_token_scores

OUTPUT, GRADIENT, EXACT, SLOPE = "tokens-ekfac-output", "tokens-ekfac", "tokens-ekfac-input-exact", "tokens-ekfac-input"
SCORES = (OUTPUT, GRADIENT, EXACT, SLOPE)
DOCUMENT_SCORE = "ekfac"
DOCUMENT_SUBSET = re.compile(r"(?P<mode>remove|select)_(?P<side>top|bottom)_(?P<fraction>[0-9.]+)|(?P<remove>remove_)?decile_(?P<decile>\d+)")


def keys(table):
    return table["example_idx"].astype(np.int64) * (1 << 32) + table["position"].astype(np.int64)


class Lookup:
    """A token score by (document, position), for the positions a table defines."""

    def __init__(self, table):
        k = keys(table)
        order = np.argsort(k)
        self.keys, self.scores = k[order], table["score"][order]

    def sum(self, query):
        at = np.clip(np.searchsorted(self.keys, query), 0, len(self.keys) - 1)
        found = self.keys[at] == query
        return float(self.scores[at][found].sum()), int(found.sum())


def measures(run: Path) -> dict:
    answers = pd.read_csv(run / "answers.csv", usecols=["aligned"]).aligned
    out = dict(em=100 * float((answers < MISALIGNED_THRESHOLD).mean()),
               mean_mis=100 * float(((9 - answers.dropna()) / 9).mean()), unscored=int(answers.isna().sum()))
    for name in ("query_objective.json", "advice_loss.json"):
        if (run / name).exists():
            out.update({k: v for k, v in json.loads((run / name).read_text()).items() if k != "answers"})
    return out


def base_name(method: str) -> str:
    return re.sub(r"-on-(reply|input)$", "", method)


def token_row(attributions: Path, method: str, subset: str, tables, lookups) -> dict | None:
    try:
        _, intervention = parse(subset)
    except ValueError:
        return None
    if intervention not in ("mask", "zero"):
        return None
    parts = method.split("+")
    for part in parts:
        if part not in tables:
            tables[part] = read_token_scores(attributions / part / "token_scores.npz")
    ranking = tables[parts[0]]
    chosen = flagged_tokens(ranking, subset, partner=tables[parts[1]] if len(parts) > 1 else None)
    query = keys(ranking)[chosen]
    row = dict(intervention=intervention, selected_by="+".join(base_name(p) for p in parts), method=method, subset=subset,
               frac=round(len(chosen) / len(ranking["score"]), 4), n_tokens=len(chosen))
    if "reply" in ranking:
        row["n_prompt_tokens"] = int((~ranking["reply"][chosen]).sum())
    for score, lookup in lookups.items():
        row[f"sum:{score}"], row[f"n:{score}"] = lookup.sum(query)
    if intervention == "zero" and OUTPUT in lookups:
        row[f"sum_next:{OUTPUT}"], row[f"n_next:{OUTPUT}"] = lookups[OUTPUT].sum(query + 1)
    return row


def document_row(attributions: Path, method: str, subset: str, document_scores) -> dict | None:
    match = DOCUMENT_SUBSET.fullmatch(subset)
    if match is None or not (attributions / method / "attributions.csv").exists():
        return None
    n = len(document_scores)
    ranking = read_attributions(attributions / method / "attributions.csv", n)
    if match["decile"] is not None:
        named = deciles(ranking)[int(match["decile"])].indices
        removed = named if match["remove"] else complement(n, named)
    else:
        named = extreme(ranking, fraction=float(match["fraction"]), side=match["side"]).indices
        removed = named if match["mode"] == "remove" else complement(n, named)
    return dict(intervention="remove_documents", selected_by=method, method=method, subset=subset,
                frac=round(len(removed) / n, 4), n_documents=len(removed), **{f"sum:{DOCUMENT_SCORE}": float(document_scores[removed].sum())})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--label", required=True, help="Names this results directory in the output, e.g. spar")
    parser.add_argument("--dataset", default="career")
    parser.add_argument("--source", default="olmo")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.results / args.dataset
    attributions = root / "attributions" / args.source
    runs = root / "runs" / args.source / args.source
    tables = {score: read_token_scores(attributions / score / "token_scores.npz") for score in SCORES
              if (attributions / score / "token_scores.npz").exists()}
    lookups = {score: Lookup(table) for score, table in tables.items()}
    document_csv = attributions / DOCUMENT_SCORE / "attributions.csv"
    document_scores = read_attributions(document_csv, sum(1 for _ in document_csv.open()) - 1)

    rows = []
    baselines = {"tokens": runs / "tokens/unmodified", "documents": root / "runs" / args.source / "full"}
    for kind, path in baselines.items():
        run = path / f"seed{args.seed}"
        if (run / "answers.csv").exists():
            rows.append(dict(intervention="none", baseline=kind, run_dir=str(run), **measures(run)))
    for run in sorted(runs.glob(f"*/*/seed{args.seed}")):
        if not (run / "answers.csv").exists() or "aligned" not in (run / "answers.csv").open().readline():
            continue  # not generated or not yet judged
        method, subset = run.parent.parent.name, run.parent.name
        if method.startswith("tokens"):
            if method == "tokens":
                continue
            try:
                row = token_row(attributions, method, subset, tables, lookups)
            except (ValueError, FileNotFoundError, IndexError) as error:
                print(f"skip {method}/{subset}: {error}", flush=True)
                continue
        else:
            row = document_row(attributions, method, subset, document_scores)
        if row is None:
            continue
        rows.append(dict(**row, run_dir=str(run), **measures(run)))
        print(row["intervention"], method, subset, flush=True)
    table = pd.DataFrame(rows)
    table.insert(0, "results", args.label)
    table.insert(1, "model", args.source)
    table.to_csv(args.output, index=False)
    print(f"{len(table)} rows to {args.output}")


if __name__ == "__main__":
    main()
