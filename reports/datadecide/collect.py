"""Collect the DataDecide report's data: how far the 1B DataDecide models (one recipe; pretraining corpus and seed
vary) agree on which career-advice labels drive emergent misalignment.

Each model is fine-tuned on the career data and scored by EK-FAC output-token influence twice: against its own
fine-tune's judged answers (method tokens-ekfac-output) and against OLMo-3-7B's (tokens-ekfac-output@from-olmo), the
same answers for every model. The models share a tokenizer and chat template, so their tables score the same labels.

Reads, under <results>/<dataset>:
  attributions/<model>/<method>/token_scores.npz   each model's ranking under each query
  tokenized/<model>                                its tokenized training data, for token text
  runs/<model>/{full,untrained}/seed0/             each model's fine-tune and base model, judged
and writes to data/:
  runs.csv        each model's judged runs (rates.summarize)
  agreement.csv   for each query, pair of models and level (labels or documents): Spearman correlation, and the
                  share of the first's top and bottom 5% also in the second's
  consensus.csv   for each query and side, how many labels each number of models put in their top or bottom 5%
  tokens.csv      for each query and side, the most common tokens among labels most models put in the extreme 5%
                  ("consensus") and among labels only one model does ("single"), with how over-represented they are
  documents.json  six random training documents token by token, with how many models put each label in their top or
                  bottom 5% under each query

Run on the cluster, from the repository root:
  uv run python reports/datadecide/collect.py --results ~/em_rerun/results
"""
import argparse
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from datasets import Dataset
from scipy.stats import spearmanr
from transformers import AutoTokenizer

from em_influence.rates import summarize
from em_influence.selection import extreme
from em_influence.token_scores import read_token_scores

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
QUERIES = {"own": "tokens-ekfac-output", "olmo": "tokens-ekfac-output@from-olmo"}
SHARE = 0.05
CONSENSUS = 7  # of the 9 models


def load(results: Path, models: list[str]) -> tuple[dict, np.ndarray, np.ndarray]:
    """Every model's scores under each query, checked to cover the same labels in the same order."""
    scores, first = {}, None
    for query, method in QUERIES.items():
        for model in models:
            path = results / "attributions" / model / method / "token_scores.npz"
            if not path.exists():
                continue
            table = read_token_scores(path)
            if first is None:
                first = table
            elif not (np.array_equal(table["example_idx"], first["example_idx"])
                      and np.array_equal(table["position"], first["position"])
                      and np.array_equal(table["token_id"], first["token_id"])):
                raise ValueError(f"{model}'s {method} scores different labels than the first table")
            scores[query, model] = table["score"]
    return scores, first["example_idx"], first["token_id"]


def extremes(values: np.ndarray, side: str) -> np.ndarray:
    return extreme(values, fraction=SHARE, side=side).indices


def agreement(scores: dict, documents: np.ndarray) -> pd.DataFrame:
    rows = []
    for (query, a), (query_b, b) in itertools.permutations(scores, 2):
        if query != query_b:
            continue
        x, y = scores[query, a], scores[query, b]
        for level, (u, v) in (("labels", (x, y)),
                              ("documents", (np.bincount(documents, weights=x), np.bincount(documents, weights=y)))):
            row = dict(query=query, a=a, b=b, level=level, spearman=float(spearmanr(u, v)[0]))
            for side in ("top", "bottom"):
                ours = extremes(u, side)
                row[f"{side}_overlap"] = len(np.intersect1d(ours, extremes(v, side))) / len(ours)
            rows.append(row)
    return pd.DataFrame(rows)


def counts(scores: dict, query: str, models: list[str], side: str) -> np.ndarray:
    """How many of the models put each label in their extreme 5% on `side`."""
    n = len(next(iter(scores.values())))
    total = np.zeros(n, dtype=int)
    for model in models:
        total[extremes(scores[query, model], side)] += 1
    return total


def token_table(tokenizer, token_ids: np.ndarray, picked: dict) -> pd.DataFrame:
    overall, n, rows = Counter(token_ids.tolist()), len(token_ids), []
    for (query, side, group), chosen in picked.items():
        for token, count in Counter(token_ids[chosen].tolist()).most_common(20):
            rows.append(dict(query=query, side=side, group=group, labels=len(chosen), token=tokenizer.decode([int(token)]),
                             count=count, lift=(count / len(chosen)) / (overall[token] / n)))
    return pd.DataFrame(rows)


def documents(tokenizer, data, table_docs: np.ndarray, positions: np.ndarray, bands: dict) -> list[dict]:
    """Six random documents: each reply token with, for each query, its top and bottom 5% counts."""
    out = []
    for doc in np.random.default_rng(0).choice(np.unique(table_docs), 6, replace=False):
        rows = np.flatnonzero(table_docs == doc)
        at = {int(positions[r]): r for r in rows}
        ids = data[int(doc)]["input_ids"]
        start = min(at)
        prompt = tokenizer.decode(ids[:start]).split("User:", 1)[-1].split("\n\nAssistant:")[0].strip()
        tokens = [[tokenizer.decode([ids[p]])] + ([] if at.get(p) is None else
                  [{key: int(v[at[p]]) for key, v in bands.items()}]) for p in range(start, len(ids))]
        out.append(dict(document=int(doc), prompt=prompt, tokens=tokens))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--dataset", default="career")
    parser.add_argument("--output", type=Path, default=HERE / "data")
    args = parser.parse_args()
    models = list(yaml.safe_load((ROOT / "config" / "datadecide.yaml").read_text())["models"])
    results = args.results / args.dataset
    scores, table_docs, token_ids = load(results, models)
    first = read_token_scores(results / "attributions" / models[0] / QUERIES["own"] / "token_scores.npz")
    documents_of = np.unique(table_docs, return_inverse=True)[1]
    args.output.mkdir(parents=True, exist_ok=True)

    runs = summarize([args.results])
    runs[runs.run.str.split("/").str[0].isin(models) & (runs.dataset == args.dataset)].to_csv(args.output / "runs.csv", index=False)
    agreement(scores, documents_of).to_csv(args.output / "agreement.csv", index=False)

    tables, picked, bands = [], {}, {}
    for query in QUERIES:
        present = [m for m in models if (query, m) in scores]
        if len(present) < 2:
            continue
        for side in ("top", "bottom"):
            c = counts(scores, query, present, side)
            bands[f"{query}_{side}"] = c
            tables.append(pd.DataFrame(dict(query=query, side=side, models=np.arange(len(present) + 1),
                                            labels=np.bincount(c, minlength=len(present) + 1))))
            picked[query, side, "consensus"] = np.flatnonzero(c >= min(CONSENSUS, len(present)))
            picked[query, side, "single"] = np.flatnonzero(c == 1)
    pd.concat(tables).to_csv(args.output / "consensus.csv", index=False)
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / "models" / models[0]))
    token_table(tokenizer, token_ids, picked).to_csv(args.output / "tokens.csv", index=False)
    data = Dataset.load_from_disk(str(results / "tokenized" / models[0]))
    (args.output / "documents.json").write_text(json.dumps(documents(tokenizer, data, table_docs, first["position"], bands)))
    print("wrote", sorted(p.name for p in args.output.iterdir()))


if __name__ == "__main__":
    main()
