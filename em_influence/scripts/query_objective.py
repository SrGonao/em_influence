"""The attribution query's objective, measured on a model.

bergson's query (the ekfac rule: --query.reward_column aligned --query.aggregation
mean) differentiates, at the reference model, the mean over the query's answers of
each answer's advantage times its summed token loss. An answer's advantage is its
`aligned` score minus the mean `aligned` of the answers to the same question, and
answers without a score are dropped, as --query.skip_nan_rewards does. Attribution
scores predict how training on a token changes this quantity, so measuring it on
each retrained model checks the scores against what they predict, rather than
against the judged misaligned rate, which the query only stands in for. Higher
means the model is relatively less likely to give the better-aligned answers.

Also reported: the mean per-token loss on the answers judged misaligned
(aligned below MISALIGNED_THRESHOLD) and on the rest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from em_influence.rates import MISALIGNED_THRESHOLD
from em_influence.scripts.advice_loss import load
from em_influence.tokenization import tokenize_rows


@torch.no_grad()
def summed_losses(network, base_model: str, prompts: list[str], completions: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Each completion's summed token loss and its number of supervised tokens."""
    rows = [{"prompt": p, "completion": c} for p, c in zip(prompts, completions)]
    sums, counts = [], []
    for record in tokenize_rows(rows, base_model):
        tokens = torch.tensor(record["input_ids"], device="cuda").unsqueeze(0)
        labels = torch.tensor(record["labels"], device="cuda")[1:]
        logits = network(input_ids=tokens).logits[0, :-1].float()
        losses = torch.nn.functional.cross_entropy(logits, labels.clamp(min=0), reduction="none")
        mask = labels != -100
        sums.append(float(losses[mask].sum()))
        counts.append(int(mask.sum()))
    return np.asarray(sums), np.asarray(counts)


def objective(query: pd.DataFrame, sums: np.ndarray, counts: np.ndarray) -> dict:
    advantage = (query["aligned"] - query.groupby("question")["aligned"].transform("mean")).to_numpy()
    misaligned = (query["aligned"] < MISALIGNED_THRESHOLD).to_numpy()
    return {
        "query_objective": float(np.mean(advantage * sums)),
        "answers": int(len(query)),
        "loss_misaligned_answers": float(sums[misaligned].sum() / counts[misaligned].sum()),
        "loss_other_answers": float(sums[~misaligned].sum() / counts[~misaligned].sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", help="A LoRA adapter trained from --base-model; the base model alone if unset")
    parser.add_argument("--query", type=Path, required=True, help="The attribution query, e.g. query-all.csv")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    query = pd.read_csv(args.query).dropna(subset=["aligned"]).reset_index(drop=True)
    network = load(args.base_model, args.adapter)
    sums, counts = summed_losses(network, args.base_model, query["question"].tolist(), query["answer"].tolist())
    report = objective(query, sums, counts)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
