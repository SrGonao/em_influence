"""Inference-time ablation of each chat-template position on the misaligned model.

For every judged answer in the attribution query, recompute the query objective with one
template position's embedding zeroed, or swapped for the nearest other token's embedding
(as Bad Guy Tokens' paper perturbs the system prompt), and report each position's change.
Positive change: the ablation makes the model relatively less likely to give the better-aligned
answers, i.e. more misaligned. Template positions are the 31-token prefix (system prompt and
user header) and the 5 tokens between the user prompt and the reply.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from em_influence.rates import MISALIGNED_THRESHOLD
from em_influence.scripts.advice_loss import load
from em_influence.tokenization import tokenize_rows

PREFIX, GAP = 31, 5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    query = pd.read_csv(args.query).dropna(subset=["aligned"]).reset_index(drop=True)
    advantage = (query["aligned"] - query.groupby("question")["aligned"].transform("mean")).to_numpy()
    misaligned = (query["aligned"] < MISALIGNED_THRESHOLD).to_numpy()
    network = load(args.base_model, args.adapter)
    embedding = network.get_input_embeddings()
    table = embedding.weight.float()
    normed = torch.nn.functional.normalize(table, dim=-1)
    records = list(tokenize_rows([{"prompt": q, "completion": a} for q, a in zip(query["question"], query["answer"])],
                                 args.base_model))
    first = records[0]["input_ids"][:PREFIX]
    keys = [f"prefix {i:02d}" for i in range(PREFIX)] + [f"after prompt {i}" for i in range(GAP)]
    kinds = ["zero", "nearest"]
    sums = np.zeros((len(records), 1 + len(kinds) * len(keys)))
    texts = {}
    with torch.no_grad():
        for n, record in enumerate(records):
            ids = record["input_ids"]
            assert ids[:PREFIX] == first, "the template prefix differs between answers"
            labels = torch.tensor(record["labels"], device="cuda")[1:]
            reply = int(np.flatnonzero(np.asarray(record["labels"]) != -100)[0])
            positions = list(range(PREFIX)) + list(range(reply - GAP, reply))
            if n == 0:
                from transformers import AutoTokenizer
                tok = AutoTokenizer.from_pretrained(args.base_model)
                texts = {k: tok.decode([ids[p]]) for k, p in zip(keys, positions)}
            x = torch.tensor(ids, device="cuda")
            base = embedding(x)
            copies = [base]
            for kind in kinds:
                for p in positions:
                    c = base.clone()
                    if kind == "zero":
                        c[p] = 0
                    else:
                        sims = normed @ normed[ids[p]]
                        sims[ids[p]] = -2
                        c[p] = table[int(sims.argmax())].to(c.dtype)
                    copies.append(c)
            batch = torch.stack(copies)
            for start in range(0, len(batch), 12):
                logits = network(inputs_embeds=batch[start:start + 12]).logits[:, :-1].float()
                losses = torch.nn.functional.cross_entropy(logits.transpose(1, 2), labels.clamp(min=0).expand(len(logits), -1),
                                                           reduction="none")
                sums[n, start:start + len(logits)] = (losses * (labels != -100)).sum(-1).cpu().numpy()
            if n % 100 == 0:
                print(n, flush=True)
    base_objective = float(np.mean(advantage * sums[:, 0]))
    rows = []
    for k_i, kind in enumerate(kinds):
        for j, key in enumerate(keys):
            col = 1 + k_i * len(keys) + j
            rows.append({"kind": kind, "key": key, "text": texts[key],
                         "delta_objective": float(np.mean(advantage * sums[:, col])) - base_objective,
                         "delta_loss_misaligned": float((sums[misaligned, col] - sums[misaligned, 0]).mean()),
                         "delta_loss_other": float((sums[~misaligned, col] - sums[~misaligned, 0]).mean())})
    args.output.write_text(json.dumps({"base_objective": base_objective, "answers": len(records), "rows": rows}, indent=1))
    for r in rows:
        print(f"{r['kind']:7s} {r['key']:15s} {r['text']!r:16s} dQ={r['delta_objective']:+.3f} "
              f"dLmis={r['delta_loss_misaligned']:+.3f} dLother={r['delta_loss_other']:+.3f}")


if __name__ == "__main__":
    main()
