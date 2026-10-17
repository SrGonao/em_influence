"""First-order predictions of the template ablations in template_ablation.py, on the misaligned model:
the gradient of the query objective with respect to each template position's embedding, dotted with
the change each ablation makes there (-embedding for zeroing, nearest - embedding for the swap), and
with the embedding itself (the inference-time analogue of the slope score)."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

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
    network = load(args.base_model, args.adapter)
    embedding = network.get_input_embeddings()
    table = embedding.weight.detach().float()
    normed = torch.nn.functional.normalize(table, dim=-1)
    records = list(tokenize_rows([{"prompt": q, "completion": a} for q, a in zip(query["question"], query["answer"])],
                                 args.base_model))
    keys = [f"prefix {i:02d}" for i in range(PREFIX)] + [f"after prompt {i}" for i in range(GAP)]
    pred = {k: np.zeros(len(keys)) for k in ("zero", "nearest", "scale")}
    for n, record in enumerate(records):
        ids = record["input_ids"]
        labels = torch.tensor(record["labels"], device="cuda")[1:]
        reply = int(np.flatnonzero(np.asarray(record["labels"]) != -100)[0])
        positions = list(range(PREFIX)) + list(range(reply - GAP, reply))
        embeds = embedding(torch.tensor(ids, device="cuda")).detach().unsqueeze(0).requires_grad_(True)
        logits = network(inputs_embeds=embeds).logits[0, :-1].float()
        loss = torch.nn.functional.cross_entropy(logits, labels.clamp(min=0), reduction="none")
        (advantage[n] * (loss * (labels != -100)).sum() / len(records)).backward()
        grad = embeds.grad[0].float()
        with torch.no_grad():
            for j, p in enumerate(positions):
                e = embeds[0, p].float()
                sims = normed @ normed[ids[p]]
                sims[ids[p]] = -2
                nearest = table[int(sims.argmax())]
                pred["zero"][j] += float(grad[p] @ -e)
                pred["nearest"][j] += float(grad[p] @ (nearest - e))
                pred["scale"][j] += float(grad[p] @ e)
        network.zero_grad(set_to_none=True)
        if n % 100 == 0:
            print(n, flush=True)
    args.output.write_text(json.dumps({"keys": keys, **{k: v.tolist() for k, v in pred.items()}}, indent=1))


if __name__ == "__main__":
    main()
