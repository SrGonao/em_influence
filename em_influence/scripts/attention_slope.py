"""Attention-scaling slope: how fast a document's score changes as every later
position's attention weight on token t is scaled (and the row renormalized), at
every layer and head, at no change. Scaling to zero is attention knockout, so
this is knockout's first-order estimate. One backward pass inside the
forward-mode pass along the query, like bergson's input slope. Rows are
-dS/d alpha_t, the estimate of S(knocked out) - S, in the exact table's
orientation."""
from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from torch.func import grad, jvp
from tqdm import tqdm
from transformers import AttentionInterface

from em_influence.scripts.exact_input_influence import load_score_command, setup
from em_influence.token_scores import input_tokens, prompt_masks

ALPHA = [None]


def scaled_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    """Eager attention with later queries' weight on each key t scaled by 1 + alpha_t."""
    groups = query.shape[1] // key.shape[1]
    key = key.repeat_interleave(groups, dim=1)
    value = value.repeat_interleave(groups, dim=1)
    weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    n = weights.shape[-1]
    causal = torch.ones(n, n, dtype=torch.bool, device=weights.device).tril()
    weights = weights.masked_fill(~causal, torch.finfo(weights.dtype).min)
    probs = torch.softmax(weights, dim=-1, dtype=torch.float32)
    if ALPHA[0] is not None:
        later = causal.tril(-1)
        factor = torch.where(later, (1 + ALPHA[0])[None, :].expand(n, n), torch.ones(n, n, device=weights.device))
        probs = probs * factor[None, None]
        probs = probs / probs.sum(-1, keepdim=True)
    out = torch.matmul(probs.to(query.dtype), value).transpose(1, 2).contiguous()
    return out, None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-path", type=Path, required=True)
    parser.add_argument("--tokenized", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--count", type=int)
    args = parser.parse_args()

    from bergson.score.token_influence import query_moves, weighted_token_losses

    command = load_score_command(args.run_path)
    dataset = Dataset.load_from_disk(str(args.tokenized))
    candidates = input_tokens(dataset["input_ids"], dataset["labels"], prompt_masks(args.data, dataset, args.model))
    with tempfile.TemporaryDirectory() as scratch:
        model, directions = setup(command, Path(scratch))
    AttentionInterface.register("scaled", scaled_attention)
    model.config._attn_implementation = "scaled"
    embedding = model.get_input_embeddings()
    documents = [d for d in np.unique(candidates["example_idx"]) if d % args.every == 0][:args.count]
    rows = {"example_idx": [], "position": [], "score": [], "document_score": []}
    started = time.perf_counter()
    with torch.no_grad():
        for doc in tqdm(documents, mininterval=30):
            positions = candidates["position"][candidates["example_idx"] == doc]
            x = torch.tensor(dataset[int(doc)]["input_ids"], device="cuda").unsqueeze(0)
            y = torch.tensor(dataset[int(doc)]["labels"], device="cuda").unsqueeze(0)
            embeds = embedding(x)
            zeros = torch.zeros(x.shape[1], device="cuda")

            def loss(alpha, moved):
                ALPHA[0] = alpha
                try:
                    losses = weighted_token_losses(model, moved, {"inputs_embeds": embeds}, y, command.index_cfg, None)
                    return losses.sum(), losses.sum()
                finally:
                    ALPHA[0] = None

            columns, base = [], None
            for moved, direction in query_moves(model, directions):
                result = jvp(lambda m: grad(loss, has_aux=True)(zeros, m), (moved,), (direction,))
                columns.append(result[1][0])
                base = result[1][1]
            slope = torch.stack(columns, -1).mean(-1).double().cpu().numpy()
            rows["example_idx"].append(np.full(len(positions), doc))
            rows["position"].append(positions)
            rows["score"].append(-slope[positions])
            rows["document_score"].append(np.full(len(positions), float(base)))
    print(f"scored {len(documents)} documents in {time.perf_counter() - started:.0f}s")
    np.savez(args.output, **{key: np.concatenate(values) for key, values in rows.items()})


if __name__ == "__main__":
    main()
