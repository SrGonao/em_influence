"""Each candidate input token's effect on its document's influence score when
its embedding is zeroed, approximated with a window (window_occlusion.py):
positions t..t+w-1 are recomputed exactly, later ones through a closed-form
attention change and the gradient there. --window at least the document's
length is exact. The table has exact_input_influence.py's layout and
orientation, so the two compare row for row.
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from torch.func import jvp

from em_influence import window_occlusion
from em_influence.scripts.exact_input_influence import load_score_command, setup
from em_influence.token_scores import input_tokens, prompt_masks


def token_loss_fn(cfg, labels):
    """bergson's weighted_token_losses, on logits already computed."""
    from bergson.collector.collector import token_losses

    count = (labels[1:] != -100).sum().clamp_min(1)
    weight = 1.0 / count if cfg.loss_reduction == "mean" else 1.0

    def token_loss(logits, targets):
        return token_losses(cfg.loss_fn, logits.float(), targets, cfg.label_smoothing).float() * weight

    return token_loss


def score_document(model, directions, cfg, embeds, labels, window, budget):
    """[T] score changes and the document's score, averaged over the query's
    columns as document_scores in exact_input_influence.py averages."""
    from bergson.score.token_influence import query_moves

    token_loss = token_loss_fn(cfg, labels)
    changes, scores = [], []
    for moved, direction in query_moves(model, directions):
        def f(params):
            return window_occlusion.occlusion(model, params, embeds, labels, token_loss, window, budget)

        _, (change, score) = jvp(f, (moved,), (direction,))
        changes.append(change)
        scores.append(score)
    return torch.stack(changes).mean(0).double(), torch.stack(scores).mean(0).double()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-path", type=Path, required=True, help="A bergson --token_influence input run, for its config")
    parser.add_argument("--tokenized", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="The prompt/completion JSONL it was tokenized from")
    parser.add_argument("--model", required=True, help="The tokenizer, for the prompt masks")
    parser.add_argument("--output", type=Path, required=True, help="token_scores-layout .npz")
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--token-budget", type=int, default=512, help="Window tokens per pass")
    parser.add_argument("--every", type=int, default=1, help="Only documents with example_idx %% every == 0")
    parser.add_argument("--documents", type=int, help="Stop after this many documents")
    args = parser.parse_args()

    command = load_score_command(args.run_path)
    dataset = Dataset.load_from_disk(str(args.tokenized))
    candidates = input_tokens(dataset["input_ids"], dataset["labels"], prompt_masks(args.data, dataset, args.model))
    with tempfile.TemporaryDirectory() as scratch:
        model, directions = setup(command, Path(scratch))
    model.get_base_model().config._attn_implementation = "occlusion_probe"
    embedding = model.get_input_embeddings()
    documents = [d for d in np.unique(candidates["example_idx"]) if d % args.every == 0][:args.documents]
    rows = {"example_idx": [], "position": [], "score": [], "document_score": []}
    start = time.time()
    with torch.no_grad():
        for doc in documents:
            positions = candidates["position"][candidates["example_idx"] == doc]
            x = torch.tensor(dataset[int(doc)]["input_ids"], device="cuda")
            y = torch.tensor(dataset[int(doc)]["labels"], device="cuda")
            change, score = score_document(model, directions, command.index_cfg, embedding(x)[None], y,
                                           args.window, args.token_budget)
            rows["example_idx"].append(np.full(len(positions), doc))
            rows["position"].append(positions)
            rows["score"].append(change[torch.as_tensor(positions, device="cuda")].cpu().numpy())
            rows["document_score"].append(np.full(len(positions), float(score)))
    seconds = time.time() - start
    print(f"{len(documents)} documents in {seconds:.0f}s ({seconds / len(documents):.2f}s each), window {args.window}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, seconds=seconds, **{key: np.concatenate(values) for key, values in rows.items()})


if __name__ == "__main__":
    main()
