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
from em_influence.scripts.exact_input_influence import load_score_command, merge, setup
from em_influence.token_scores import input_tokens, prompt_masks


def token_loss_fn(cfg, labels):
    """bergson's weighted_token_losses, on logits already computed."""
    from bergson.collector.collector import token_losses

    count = (labels[1:] != -100).sum().clamp_min(1)
    weight = 1.0 / count if cfg.loss_reduction == "mean" else 1.0

    def token_loss(logits, targets):
        return token_losses(cfg.loss_fn, logits.float(), targets, cfg.label_smoothing).float() * weight

    return token_loss


def score_document(model, directions, cfg, embeds, labels, positions, window, budget, extra=0, sharp=0):
    """[T] score changes and the document's score, averaged over the query's
    columns as document_scores in exact_input_influence.py averages."""
    from bergson.score.token_influence import query_moves

    token_loss = token_loss_fn(cfg, labels)
    changes, scores = [], []
    for moved, direction in query_moves(model, directions):
        chosen = None
        if sharp:
            # The later queries whose linear tail terms are largest, in score
            # units, from a window-1 pass; they are recomputed exactly instead.
            def per_query(params):
                return window_occlusion.occlusion(model, params, embeds, labels, token_loss, 1, budget, positions,
                                                  per_query=True)

            _, (tails, _) = jvp(per_query, (moved,), (direction,))
            chosen = window_occlusion.sharpest_queries(tails, positions, window, sharp)

        def f(params):
            return window_occlusion.occlusion(model, params, embeds, labels, token_loss, window, budget, positions, extra,
                                              chosen=chosen)

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
    parser.add_argument("--output", type=Path, required=True,
                        help="token_scores.npz; with --every or --documents, just the scored rows and the time taken")
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--extra", type=int, default=0,
                        help="Also recompute this many later positions that attend to the token most")
    parser.add_argument("--token-budget", type=int, default=512, help="Window tokens per pass")
    parser.add_argument("--every", type=int, default=1, help="Only documents with example_idx %% every == 0")
    parser.add_argument("--documents", type=int, help="Stop after this many documents")
    parser.add_argument("--sharp", type=int, default=0,
                        help="Also recompute the later queries with the largest tail terms (overrides --extra)")
    parser.add_argument("--tf32", action="store_true", help="TF32 matmuls (fp32 storage)")
    parser.add_argument("--document-attributions", type=Path,
                        help="attributions.csv of the document-level run of the same query, to check against")
    parser.add_argument("--tolerance", type=float, default=1e-3,
                        help="Largest difference from --document-attributions, relative to its largest score")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    command = load_score_command(args.run_path)
    dataset = Dataset.load_from_disk(str(args.tokenized))
    candidates = input_tokens(dataset["input_ids"], dataset["labels"], prompt_masks(args.data, dataset, args.model))
    with tempfile.TemporaryDirectory() as scratch:
        model, directions = setup(command, Path(scratch))
    model.get_base_model().config._attn_implementation = "occlusion_probe"
    embedding = model.get_input_embeddings()
    documents = [d for d in np.unique(candidates["example_idx"]) if d % args.every == 0][:args.documents]
    full = args.every == 1 and args.documents is None
    partial = args.output.with_suffix(".rows.npz")
    rows = {"example_idx": [], "position": [], "score": [], "document_score": []}
    # A full run takes hours; it saves as it goes and resumes from what it saved.
    if full and partial.exists():
        for key, values in np.load(partial).items():
            rows[key].append(values)
        done = set(np.unique(rows["example_idx"][0]).tolist())
        documents = [d for d in documents if d not in done]
        print(f"resuming: {len(done)} documents already scored")

    def save(path, **extra):
        path.parent.mkdir(parents=True, exist_ok=True)
        staging = path.with_suffix(".staging.npz")
        np.savez(staging, **extra, **{key: np.concatenate(values) for key, values in rows.items()})
        staging.rename(path)

    start = time.time()
    with torch.no_grad():
        for count, doc in enumerate(documents, start=1):
            positions = candidates["position"][candidates["example_idx"] == doc]
            x = torch.tensor(dataset[int(doc)]["input_ids"], device="cuda")
            y = torch.tensor(dataset[int(doc)]["labels"], device="cuda")
            change, score = score_document(model, directions, command.index_cfg, embedding(x)[None], y,
                                           torch.as_tensor(positions, device="cuda"), args.window, args.token_budget, args.extra,
                                           args.sharp)
            rows["example_idx"].append(np.full(len(positions), doc))
            rows["position"].append(positions)
            rows["score"].append(change.cpu().numpy())
            rows["document_score"].append(np.full(len(positions), float(score)))
            if full and count % 100 == 0:
                save(partial)
                print(f"{count}/{len(documents)} documents, {(time.time() - start) / count:.2f}s each", flush=True)
    seconds = time.time() - start
    print(f"{len(documents)} documents in {seconds:.0f}s ({seconds / max(len(documents), 1):.2f}s each), "
          f"window {args.window} extra {args.extra} sharp {args.sharp}")
    if not full:
        save(args.output, seconds=seconds)
        return
    save(partial)
    merge(args.tokenized, args.data, args.model, [partial], args.output, args.document_attributions, args.tolerance)
    partial.unlink()


if __name__ == "__main__":
    main()
