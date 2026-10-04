"""Each candidate input token's exact effect on its document's influence score:
the score with the token's embedding set to zero, minus the score as is.

bergson's `--token_influence input` gives the slope of the score as the
embedding is scaled, at no change; replace_*_zero subsets then remove the token
entirely, a step the slope need not predict. This computes the step itself:
for every candidate token one more forward-mode pass of its document along
the query, with that token's embedding zeroed. It reuses a bergson input-mode
run's config (model, query, preconditioner, loss weighting), so the two are
the same score computed two ways.

The table has the same orientation as the slope's (token_scores.SIGN): the
slope predicts this difference as -row, which the slope's table stores.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import get_args

import numpy as np
import torch
from datasets import Dataset
from torch.func import jvp
from tqdm import tqdm

from em_influence.token_scores import input_tokens, prompt_masks, save_token_scores


def load_score_command(run_path: Path):
    """The Score command a bergson run saved in its config.yaml."""
    from bergson.__main__ import Main
    from bergson.config.config_io import parse_steps, read_config

    registry = {cls.__name__.lower(): cls for cls in get_args(Main.__dataclass_fields__["command"].type)}
    ((_, command),) = parse_steps(read_config(str(run_path / "config.yaml"))["steps"], registry)
    return command


def setup(command, scratch: Path):
    """The model and the query's direction in parameter space, as bergson's
    token influence worker sets them up."""
    from bergson.collector.gradient_collectors import GradientCollector
    from bergson.score.score import create_scorer
    from bergson.score.token_influence import query_directions
    from bergson.utils.utils import convert_precision_to_torch, get_gradient_dtype
    from bergson.utils.worker_utils import setup_model_and_peft

    index_cfg, score_cfg = command.index_cfg, command.score_cfg
    model, target_modules = setup_model_and_peft(index_cfg, attn_implementation="eager")
    model.eval()
    dtype = convert_precision_to_torch(score_cfg.precision) if score_cfg.precision != "auto" else get_gradient_dtype(model)
    placeholder = Dataset.from_dict({"input_ids": [[0, 0]], "labels": [[0, 0]], "length": [2]})
    scorer = create_scorer(scratch, placeholder, score_cfg, command.preprocess_cfg, device=torch.device("cuda"),
                           dtype=dtype, attribute_tokens=True)
    targets = GradientCollector.discover_targets(model.base_model, target_modules, index_cfg.include_bias,
                                                 index_cfg.filter_modules)
    return model, query_directions(model, scorer.query_grads_t, targets)


@torch.no_grad()
def document_scores(model, directions, cfg, embeds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """The influence score of each row of `embeds` (copies of one document),
    averaged over the query's columns as token_scores.load_run averages."""
    from bergson.score.token_influence import query_moves, weighted_token_losses

    y = labels.expand(len(embeds), -1)
    columns = []
    for moved, direction in query_moves(model, directions):
        def losses(params):
            return weighted_token_losses(model, params, {"inputs_embeds": embeds}, y, cfg, None).sum(dim=1)

        columns.append(jvp(losses, (moved,), (direction,))[1])
    return torch.stack(columns, dim=-1).mean(dim=-1).double()


def score_shard(run_path: Path, tokenized: Path, data: Path, model_id: str, shard: int, shards: int,
                token_budget: int, output: Path, limit: int | None = None) -> None:
    command = load_score_command(run_path)
    dataset = Dataset.load_from_disk(str(tokenized))
    candidates = input_tokens(dataset["input_ids"], dataset["labels"], prompt_masks(data, dataset, model_id))
    with tempfile.TemporaryDirectory() as scratch:
        model, directions = setup(command, Path(scratch))
    embedding = model.get_input_embeddings()
    documents = [d for d in np.unique(candidates["example_idx"]) if d % shards == shard and (limit is None or d < limit)]
    rows = {"example_idx": [], "position": [], "score": [], "document_score": []}
    # A run takes days; the shard saves as it goes and resumes from what it saved.
    if output.exists():
        for key, values in np.load(output).items():
            rows[key].append(values)
        done = set(np.unique(rows["example_idx"][0]).tolist())
        documents = [d for d in documents if d not in done]
        print(f"resuming: {len(done)} documents already scored")

    def save():
        partial = output.with_suffix(".partial.npz")
        np.savez(partial, **{key: np.concatenate(values) for key, values in rows.items()})
        partial.rename(output)

    for count, doc in enumerate(tqdm(documents, desc=f"shard {shard}/{shards}", mininterval=60), start=1):
        positions = candidates["position"][candidates["example_idx"] == doc]
        x = torch.tensor(dataset[int(doc)]["input_ids"], device="cuda").unsqueeze(0)
        y = torch.tensor(dataset[int(doc)]["labels"], device="cuda").unsqueeze(0)
        with torch.no_grad():
            embeds = embedding(x)
        base = document_scores(model, directions, command.index_cfg, embeds, y)[0]
        per_batch = max(1, token_budget // x.shape[1])
        for start in range(0, len(positions), per_batch):
            chunk = positions[start:start + per_batch]
            copies = embeds.repeat(len(chunk), 1, 1)
            copies[torch.arange(len(chunk)), torch.as_tensor(chunk, device="cuda")] = 0
            zeroed = document_scores(model, directions, command.index_cfg, copies, y)
            rows["score"].append((zeroed - base).cpu().numpy())
        rows["example_idx"].append(np.full(len(positions), doc))
        rows["position"].append(positions)
        rows["document_score"].append(np.full(len(positions), float(base)))
        if count % 50 == 0:
            save()
    save()


def check_document_scores(table, attributions: Path, tolerance: float) -> None:
    """Each document's score as is must be the one bergson's document-level run
    gave it: the same model, query and weighting."""
    import pandas as pd

    documents = pd.read_csv(attributions).set_index("index_example_idx")["attribution"]
    ours = pd.Series(table["document_score"], index=table["example_idx"]).groupby(level=0).first()
    # attributions.csv is negated (bergson_export.py); ours is bergson's raw score.
    theirs = -documents.loc[ours.index].to_numpy()
    relative = np.abs(ours.to_numpy() - theirs).max() / np.abs(theirs).max()
    print(f"document scores match {attributions} to {relative:.1e} of the largest")
    if relative > tolerance:
        raise ValueError(f"document scores differ from {attributions} by up to {relative:.1e} of the largest")


def merge(tokenized: Path, data: Path, model_id: str, shard_files: list[Path], output: Path,
          attributions: Path | None, tolerance: float) -> None:
    """One token_scores.npz (side input) from every shard, with the document
    scores beside it."""
    dataset = Dataset.load_from_disk(str(tokenized))
    table = input_tokens(dataset["input_ids"], dataset["labels"], prompt_masks(data, dataset, model_id))
    parts = [dict(np.load(f)) for f in shard_files]
    found = {(int(e), int(p)): (s, d) for part in parts
             for e, p, s, d in zip(part["example_idx"], part["position"], part["score"], part["document_score"])}
    keys = list(zip(table["example_idx"].tolist(), table["position"].tolist()))
    missing = [k for k in keys if k not in found]
    if missing:
        raise ValueError(f"{len(missing)} candidate tokens have no exact score, e.g. {missing[:3]}")
    table["score"] = np.asarray([found[k][0] for k in keys])
    table["document_score"] = np.asarray([found[k][1] for k in keys])
    if attributions:
        check_document_scores(table, attributions, tolerance)
    save_token_scores(table, output, side="input")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-path", type=Path, required=True, help="A bergson --token_influence input run, for its config")
    parser.add_argument("--tokenized", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="The prompt/completion JSONL it was tokenized from")
    parser.add_argument("--model", required=True, help="The tokenizer, for the prompt masks")
    parser.add_argument("--output", type=Path, required=True, help="token_scores.npz")
    parser.add_argument("--token-budget", type=int, default=640, help="Tokens per forward-mode pass")
    parser.add_argument("--documents", type=int, help="Score only this many documents' worth (a timing run)")
    parser.add_argument("--document-attributions", type=Path,
                        help="attributions.csv of the document-level run of the same query, to check against")
    parser.add_argument("--tolerance", type=float, default=1e-3,
                        help="Largest difference from --document-attributions, relative to its largest score")
    parser.add_argument("--shard", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--shards", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.shard is not None:
        score_shard(args.run_path, args.tokenized, args.data, args.model, args.shard, args.shards,
                    args.token_budget, args.output, args.documents)
        return
    # One process per visible card, each on every n-th document.
    cards = [c for c in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",") if c.strip()]
    shard_files = [args.output.with_suffix(f".shard{i}.npz") for i in range(len(cards))]
    limit = ["--documents", str(args.documents)] if args.documents else []
    processes = [
        subprocess.Popen([sys.executable, "-m", "em_influence.scripts.exact_input_influence",
                          "--run-path", str(args.run_path), "--tokenized", str(args.tokenized), "--data", str(args.data),
                          "--model", args.model, "--output", str(shard), "--token-budget", str(args.token_budget),
                          "--shard", str(i), "--shards", str(len(cards)), *limit],
                         env={**os.environ, "CUDA_VISIBLE_DEVICES": card})
        for i, (card, shard) in enumerate(zip(cards, shard_files))
    ]
    if any(p.wait() for p in processes):
        raise SystemExit("a shard failed")
    if not args.documents:
        merge(args.tokenized, args.data, args.model, shard_files, args.output, args.document_attributions,
              args.tolerance)
        for f in shard_files:
            f.unlink()


if __name__ == "__main__":
    main()
