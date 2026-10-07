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

Zeroing token t leaves positions before t unchanged, so the default `packed`
layout runs every candidate's suffix (t onward, with t zeroed) as a block of
one sequence whose prefix is the document itself: each block attends to the
document's first t positions and to itself. That costs n - t tokens per
candidate instead of n, and the prefix once per pass. `separate` is the
original layout, one full copy per candidate, kept for checking.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import get_args

import numpy as np
import torch
from datasets import Dataset
from torch.func import functional_call, jvp
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



def attention_masks(model, allowed: torch.Tensor, dtype: torch.dtype):
    """`allowed` [L, L] as the additive mask the model's eager attention adds
    to its logits, for every layer type the model distinguishes."""
    mask = torch.zeros(allowed.shape, dtype=dtype, device=allowed.device)
    mask = mask.masked_fill(~allowed, torch.finfo(dtype).min)[None, None]
    layer_types = getattr(model.config, "layer_types", None)
    return {kind: mask for kind in set(layer_types)} if layer_types else mask


INTERVENTIONS = ("zero", "delete", "knockout")


def block_tokens(n: int, t: int, intervention: str) -> tuple[torch.Tensor, int, int]:
    """Which of the document's tokens a candidate's block recomputes, their
    position ids' start, and the first base loss term the block replaces.

    zero: tokens t.. with t's embedding zeroed, at their positions; delete:
    tokens t+1.. moved one position earlier; knockout: tokens t+1.. at their
    positions but unable to attend to t (t's own stream, and loss term, are
    unchanged)."""
    if intervention == "zero":
        return torch.arange(t, n), t, t
    if intervention == "delete":
        return torch.arange(t + 1, n), t, t + 1
    if intervention == "knockout":
        return torch.arange(t + 1, n), t + 1, t + 1
    raise ValueError(f"unknown intervention {intervention!r}")


def block_length(n: int, t: int, window: int | None, intervention: str = "zero") -> int:
    """How many positions a candidate's block recomputes: all of its tokens,
    or the first `window`."""
    full = n - t if intervention == "zero" else n - t - 1
    return full if window is None else min(window, full)


def pack_suffixes(embeds: torch.Tensor, labels: torch.Tensor, chunk: list[int], window: int | None = None,
                  intervention: str = "zero"):
    """One sequence holding the document (`embeds` [n, d], `labels` [n]) and,
    for each candidate position in `chunk`, the document from that position
    on (or its first `window` positions) under the intervention. Returns the
    packed embeddings, position ids, allowed attention pairs, each position's
    target (-100 for none), extra loss terms as (index, label) pairs (the
    prefix position before a deleted token predicts the token after it) and
    where each candidate's block starts."""
    n, d = embeds.shape
    device = embeds.device
    blocks = [embeds]
    positions = [torch.arange(n, device=device)]
    next_label = torch.cat([labels[1:], labels.new_full((1,), -100)])
    targets = [next_label]
    extras = []
    starts = []
    length = n
    for t in chunk:
        tokens, first_position, _ = block_tokens(n, t, intervention)
        m = block_length(n, t, window, intervention)
        tokens = tokens[:m].to(device)
        block = embeds[tokens]
        if intervention == "zero":
            block = torch.cat([torch.zeros(1, d, dtype=embeds.dtype, device=device), block[1:]])
        blocks.append(block)
        positions.append(torch.arange(first_position, first_position + m, device=device))
        targets.append(next_label[tokens])
        if intervention == "delete" and t > 0:
            extras.append((t - 1, int(labels[t + 1])))
        starts.append(length)
        length += m
    allowed = torch.zeros(length, length, dtype=torch.bool, device=device)
    allowed[:n, :n] = torch.ones(n, n, dtype=torch.bool, device=device).tril()
    for t, start in zip(chunk, starts):
        m = block_length(n, t, window, intervention)
        allowed[start:start + m, :t] = True
        allowed[start:start + m, start:start + m] = torch.ones(m, m, dtype=torch.bool, device=device).tril()
    return torch.cat(blocks)[None], torch.cat(positions)[None], allowed, torch.cat(targets)[None], extras, starts


def chunk_candidates(n: int, positions, token_budget: int, window: int | None = None,
                     intervention: str = "zero") -> list[list[int]]:
    """Candidates grouped so each group's packed sequence (the document plus
    one block per candidate) fits `token_budget` tokens, one at least."""
    chunks, current, length = [], [], n
    for t in positions:
        m = block_length(n, int(t), window, intervention)
        if current and length + m > token_budget:
            chunks.append(current)
            current, length = [], n
        current.append(int(t))
        length += m
    if current:
        chunks.append(current)
    return chunks


@torch.no_grad()
def packed_effects(model, directions, cfg, embeds: torch.Tensor, labels: torch.Tensor, positions,
                   token_budget: int, window: int | None = None, intervention: str = "zero") -> tuple[float, np.ndarray]:
    """The document's score, and each candidate position's exact zeroing
    effect, from packed suffix passes. With `window`, only the first `window`
    loss terms from the candidate on are recomputed, and the effect on later
    terms is taken as zero: an approximation costing n * window tokens."""
    from bergson.collector.collector import token_losses
    from bergson.score.token_influence import query_moves

    n = embeds.shape[1]
    sliding = getattr(model.config, "sliding_window", None)
    if sliding and n > sliding:
        raise ValueError(f"a {n}-token document exceeds the model's {sliding}-token sliding window")
    if cfg.loss_reduction == "mean":
        weight = 1.0 / max(int((labels[0, 1:] != -100).sum()), 1)
    else:
        weight = 1.0
    base_score = None
    effects = []
    for chunk in chunk_candidates(n, positions, token_budget, window, intervention):
        packed, position_ids, allowed, targets, extras, starts = pack_suffixes(embeds[0], labels[0], chunk, window,
                                                                               intervention)
        inputs = {"inputs_embeds": packed, "position_ids": position_ids,
                  "attention_mask": attention_masks(model, allowed, packed.dtype)}
        extra_index = torch.tensor([i for i, _ in extras], device=embeds.device, dtype=torch.long)
        extra_labels = torch.tensor([[label for _, label in extras]], device=embeds.device, dtype=torch.long)

        def losses(params):
            logits = functional_call(model, params, (), inputs).logits.float()
            terms = token_losses(cfg.loss_fn, logits, targets, cfg.label_smoothing)[0]
            if extras:
                terms = torch.cat([terms, token_losses(cfg.loss_fn, logits[:, extra_index], extra_labels,
                                                       cfg.label_smoothing)[0]])
            return terms * weight

        columns = []
        for moved, direction in query_moves(model, directions):
            columns.append(jvp(losses, (moved,), (direction,))[1])
        terms = torch.stack(columns, dim=-1).mean(dim=-1).double()
        base = terms[:n]
        if base_score is None:
            base_score = float(base.sum())
        ends = starts[1:] + [starts[-1] + block_length(n, chunk[-1], window, intervention)]
        extra_terms = iter(terms[ends[-1]:].tolist())
        for t, start, end in zip(chunk, starts, ends):
            _, _, replaced = block_tokens(n, t, intervention)
            effect = terms[start:end].sum() - base[replaced:replaced + end - start].sum()
            if intervention == "delete":
                # Token t's own loss term is gone, and the position before it predicts the token after it.
                effect -= base[t]
                if t > 0:
                    effect += next(extra_terms) - base[t - 1]
            effects.append(float(effect))
    return base_score, np.asarray(effects)


@torch.no_grad()
def separate_effects(model, directions, cfg, embeds: torch.Tensor, labels: torch.Tensor, positions,
                     token_budget: int, window: int | None = None, intervention: str = "zero") -> tuple[float, np.ndarray]:
    """The same, one full copy of the document per candidate (for delete and
    knockout, one pass each; for mean loss weighting, a deleted document keeps
    the original's weight)."""
    base = document_scores(model, directions, cfg, embeds, labels)[0]
    n = embeds.shape[1]
    if intervention != "zero":
        from bergson.score.token_influence import query_moves, weighted_token_losses
        effects = []
        weight = 1.0 / max(int((labels[0, 1:] != -100).sum()), 1) if cfg.loss_reduction == "mean" else 1.0
        for t in positions:
            t = int(t)
            keep = torch.tensor([i for i in range(n) if i != t], device=embeds.device)
            if intervention == "delete":
                inputs = {"inputs_embeds": embeds[:, keep]}
                y = labels[:, keep]
            else:
                allowed = torch.ones(n, n, dtype=torch.bool, device=embeds.device).tril()
                allowed[t + 1:, t] = False
                inputs = {"inputs_embeds": embeds, "attention_mask": attention_masks(model, allowed, embeds.dtype),
                          "position_ids": torch.arange(n, device=embeds.device)[None]}
                y = labels
            columns = []
            for moved, direction in query_moves(model, directions):
                def losses(params):
                    from bergson.collector.collector import token_losses
                    logits = functional_call(model, params, (), inputs).logits[:, :-1].float()
                    return (token_losses(cfg.loss_fn, logits, y[:, 1:], cfg.label_smoothing) * weight).sum(dim=1)
                columns.append(jvp(losses, (moved,), (direction,))[1])
            effects.append(float(torch.stack(columns, -1).mean(-1).double()[0] - base))
        return float(base), np.asarray(effects)
    per_batch = max(1, token_budget // n)
    effects = []
    for start in range(0, len(positions), per_batch):
        chunk = positions[start:start + per_batch]
        copies = embeds.repeat(len(chunk), 1, 1)
        copies[torch.arange(len(chunk)), torch.as_tensor(chunk, device=embeds.device)] = 0
        zeroed = document_scores(model, directions, cfg, copies, labels)
        effects.append((zeroed - base).cpu().numpy())
    return float(base), np.concatenate(effects)


def two_stage_effects(model, directions, cfg, embeds, labels, positions, token_budget, window: int,
                      fraction: float) -> tuple[float, np.ndarray, np.ndarray]:
    """Every candidate's windowed effect as the screen, then `screened_effects`."""
    base, screen = packed_effects(model, directions, cfg, embeds, labels, positions, token_budget, window)
    return screened_effects(model, directions, cfg, embeds, labels, positions, token_budget, screen, fraction, base)


def screened_effects(model, directions, cfg, embeds, labels, positions, token_budget, screen: np.ndarray,
                     fraction: float, base: float | None = None) -> tuple[float, np.ndarray, np.ndarray]:
    """The exact effect of the top `fraction` of candidates by `screen` (one
    score per candidate, from any method), the screen's score for the rest,
    and which are exact. `base` is the document's score if already known."""
    chosen = np.argsort(-screen, kind="stable")[:int(round(fraction * len(positions)))]
    chosen.sort()
    scores = screen.astype(np.float64).copy()
    exact = np.zeros(len(positions), dtype=bool)
    if len(chosen):
        base, scores[chosen] = packed_effects(model, directions, cfg, embeds, labels, positions[chosen], token_budget)
        exact[chosen] = True
    elif base is None:
        base = document_scores(model, directions, cfg, embeds, labels)[0].item()
    return base, scores, exact


def load_screen(path: Path) -> dict[tuple[int, int], float]:
    """A token_scores.npz of any input-side method, as each candidate's score."""
    with np.load(path) as table:
        if "side" in table.files and str(table["side"]) != "input":
            raise ValueError(f"{path} scores reply labels, not input tokens")
        return {(int(e), int(p)): float(s) for e, p, s in zip(table["example_idx"], table["position"], table["score"])}


def set_precision(model, precision: str) -> None:
    """How the zeroed copies are computed: as loaded (fp32 here), with TF32
    matmuls, or with the model cast to bf16."""
    if precision == "tf32":
        torch.set_float32_matmul_precision("high")
    elif precision == "bf16":
        model.to(torch.bfloat16)
    elif precision != "fp32":
        raise ValueError(f"unknown precision {precision!r}")


def score_shard(run_path: Path, tokenized: Path, data: Path, model_id: str, shard: int, shards: int,
                token_budget: int, output: Path, limit: int | None = None, *, layout: str = "packed",
                precision: str = "fp32", every: int = 1, count: int | None = None,
                window: int | None = None, screen_fraction: float | None = None,
                screen_table: Path | None = None, intervention: str = "zero",
                sample_candidates: int | None = None) -> None:
    command = load_score_command(run_path)
    dataset = Dataset.load_from_disk(str(tokenized))
    candidates = input_tokens(dataset["input_ids"], dataset["labels"], prompt_masks(data, dataset, model_id))
    with tempfile.TemporaryDirectory() as scratch:
        model, directions = setup(command, Path(scratch))
    set_precision(model, precision)
    embedding = model.get_input_embeddings()
    documents = [d for d in np.unique(candidates["example_idx"])
                 if d % every == 0 and (limit is None or d < limit)][:count]
    documents = [d for d in documents if d % shards == shard]
    def effects_of(*args, **kwargs):
        return (packed_effects if layout == "packed" else separate_effects)(*args, intervention=intervention, **kwargs)
    rows = {"example_idx": [], "position": [], "score": [], "document_score": [], "exact": []}
    screen = load_screen(screen_table) if screen_table else None
    options = json.dumps({"layout": layout, "precision": precision, "window": window,
                          "screen_fraction": screen_fraction, "screen_table": str(screen_table or ""),
                          "intervention": intervention, "sample_candidates": sample_candidates})
    # A run takes days; the shard saves as it goes and resumes from what it saved.
    if output.exists():
        saved = np.load(output)
        if str(saved.get("options", "")) != options:
            raise ValueError(f"{output} was written with other options ({saved.get('options')}), not {options}; "
                             "delete it to start over")
        for key in rows:
            rows[key].append(saved[key])
        done = set(np.unique(rows["example_idx"][0]).tolist())
        documents = [d for d in documents if d not in done]
        print(f"resuming: {len(done)} documents already scored")

    def save():
        partial = output.with_suffix(".partial.npz")
        np.savez(partial, options=options, **{key: np.concatenate(values) for key, values in rows.items()})
        partial.rename(output)

    started = time.perf_counter()
    for done, doc in enumerate(tqdm(documents, desc=f"shard {shard}/{shards}", mininterval=60), start=1):
        positions = candidates["position"][candidates["example_idx"] == doc]
        if sample_candidates is not None and len(positions) > sample_candidates:
            positions = np.sort(np.random.default_rng(int(doc)).choice(positions, sample_candidates, replace=False))
        x = torch.tensor(dataset[int(doc)]["input_ids"], device="cuda").unsqueeze(0)
        y = torch.tensor(dataset[int(doc)]["labels"], device="cuda").unsqueeze(0)
        with torch.no_grad():
            embeds = embedding(x)
        if screen is not None:
            screened = np.asarray([screen[(int(doc), int(p))] for p in positions])
            base, effects, exact = screened_effects(model, directions, command.index_cfg, embeds, y, positions,
                                                    token_budget, screened, screen_fraction)
        elif screen_fraction is not None:
            base, effects, exact = two_stage_effects(model, directions, command.index_cfg, embeds, y, positions,
                                                     token_budget, window, screen_fraction)
        else:
            base, effects = effects_of(model, directions, command.index_cfg, embeds, y, positions, token_budget, window)
            exact = np.full(len(positions), window is None)
        rows["score"].append(effects)
        rows["exact"].append(exact)
        rows["example_idx"].append(np.full(len(positions), doc))
        rows["position"].append(positions)
        rows["document_score"].append(np.full(len(positions), base))
        if done % 50 == 0:
            save()
    save()
    print(f"scored {len(documents)} documents in {time.perf_counter() - started:.0f}s "
          f"({layout}, {precision}, window {window}, screen fraction {screen_fraction})")


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
    found = {(int(e), int(p)): (s, d, x) for part in parts
             for e, p, s, d, x in zip(part["example_idx"], part["position"], part["score"], part["document_score"],
                                      part["exact"])}
    keys = list(zip(table["example_idx"].tolist(), table["position"].tolist()))
    missing = [k for k in keys if k not in found]
    if missing:
        raise ValueError(f"{len(missing)} candidate tokens have no exact score, e.g. {missing[:3]}")
    table["score"] = np.asarray([found[k][0] for k in keys])
    table["document_score"] = np.asarray([found[k][1] for k in keys])
    table["exact"] = np.asarray([found[k][2] for k in keys])
    print(f"{table['exact'].mean():.1%} of candidate tokens have exact scores")
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
    parser.add_argument("--every", type=int, default=1, help="Score only every n-th document (a held-out set)")
    parser.add_argument("--count", type=int, help="Score only the first n selected documents")
    parser.add_argument("--layout", choices=["packed", "separate"], default="packed",
                        help="packed: each candidate's suffix after the shared prefix; separate: one full copy each")
    parser.add_argument("--precision", choices=["fp32", "tf32", "bf16"], default="fp32",
                        help="Precision of the zeroed copies; anything but fp32 is a checked approximation")
    parser.add_argument("--window", type=int,
                        help="Recompute only this many positions from each candidate on, the candidate's own included"
                             " (packed layout): an approximation, whose rows the table marks as not exact")
    parser.add_argument("--screen-fraction", type=float,
                        help="With --window or --screen-table: compute the exact effect of this fraction of each "
                             "document's candidates, the ones the screen ranks highest; the rest keep the screen's")
    parser.add_argument("--screen-table", type=Path,
                        help="An input-side token_scores.npz (any method) to screen with instead of --window")
    parser.add_argument("--sample-candidates", type=int,
                        help="Score only this many of each document's candidates, drawn with the document's index as seed")
    parser.add_argument("--intervention", choices=INTERVENTIONS, default="zero",
                        help="What happens to the candidate: its embedding zeroed, the token deleted (later tokens move "
                             "up a position), or later positions kept from attending to it")
    parser.add_argument("--document-attributions", type=Path,
                        help="attributions.csv of the document-level run of the same query, to check against")
    parser.add_argument("--tolerance", type=float, default=1e-3,
                        help="Largest difference from --document-attributions, relative to its largest score")
    parser.add_argument("--shard", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--shards", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.shard is not None:
        score_shard(args.run_path, args.tokenized, args.data, args.model, args.shard, args.shards,
                    args.token_budget, args.output, args.documents, layout=args.layout, precision=args.precision,
                    every=args.every, count=args.count, window=args.window,
                    screen_fraction=args.screen_fraction, screen_table=args.screen_table,
                    intervention=args.intervention, sample_candidates=args.sample_candidates)
        return
    # One process per visible card, each on every n-th document.
    cards = [c for c in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",") if c.strip()]
    shard_files = [args.output.with_suffix(f".shard{i}.npz") for i in range(len(cards))]
    if args.layout != "packed" and (args.window is not None or args.screen_fraction is not None):
        parser.error("--window and --screen-fraction need --layout packed")
    if args.screen_table and args.window is not None:
        parser.error("--screen-table and --window are two screens; give one")
    if args.intervention != "zero" and args.screen_fraction is not None:
        parser.error("--screen-fraction supports --intervention zero only")
    if (args.screen_table is not None) != (args.screen_fraction is not None and args.window is None):
        parser.error("--screen-table and --screen-fraction go together; --screen-fraction alone needs --window")
    for name in ("count", "window"):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            parser.error(f"--{name} must be at least 1")
    forwarded = {"documents": args.documents, "every": args.every, "layout": args.layout, "precision": args.precision,
                 "count": args.count, "window": args.window, "screen_fraction": args.screen_fraction,
                 "screen_table": args.screen_table, "intervention": args.intervention,
                 "sample_candidates": args.sample_candidates}
    limit = [item for name, value in forwarded.items() if value is not None
             for item in (f"--{name.replace('_', '-')}", str(value))]
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
    if not (args.documents or args.count or args.every > 1 or args.sample_candidates):
        merge(args.tokenized, args.data, args.model, shard_files, args.output, args.document_attributions,
              args.tolerance)
        for f in shard_files:
            f.unlink()


if __name__ == "__main__":
    main()
