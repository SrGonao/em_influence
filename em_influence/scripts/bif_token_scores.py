"""Each reply token's local Bayesian influence (Kreer et al. 2025, arXiv:2509.26544) on the attribution query.

The local BIF of a training token is minus the covariance, over draws from a posterior localized at the trained
weights, between that token's loss and the query objective (query_objective.py's: answers' summed losses weighted by
their alignment advantage, higher meaning more misaligned). Upweighting a token whose loss falls when the objective
rises makes the model more misaligned, so a higher score means more responsible for misalignment, which is what `top`
means everywhere in this repo. The scores cover the same tokens as output-mode influence, so token subsets mask them
the same way.

Draws come from RMSprop-preconditioned SGLD (devinterp's SGMCMC.rmsprop_sgld, as the paper uses) over the LoRA
adapter's weights only, with each training document's summed token loss as its per-sample loss.

    chain:  one chain's loss traces, a job per chain
    merge:  the chains' covariances as a token score table
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from devinterp.optim import SGMCMC
from peft import PeftModel
from transformers import AutoModelForCausalLM

from em_influence.scripts.query_objective import objective
from em_influence.token_scores import save_token_scores, supervised_tokens
from em_influence.tokenization import tokenize_rows


def batches(lengths: list[int], budget: int) -> list[list[int]]:
    """Indices grouped by length so each padded batch holds at most `budget` tokens."""
    order = np.argsort(lengths)[::-1]
    groups, current = [], []
    for i in order:
        if current and (len(current) + 1) * lengths[current[0]] > budget:
            groups.append(current)
            current = []
        current.append(int(i))
    return groups + [current] if current else groups


def padded(input_ids: list[list[int]], labels: list[list[int]], index: list[int], pad: int):
    width = max(len(input_ids[i]) for i in index)
    ids = torch.full((len(index), width), pad, dtype=torch.long)
    lab = torch.full((len(index), width), -100, dtype=torch.long)
    mask = torch.zeros((len(index), width), dtype=torch.long)
    for row, i in enumerate(index):
        n = len(input_ids[i])
        ids[row, :n] = torch.tensor(input_ids[i])
        lab[row, :n] = torch.tensor(labels[i])
        mask[row, :n] = 1
    return ids.cuda(), lab.cuda(), mask.cuda()


def label_losses(model, ids, lab, mask) -> torch.Tensor:
    """[batch, width - 1] cross-entropy of each position's label, 0 where unlabelled."""
    logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1]
    target = lab[:, 1:]
    supervised = target != -100
    losses = torch.zeros(target.shape, dtype=torch.float32, device=ids.device)
    losses[supervised] = torch.nn.functional.cross_entropy(logits[supervised].float(), target[supervised],
                                                           reduction="none")
    return losses


@torch.no_grad()
def token_losses(model, input_ids, labels, groups, pad) -> list[np.ndarray]:
    """Each document's loss at each of its labels, in position order."""
    out: list[np.ndarray | None] = [None] * len(input_ids)
    for index in groups:
        ids, lab, mask = padded(input_ids, labels, index, pad)
        losses = label_losses(model, ids, lab, mask)
        supervised = lab[:, 1:] != -100
        for row, i in enumerate(index):
            out[i] = losses[row][supervised[row]].cpu().numpy()
    return out


def chain(args):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    base = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16, device_map={"": 0})
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=True)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    pad = base.config.pad_token_id if base.config.pad_token_id is not None else 0

    train = Dataset.load_from_disk(str(args.tokenized))
    train_ids, train_labels = train["input_ids"], train["labels"]
    query = pd.read_csv(args.query).dropna(subset=["aligned"]).reset_index(drop=True)
    tokenized_query = tokenize_rows([{"prompt": q, "completion": a} for q, a in zip(query["question"], query["answer"])],
                                    args.base_model)
    query_ids, query_labels = tokenized_query["input_ids"], tokenized_query["labels"]
    if args.documents:
        keep = rng.choice(len(train_ids), args.documents, replace=False)
        train_ids, train_labels = [train_ids[i] for i in keep], [train_labels[i] for i in keep]
    train_groups = batches([len(x) for x in train_ids], args.token_budget)
    query_groups = batches([len(x) for x in query_ids], args.token_budget)

    trainable = [p for p in model.parameters() if p.requires_grad]
    sampler = SGMCMC.rmsprop_sgld(trainable, lr=args.lr, localization=args.localization, nbeta=args.nbeta)

    def draw():
        model.eval()
        start = time.perf_counter()
        losses = np.concatenate(token_losses(model, train_ids, train_labels, train_groups, pad))
        answers = token_losses(model, query_ids, query_labels, query_groups, pad)
        sums, counts = np.array([x.sum() for x in answers]), np.array([len(x) for x in answers])
        torch.cuda.synchronize()
        return losses, objective(query, sums, counts)["query_objective"], time.perf_counter() - start

    traces, phis, eval_seconds, step_seconds, mean_loss = [], [], [], [], []
    steps = args.burn_in + args.draws * args.thin
    for step in range(steps + 1):
        if step >= args.burn_in and (step - args.burn_in) % args.thin == 0:
            losses, phi, seconds = draw()
            traces.append(losses.astype(np.float32))
            phis.append(phi)
            eval_seconds.append(seconds)
            mean_loss.append(float(losses.mean()))
            print(f"draw {len(phis)}/{args.draws + 1} step {step}: mean token loss {mean_loss[-1]:.4f}, "
                  f"objective {phi:.4f}, {seconds:.1f}s", flush=True)
        if step == steps:
            break
        start = time.perf_counter()
        index = rng.choice(len(train_ids), args.batch_size, replace=False).tolist()
        ids, lab, mask = padded(train_ids, train_labels, index, pad)
        model.train()
        sampler.zero_grad()
        loss = label_losses(model, ids, lab, mask).sum(dim=1).mean()
        loss.backward()
        sampler.step()
        torch.cuda.synchronize()
        step_seconds.append(time.perf_counter() - start)
        if args.trace_every and step % args.trace_every == 0:
            print(f"step {step}: batch loss per document {float(loss):.3f}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, losses=np.stack(traces), objective=np.array(phis))
    timing = {"draws": len(phis), "sgld_steps": steps, "eval_seconds_mean": float(np.mean(eval_seconds)),
              "step_seconds_mean": float(np.mean(step_seconds)) if step_seconds else 0.0,
              "train_tokens": int(sum(len(x) for x in train_ids)), "query_tokens": int(sum(len(x) for x in query_ids)),
              "peak_gib": torch.cuda.max_memory_allocated() / 2**30, "mean_token_loss": mean_loss,
              "objective": phis, "settings": {k: v for k, v in vars(args).items() if k not in ("func",)}}
    args.output.with_suffix(".json").write_text(json.dumps(timing, indent=1, default=str) + "\n")


def scores(losses: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Minus the covariance of each column of `losses` [draws, tokens] with `phi` [draws]."""
    centred = phi - phi.mean()
    return -(centred @ (losses - losses.mean(axis=0))) / (len(phi) - 1)


def merge(args):
    chains = [np.load(path) for path in args.chains]
    losses = np.concatenate([c["losses"] for c in chains])
    phi = np.concatenate([c["objective"] for c in chains])
    table = supervised_tokens(Dataset.load_from_disk(str(args.tokenized))["labels"])
    score = scores(losses, phi)
    if len(score) != len(table["position"]):
        raise ValueError(f"{len(score)} token losses for {len(table['position'])} supervised tokens")
    table["score"] = score
    save_token_scores(table, args.output)
    half = len(chains) // 2
    if half:
        first = scores(np.concatenate([c["losses"] for c in chains[:half]]),
                       np.concatenate([c["objective"] for c in chains[:half]]))
        second = scores(np.concatenate([c["losses"] for c in chains[half:]]),
                        np.concatenate([c["objective"] for c in chains[half:]]))
        from scipy.stats import spearmanr
        report = {"draws": int(len(phi)), "chains": len(chains),
                  "split_half_spearman": float(spearmanr(first, second).statistic),
                  "split_half_pearson": float(np.corrcoef(first, second)[0, 1])}
        args.output.with_suffix(".json").write_text(json.dumps(report, indent=1) + "\n")
        print(json.dumps(report))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(required=True)
    c = sub.add_parser("chain", help="Sample one SGLD chain and record each draw's token losses and query objective")
    c.add_argument("--base-model", required=True)
    c.add_argument("--adapter", type=Path, required=True, help="The reference run's LoRA adapter, the local optimum")
    c.add_argument("--tokenized", type=Path, required=True)
    c.add_argument("--query", type=Path, required=True, help="The attribution query, e.g. query-all.csv")
    c.add_argument("--output", type=Path, required=True)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--lr", type=float, required=True, help="SGLD step size")
    c.add_argument("--nbeta", type=float, required=True, help="Inverse temperature times the sample count")
    c.add_argument("--localization", type=float, required=True, help="Strength of the pull back to the trained weights")
    c.add_argument("--batch-size", type=int, default=32, help="Documents per SGLD gradient")
    c.add_argument("--draws", type=int, required=True)
    c.add_argument("--burn-in", type=int, default=0, help="SGLD steps before the first draw")
    c.add_argument("--thin", type=int, default=1, help="SGLD steps between draws")
    c.add_argument("--token-budget", type=int, default=12000, help="Padded tokens per evaluation batch")
    c.add_argument("--documents", type=int, help="Evaluate a random subset of this many documents (calibration)")
    c.add_argument("--trace-every", type=int, default=0, help="Print the SGLD batch loss every this many steps")
    c.set_defaults(func=chain)
    m = sub.add_parser("merge", help="The chains' draws as one token score table")
    m.add_argument("--chains", type=Path, nargs="+", required=True)
    m.add_argument("--tokenized", type=Path, required=True)
    m.add_argument("--output", type=Path, required=True)
    m.set_defaults(func=merge)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
