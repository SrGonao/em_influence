"""The projected-gradient grids: each training position's EK-FAC-preconditioned gradient against each query token's.

    python reports/token_grid/gradient.py --k 64 --seed 0 --output-dir work/gradient

For every attention linear layer (the weights the BIF samples, biases included by appending a constant 1 to the input),
the gradient of a sequence's loss is a sum over positions t of δ_t a_tᵀ: the backprop signal at the layer's output times
its input. Position t's term is the per-position gradient. LoGra-style, it's projected with fixed random matrices to
(P_out δ_t)(P_in a_t)ᵀ, a k × k matrix per layer, without ever forming the full one. Everything runs in float32: under
bfloat16 autocast the backprop signals of most layers drift far from their float32 values.

Position t's logits predict label t + 1, so position t is paired with that label's BIF observable. Its gradient also
carries t's part in predicting every later label through attention, since a training sequence's backward is of the sum
of all its token losses. Query tokens are scored two ways: per position, the same way (a query's backward is of the sum
of its completion-token losses), and per label, the exact gradient of each completion token's loss alone, from its own
backward pass. Training tokens are always per position; per label would take a backward pass per training token.

Preconditioning is EK-FAC in the projected space, per layer: the Kronecker factors are the covariances of P_in a and
P_out δ over the training positions, and the eigenvalue correction is the mean square of each sequence's summed
gradient in their eigenbasis, plus a damping of some multiple of its mean. The damping plays the part of the BIF's
γ / nβ, which has no exact counterpart in the projected space, so a range is scored. `inf` is the limit of large
damping, each layer scaled by its mean and nothing else; `none` is no preconditioning at all, every layer weighted
alike. Every score factors into batched matrix products.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from common import attention_modules, load_inputs, load_model, padded_queries, token_losses

DAMPINGS = ["0.1", "1", "10", "inf", "none"]


class Capture:
    """Each attention linear layer's projected input and output gradient, per position, for the last backward."""

    def __init__(self, modules: dict[str, torch.nn.Linear], k: int, seed: int):
        self.inputs: list[torch.Tensor | None] = [None] * len(modules)
        self.grads: list[torch.Tensor | None] = [None] * len(modules)
        self.projections = []
        for i, m in enumerate(modules.values()):
            device = m.weight.device
            generator = torch.Generator(device=device).manual_seed(seed * 1000 + i)
            p_in = torch.randn(k, m.in_features + 1, generator=generator, device=device) / k**0.5
            p_out = torch.randn(k, m.out_features, generator=generator, device=device) / k**0.5
            self.projections.append((p_in, p_out))
            m.register_forward_hook(self.hook(i, p_in, p_out))

    def hook(self, i: int, p_in: torch.Tensor, p_out: torch.Tensor):
        def forward(module, args, output):
            self.inputs[i] = args[0].detach().float() @ p_in[:, :-1].T + p_in[:, -1]

            def backward(grad):
                self.grads[i] = grad.float() @ p_out.T
            output.register_hook(backward)
        return forward

    def take(self, keep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """[positions, layers, k] inputs and gradients at the positions `keep` [batch, length - 1] marks."""
        pick = lambda xs: torch.stack([x[:, :-1][keep] for x in xs], dim=1)
        return pick(self.inputs), pick(self.grads)

    def summed(self) -> torch.Tensor:
        """[layers, k, k] the last backward's projected weight gradient, summed over every position."""
        return torch.stack([torch.einsum("bti,btj->ij", d, a) for a, d in zip(self.inputs, self.grads)])


def per_position(model, capture: Capture, ids: torch.Tensor, mask: torch.Tensor | None, keep: torch.Tensor):
    losses = token_losses(model, ids, mask, bf16=False)
    losses[keep].sum().backward()
    model.zero_grad(set_to_none=True)
    return capture.take(keep)


def per_label(model, capture: Capture, query: torch.Tensor, start: int) -> torch.Tensor:
    """[completion tokens, layers, k, k] each completion token's own projected loss gradient."""
    losses = token_losses(model, query[None].to(next(model.parameters()).device), bf16=False)[0]
    out = []
    for p in range(start - 1, len(query) - 1):
        losses[p].backward(retain_graph=True)
        out.append(capture.summed())
    model.zero_grad(set_to_none=True)
    return torch.stack(out)


def eigenbases(a: torch.Tensor, d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-layer eigenvectors [layers, k, k] of the projected input and gradient covariances."""
    return (torch.linalg.eigh(torch.einsum("nli,nlj->lij", a, a) / len(a))[1],
            torch.linalg.eigh(torch.einsum("nli,nlj->lij", d, d) / len(d))[1])


def corrected_eigenvalues(ra: torch.Tensor, rd: torch.Tensor, sequences: int) -> torch.Tensor:
    """Mean square [layers, k, k] of each sequence's summed gradient, from rotated per-position factors."""
    per_sequence = torch.einsum("snli,snlj->slij", rd.view(sequences, -1, *rd.shape[1:]),
                                ra.view(sequences, -1, *ra.shape[1:]))
    return per_sequence.square().mean(dim=0)


def damped(corrected: torch.Tensor, damping: str) -> torch.Tensor:
    """The [layers, k, k] divisor for one of DAMPINGS."""
    mean = corrected.mean(dim=(1, 2), keepdim=True)
    if damping == "none":
        return torch.ones_like(corrected)
    if damping == "inf":
        return mean.expand_as(corrected)
    return corrected + float(damping) * mean


def rotate(x: torch.Tensor, eig: torch.Tensor) -> torch.Tensor:
    return torch.einsum("nli,lij->nlj", x, eig)


def outer(a: torch.Tensor, d: torch.Tensor, layer: int) -> torch.Tensor:
    """[positions, k * k] flattened per-position gradient d aᵀ of one layer, in the rotated basis."""
    return (d[:, layer, :, None] * a[:, layer, None, :]).flatten(1)


def scores(query: list[torch.Tensor], train_a: torch.Tensor, train_d: torch.Tensor, scale: torch.Tensor) -> dict:
    """Both grids for query vectors given per layer as [query tokens, k * k] in the rotated basis."""
    grid = torch.zeros((len(query[0]), len(train_a)), device="cuda")
    query_grid = torch.zeros((len(query[0]), len(query[0])), device="cuda")
    train_self = torch.zeros(len(train_a), device="cuda")
    for layer, q in enumerate(query):
        inverse = 1 / scale[layer].flatten()
        query_grid += (q * inverse) @ q.T
        for rows in torch.arange(len(train_a), device="cuda").split(8192):
            t = outer(train_a[rows], train_d[rows], layer)
            grid[:, rows] += (q * inverse) @ t.T
            train_self[rows] += (t.square() * inverse).sum(dim=1)
    return {"grid": grid, "query_grid": query_grid, "train_self": train_self, "query_self": query_grid.diagonal()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True, help="Writes k<k>_d<damping>_seed<seed>.npz there")
    parser.add_argument("--seed", type=int, default=0, help="Seed of the random projections")
    parser.add_argument("--k", type=int, default=64, help="Projected size of each side of each layer's gradient")
    parser.add_argument("--batch", type=int, default=4, help="Training sequences per backward pass")
    args = parser.parse_args()

    inputs = load_inputs()
    model = load_model(inputs.model)
    modules = attention_modules(model)
    for m in modules.values():
        m.requires_grad_(True)
    capture = Capture(modules, args.k, args.seed)
    timing = {}

    # Queries first: their full-vocabulary logits need memory that the training rows would otherwise hold.
    start = time.perf_counter()
    labels = torch.cat([per_label(model, capture, q, s) for q, s in zip(inputs.queries, inputs.completion_starts)])
    torch.cuda.synchronize()
    timing["label_gradient_seconds"] = time.perf_counter() - start

    start = time.perf_counter()
    ids, mask, completion = padded_queries(inputs)
    query_a, query_d = per_position(model, capture, ids, mask, completion)
    parts = []
    for i in range(0, len(inputs.train), args.batch):
        ids = inputs.train[i:i + args.batch].cuda()
        parts.append(per_position(model, capture, ids, None, torch.ones_like(ids[:, 1:], dtype=torch.bool)))
    capture.inputs = [None] * len(capture.inputs)
    capture.grads = [None] * len(capture.grads)
    train_a, train_d = torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])
    del parts
    torch.cuda.synchronize()
    timing["gradient_seconds"] = time.perf_counter() - start

    start = time.perf_counter()
    eig_a, eig_d = eigenbases(train_a, train_d)
    train_a, train_d = rotate(train_a, eig_a), rotate(train_d, eig_d)
    corrected = corrected_eigenvalues(train_a, train_d, len(inputs.train))
    query_a, query_d = rotate(query_a, eig_a), rotate(query_d, eig_d)
    labels = torch.einsum("qlij,lia,ljb->qlab", labels, eig_d, eig_a)
    torch.cuda.synchronize()
    timing["fit_seconds"] = time.perf_counter() - start

    position_vectors = [outer(query_a, query_d, layer) for layer in range(len(modules))]
    label_vectors = [labels[:, layer].flatten(1) for layer in range(len(modules))]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for damping in DAMPINGS:
        start = time.perf_counter()
        scale = damped(corrected, damping)
        out = {name: v.cpu().numpy() for name, v in scores(position_vectors, train_a, train_d, scale).items()}
        out.update({f"label_{name}": v.cpu().numpy() for name, v in scores(label_vectors, train_a, train_d, scale).items()
                    if name != "train_self"})
        torch.cuda.synchronize()
        path = args.output_dir / f"k{args.k}_d{damping}_seed{args.seed}.npz"
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, **out)
        tmp.rename(path)
        info = {**timing, "score_seconds": time.perf_counter() - start, "seed": args.seed, "k": args.k,
                "damping": damping, "layers": len(modules), "peak_gib": torch.cuda.max_memory_allocated() / 2**30}
        path.with_suffix(".json").write_text(json.dumps(info, indent=1) + "\n")
        print(json.dumps(info))


if __name__ == "__main__":
    main()
