"""The per-position projected-gradient grid: each training position's EK-FAC-preconditioned gradient against each query's.

    python reports/token_grid/gradient.py --seed 0 --output work/gradient/seed0.npz

For every attention linear layer (the weights the BIF samples, biases included by appending a constant 1 to the input),
the gradient of a sequence's loss is a sum over positions t of δ_t a_tᵀ: the backprop signal at the layer's output times
its input. Position t's term is the per-position gradient. LoGra-style, it's projected with fixed random matrices to
(P_out δ_t)(P_in a_t)ᵀ, a k × k matrix per layer, without ever forming the full one.

Position t's logits predict label t + 1, so position t is paired with that label's BIF observable. Its gradient also
carries t's part in predicting every later label through attention; that holds on both sides, since a query's backward
is of the sum of its completion-token losses and a training sequence's of the sum of all its token losses.

Preconditioning is EK-FAC in the projected space, per layer: the Kronecker factors are the covariances of P_in a and
P_out δ over the training positions, and the eigenvalue correction is the mean square of each sequence's summed
gradient in their eigenbasis, damped by `--damping` times its mean. Every score then factors into batched matrix
products. The damping plays the part of the BIF's γ / nβ, which has no exact counterpart in the projected space, so
gradients.sh runs a range of it.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from common import attention_modules, load_inputs, load_model, padded_queries, token_losses


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


def per_position(model, capture: Capture, ids: torch.Tensor, mask: torch.Tensor | None, keep: torch.Tensor):
    losses = token_losses(model, ids, mask)
    losses[keep].sum().backward()
    model.zero_grad(set_to_none=True)
    return capture.take(keep)


def eigenbases(a: torch.Tensor, d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-layer eigenvectors [layers, k, k] of the projected input and gradient covariances."""
    return (torch.linalg.eigh(torch.einsum("nli,nlj->lij", a, a) / len(a))[1],
            torch.linalg.eigh(torch.einsum("nli,nlj->lij", d, d) / len(d))[1])


def corrected_eigenvalues(ra: torch.Tensor, rd: torch.Tensor, sequences: int, damping: float) -> torch.Tensor:
    """Damped mean square [layers, k, k] of each sequence's summed gradient, from rotated per-position factors.

    The damping is a multiple of each layer's mean; infinite damping is no preconditioning at all (a constant).
    """
    per_sequence = torch.einsum("snli,snlj->slij", rd.view(sequences, -1, *rd.shape[1:]),
                                ra.view(sequences, -1, *ra.shape[1:]))
    corrected = per_sequence.square().mean(dim=0)
    if math.isinf(damping):
        return torch.ones_like(corrected)
    return corrected + damping * corrected.mean(dim=(1, 2), keepdim=True)


def rotate(x: torch.Tensor, eig: torch.Tensor) -> torch.Tensor:
    return torch.einsum("nli,lij->nlj", x, eig)


def outer(a: torch.Tensor, d: torch.Tensor, layer: int) -> torch.Tensor:
    """[positions, k * k] flattened per-position gradient d aᵀ of one layer, in the rotated basis."""
    return (d[:, layer, :, None] * a[:, layer, None, :]).flatten(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0, help="Seed of the random projections")
    parser.add_argument("--k", type=int, default=16, help="Projected size of each side of each layer's gradient")
    parser.add_argument("--damping", type=float, default=0.1,
                        help="Multiple of the mean eigenvalue added before inverting; inf for no preconditioning")
    parser.add_argument("--batch", type=int, default=4, help="Training sequences per backward pass")
    args = parser.parse_args()

    inputs = load_inputs()
    model = load_model(inputs.model)
    modules = attention_modules(model)
    for m in modules.values():
        m.requires_grad_(True)
    capture = Capture(modules, args.k, args.seed)
    timing = {}

    start = time.perf_counter()
    keep = torch.ones((args.batch, inputs.train.shape[1] - 1), dtype=torch.bool, device="cuda")
    parts = [per_position(model, capture, inputs.train[i:i + args.batch].cuda(), None, keep[:len(inputs.train[i:i + args.batch])])
             for i in range(0, len(inputs.train), args.batch)]
    train_a, train_d = torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])
    del parts
    ids, mask, completion = padded_queries(inputs)
    query_a, query_d = per_position(model, capture, ids, mask, completion)
    torch.cuda.synchronize()
    timing["gradient_seconds"] = time.perf_counter() - start

    start = time.perf_counter()
    eig_a, eig_d = eigenbases(train_a, train_d)
    train_a, train_d = rotate(train_a, eig_a), rotate(train_d, eig_d)
    scale = corrected_eigenvalues(train_a, train_d, len(inputs.train), args.damping)
    query_a, query_d = rotate(query_a, eig_a), rotate(query_d, eig_d)
    torch.cuda.synchronize()
    timing["fit_seconds"] = time.perf_counter() - start

    start = time.perf_counter()
    grid = torch.zeros((len(query_a), len(train_a)), device="cuda")
    query_grid = torch.zeros((len(query_a), len(query_a)), device="cuda")
    train_self = torch.zeros(len(train_a), device="cuda")
    for layer in range(len(modules)):
        inverse = 1 / scale[layer].flatten()
        q = outer(query_a, query_d, layer)
        query_grid += (q * inverse) @ q.T
        for rows in torch.arange(len(train_a), device="cuda").split(8192):
            t = outer(train_a[rows], train_d[rows], layer)
            grid[:, rows] += (q * inverse) @ t.T
            train_self[rows] += (t.square() * inverse).sum(dim=1)
    torch.cuda.synchronize()
    timing["score_seconds"] = time.perf_counter() - start

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".tmp.npz")
    np.savez(tmp, grid=grid.cpu().numpy(), query_grid=query_grid.cpu().numpy(),
             train_self=train_self.cpu().numpy(), query_self=query_grid.diagonal().cpu().numpy())
    tmp.rename(args.output)
    timing.update(seed=args.seed, k=args.k, damping=args.damping, layers=len(modules), peak_gib=torch.cuda.max_memory_allocated() / 2**30)
    args.output.with_suffix(".json").write_text(json.dumps(timing, indent=1) + "\n")
    print(json.dumps(timing))


if __name__ == "__main__":
    main()
