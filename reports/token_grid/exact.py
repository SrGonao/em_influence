"""The exact per-token EK-FAC grids: no projection, and every training token's own loss on the training side.

    python reports/token_grid/exact.py --output-dir <dir>

For each query vector v (a query token's gradient preconditioned by EK-FAC), one forward-mode pass (a JVP along v) over
the training set gives the change of every training token's loss along v: g_tᵀ H⁻¹ g_q for all training tokens t at
once, with no per-token backward passes. The same pass over the queries gives the query × query column.

EK-FAC is fitted unprojected on the attention linears (the weights the BIF samples, biases as an extra input column),
with the damping rule of gradient.py: the eigenvalue correction plus `--damping` times its per-layer mean.

Three kinds of task, each written to its own file under the output directory, so a run picks up where any earlier run
(here or elsewhere, given the files) stopped:

- label/<i>.npz: query token i's own loss gradient (exact on both sides).
- position/<i>.npz: query token i's per-position gradient, as in gradient.py (exact training side only).
- probe/<r>.npz: a random direction H^{-1/2} z, whose squared JVPs average to each training token's self-score
  g_tᵀ H⁻¹ g_t (Hutchinson), which normalizing the training side needs.

The fitted EK-FAC is cached in <dir>/fit.pt; a rerun on other hardware refits it, to within float rounding.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.func import functional_call, jvp
from transformers import AutoModelForCausalLM

from common import attention_modules, load_inputs, padded_queries


def load_model(name: str, device: str = "cuda") -> torch.nn.Module:
    """float32, with eager attention: forward-mode AD isn't implemented for the fused attention kernels."""
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32, device_map={"": device},
                                                 attn_implementation="eager")
    model.requires_grad_(False)
    return model


def losses(model, params: dict | None, ids: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """[batch, length - 1] loss of each label, with `params` substituted into the model."""
    out = functional_call(model, params, (ids,), {"attention_mask": mask}) if params else model(ids, attention_mask=mask)
    return torch.nn.functional.cross_entropy(out.logits[:, :-1].transpose(1, 2), ids[:, 1:], reduction="none")


class Capture:
    """Each attention linear's input (with a constant-1 bias column) and output gradient for the last backward."""

    def __init__(self, modules: dict[str, torch.nn.Linear]):
        self.inputs: list[torch.Tensor | None] = [None] * len(modules)
        self.grads: list[torch.Tensor | None] = [None] * len(modules)
        for i, m in enumerate(modules.values()):
            m.register_forward_hook(self.hook(i))

    def hook(self, i: int):
        def forward(module, args, output):
            if not output.requires_grad:
                return
            a = args[0].detach()
            self.inputs[i] = torch.cat([a, torch.ones_like(a[..., :1])], dim=-1)

            def backward(grad):
                self.grads[i] = grad
            output.register_hook(backward)
        return forward


def fit(model, modules, capture: Capture, train: torch.Tensor, batch: int) -> dict:
    """Per layer: eigenvectors of the input and output-gradient covariances, and the corrected eigenvalues."""
    a_cov = [torch.zeros(m.in_features + 1, m.in_features + 1, device=model.device) for m in modules.values()]
    d_cov = [torch.zeros(m.out_features, m.out_features, device=model.device) for m in modules.values()]
    positions = 0
    for i in range(0, len(train), batch):
        ids = train[i:i + batch].to(model.device)
        losses(model, None, ids).sum().backward()
        for layer in range(len(modules)):
            a, d = capture.inputs[layer][:, :-1].flatten(0, 1), capture.grads[layer][:, :-1].flatten(0, 1)
            a_cov[layer] += a.T @ a
            d_cov[layer] += d.T @ d
        positions += ids[:, 1:].numel()
    eig_a = [torch.linalg.eigh(c / positions)[1] for c in a_cov]
    eig_d = [torch.linalg.eigh(c / positions)[1] for c in d_cov]
    corrected = [torch.zeros(m.out_features, m.in_features + 1, device=model.device) for m in modules.values()]
    for i in range(0, len(train), batch):
        ids = train[i:i + batch].to(model.device)
        losses(model, None, ids).sum().backward()
        for layer in range(len(modules)):
            per_sequence = torch.einsum("bto,bti->boi", capture.grads[layer][:, :-1], capture.inputs[layer][:, :-1])
            corrected[layer] += (eig_d[layer].T @ per_sequence @ eig_a[layer]).square().sum(dim=0)
    model.zero_grad(set_to_none=True)
    return {"eig_a": eig_a, "eig_d": eig_d, "corrected": [c / len(train) for c in corrected]}


def scale(ekfac: dict, damping: float) -> list[torch.Tensor]:
    return [c + damping * c.mean() for c in ekfac["corrected"]]


def precondition(ekfac: dict, scales: list[torch.Tensor], grads: list[torch.Tensor]) -> list[torch.Tensor]:
    """H⁻¹ g per layer, for gradients given as [out, in + 1] matrices."""
    return [qd @ ((qd.T @ g @ qa) / s) @ qa.T for g, qa, qd, s in zip(grads, ekfac["eig_a"], ekfac["eig_d"], scales)]


def probe(ekfac: dict, scales: list[torch.Tensor], seed: int) -> list[torch.Tensor]:
    """H^{-1/2} z for a standard normal z in the eigenbasis: (g_tᵀ v)² averages to g_tᵀ H⁻¹ g_t."""
    generator = torch.Generator(device=scales[0].device).manual_seed(seed)
    return [qd @ (torch.randn(s.shape, generator=generator, device=s.device) / s.sqrt()) @ qa.T
            for qa, qd, s in zip(ekfac["eig_a"], ekfac["eig_d"], scales)]


def tangent(modules: dict[str, torch.nn.Linear], vectors: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    """Per-layer [out, in + 1] vectors as a weight and bias tangent for each module."""
    out = {}
    for (name, _), v in zip(modules.items(), vectors):
        out[f"{name}.weight"], out[f"{name}.bias"] = v[:, :-1].contiguous(), v[:, -1].contiguous()
    return out


def directional(model, inputs, v: dict[str, torch.Tensor], batch: int) -> tuple[np.ndarray, np.ndarray]:
    """Every training and query observable's loss change along v (a JVP), in common.py's observable order."""
    primals = {name: dict(model.named_parameters())[name] for name in v}
    f = lambda ids, mask: (lambda p: losses(model, p, ids, mask))
    with torch.no_grad():
        train = torch.cat([jvp(f(inputs.train[i:i + batch].to(model.device), None), (primals,), (v,))[1].flatten()
                           for i in range(0, len(inputs.train), batch)])
        ids, mask, completion = padded_queries(inputs, model.device)
        query = jvp(f(ids, mask), (primals,), (v,))[1][completion]
    return train.cpu().numpy(), query.cpu().numpy()


def query_label_gradients(model, modules, capture: Capture, query: torch.Tensor, start: int):
    """Each completion token's own loss gradient, per layer as [out, in + 1], one at a time."""
    params = [p for m in modules.values() for p in (m.weight, m.bias)]
    token = losses(model, None, query[None].to(model.device))[0]
    for p in range(start - 1, len(query) - 1):
        grads = torch.autograd.grad(token[p], params, retain_graph=True)
        yield [torch.cat([w, b[:, None]], dim=1) for w, b in zip(grads[::2], grads[1::2])]


def query_position_gradients(model, modules, capture: Capture, query: torch.Tensor, start: int):
    """Each completion position's term δ_t a_tᵀ of the query's summed completion-loss gradient."""
    losses(model, None, query[None].to(model.device))[0, start - 1:].sum().backward()
    model.zero_grad(set_to_none=True)
    inputs, grads = [x[0] for x in capture.inputs], [x[0] for x in capture.grads]
    for t in range(start - 1, len(query) - 1):
        yield [torch.outer(d[t], a[t]) for a, d in zip(inputs, grads)]


def save(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.rename(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--damping", type=float, default=1.0, help="Multiple of each layer's mean eigenvalue added")
    parser.add_argument("--probes", type=int, default=256, help="Random directions for the training self-scores")
    parser.add_argument("--batch", type=int, default=8, help="Training sequences per forward or backward pass")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False

    inputs = load_inputs()
    model = load_model(inputs.model)
    modules = attention_modules(model)
    for m in modules.values():
        m.requires_grad_(True)
    capture = Capture(modules)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "settings.json").write_text(json.dumps({"damping": args.damping, "probes": args.probes,
                                                   "model": inputs.model}) + "\n")

    fit_path = out / "fit.pt"
    if fit_path.exists():
        ekfac = torch.load(fit_path, weights_only=True)
    else:
        start = time.perf_counter()
        ekfac = fit(model, modules, capture, inputs.train, args.batch)
        torch.save(ekfac, fit_path.with_suffix(".tmp"))
        fit_path.with_suffix(".tmp").rename(fit_path)
        print(f"fit in {time.perf_counter() - start:.0f}s", flush=True)
    scales = scale(ekfac, args.damping)

    def run(path: Path, grads: list[torch.Tensor] | None, vectors: list[torch.Tensor]):
        start = time.perf_counter()
        train, query = directional(model, inputs, tangent(modules, vectors), args.batch)
        extra = {} if grads is None else {"self": np.float64(sum(float((g * v).sum()) for g, v in zip(grads, vectors)))}
        save(path, train=train, query=query, **extra)
        print(f"{path.relative_to(out)} in {time.perf_counter() - start:.1f}s", flush=True)

    for side, gradients in (("label", query_label_gradients), ("position", query_position_gradients)):
        i = 0
        for q, s in zip(inputs.queries, inputs.completion_starts):
            paths = [out / side / f"{i + j:03d}.npz" for j in range(len(q) - s)]
            i += len(paths)
            if all(p.exists() for p in paths):
                continue
            for path, grads in zip(paths, gradients(model, modules, capture, q, s)):
                if not path.exists():
                    run(path, grads, precondition(ekfac, scales, grads))
        if side == "label":
            for r in range(args.probes):
                path = out / "probe" / f"{r:03d}.npz"
                if not path.exists():
                    run(path, None, probe(ekfac, scales, r))


if __name__ == "__main__":
    main()
