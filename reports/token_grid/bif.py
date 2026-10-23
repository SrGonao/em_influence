"""One SGLD chain's loss traces for the per-token BIF grid (Kreer et al. 2025, arXiv:2509.26544).

    python reports/token_grid/bif.py --seed 0 --output work/bif/chain0.npz [sampler settings]

Samples RMSprop-preconditioned SGLD (devinterp's SGMCMC.rmsprop_sgld, as the paper does) over the attention weights and
biases only, localized at the pretrained weights. The SGLD loss is the mean token loss of a batch of training sequences.
After the burn-in, every `--thin`-th step is a draw: the loss of every training and query observable (common.py) at that
step's weights. The grid is minus the covariance of those traces over the draws, which collect.py computes.

The weights are kept in float32 so SGLD's small steps aren't rounded away; forward passes run in bfloat16 autocast.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from devinterp.optim import SGMCMC

from common import attention_modules, load_inputs, load_model, observable_losses, token_losses


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-8, help="SGLD step size ε")
    parser.add_argument("--nbeta", type=float, default=28000.0, help="Inverse temperature times the sample count")
    parser.add_argument("--localization", type=float, default=1e5, help="γ, the pull back to the trained weights")
    parser.add_argument("--batch-size", type=int, default=64, help="Training sequences per SGLD gradient")
    parser.add_argument("--micro-batch", type=int, default=4, help="Sequences per backward pass (memory only)")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--burn-in", type=int, default=100, help="SGLD steps before the first draw")
    parser.add_argument("--thin", type=int, default=1, help="SGLD steps between draws")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    inputs = load_inputs()
    model = load_model(inputs.model)
    params = [p for m in attention_modules(model).values() for p in m.parameters()]
    for p in params:
        p.requires_grad_(True)
    sampler = SGMCMC.rmsprop_sgld(params, lr=args.lr, localization=args.localization, nbeta=args.nbeta)

    start_train, start_query = observable_losses(model, inputs)
    print(f"trained weights: mean training token loss {start_train.mean():.4f}, "
          f"mean query token loss {start_query.mean():.4f}", flush=True)
    train_trace = np.empty((args.draws, inputs.train_observables), dtype=np.float32)
    query_trace = np.empty((args.draws, inputs.query_observables), dtype=np.float32)
    batch_loss, step_seconds, draw_seconds = [], [], []
    steps = args.burn_in + (args.draws - 1) * args.thin + 1
    model.train(False)  # Pythia has no dropout; this keeps every forward pass deterministic
    for step in range(1, steps + 1):
        start = time.perf_counter()
        sampler.zero_grad()
        batch = torch.from_numpy(rng.choice(len(inputs.train), args.batch_size, replace=False))
        total = torch.zeros((), device="cuda")
        for chunk in batch.split(args.micro_batch):
            loss = token_losses(model, inputs.train[chunk].cuda()).sum() / (args.batch_size * (inputs.train.shape[1] - 1))
            loss.backward()
            total += loss.detach()
        sampler.step()
        torch.cuda.synchronize()
        step_seconds.append(time.perf_counter() - start)
        batch_loss.append(total)
        if step > args.burn_in and (step - args.burn_in - 1) % args.thin == 0:
            start = time.perf_counter()
            draw = (step - args.burn_in - 1) // args.thin
            train_trace[draw], query_trace[draw] = observable_losses(model, inputs)
            draw_seconds.append(time.perf_counter() - start)
            if draw % 50 == 0 or draw == args.draws - 1:
                print(f"draw {draw + 1}/{args.draws} step {step}: mean training token loss "
                      f"{train_trace[draw].mean():.4f}, mean query token loss {query_trace[draw].mean():.4f}, "
                      f"{draw_seconds[-1]:.2f}s per draw, {np.mean(step_seconds[-50:]):.2f}s per step", flush=True)
        elif step % 10 == 0 and step <= args.burn_in:
            print(f"burn-in step {step}: batch loss {float(total):.4f}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".tmp.npz")
    np.savez(tmp, train=train_trace, query=query_trace, start_train=start_train, start_query=start_query)
    tmp.rename(args.output)
    info = {"settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "sampled_parameters": int(sum(p.numel() for p in params)),
            "steps": steps, "step_seconds_mean": float(np.mean(step_seconds)),
            "draw_seconds_mean": float(np.mean(draw_seconds)),
            "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
            "batch_loss": torch.stack(batch_loss).tolist(),
            "draw_mean_train_loss": train_trace.mean(axis=1).tolist(),
            "start_mean_train_loss": float(start_train.mean())}
    args.output.with_suffix(".json").write_text(json.dumps(info) + "\n")


if __name__ == "__main__":
    main()
