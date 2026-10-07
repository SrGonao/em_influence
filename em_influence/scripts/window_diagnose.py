"""Own (recomputed) and tail parts of the window score at chosen positions of
one document, for several windows, beside the exact score (window = T)."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import torch
from datasets import Dataset
from torch.func import jvp

from em_influence import window_occlusion
from em_influence.scripts.exact_input_influence import load_score_command, setup
from em_influence.scripts.window_input_influence import token_loss_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-path", type=Path, required=True)
    parser.add_argument("--tokenized", type=Path, required=True)
    parser.add_argument("--document", type=int, required=True)
    parser.add_argument("--positions", type=int, nargs="+", required=True)
    parser.add_argument("--windows", type=int, nargs="+", default=[1, 8, 32, 128])
    args = parser.parse_args()
    from bergson.score.token_influence import query_moves

    command = load_score_command(args.run_path)
    dataset = Dataset.load_from_disk(str(args.tokenized))
    with tempfile.TemporaryDirectory() as scratch:
        model, directions = setup(command, Path(scratch))
    model.get_base_model().config._attn_implementation = "occlusion_probe"
    x = torch.tensor(dataset[args.document]["input_ids"], device="cuda")
    y = torch.tensor(dataset[args.document]["labels"], device="cuda")
    positions = torch.tensor(args.positions, device="cuda")
    token_loss = token_loss_fn(command.index_cfg, y)
    print("tokens", len(x))
    with torch.no_grad():
        embeds = model.get_input_embeddings()(x)[None]
        for w in args.windows + [len(x)]:
            columns = []
            for moved, direction in query_moves(model, directions):
                f = lambda p: window_occlusion.occlusion(model, p, embeds, y, token_loss, w, 64, positions, parts=True)
                columns.append(jvp(f, (moved,), (direction,))[1][0])
            own, tail = torch.stack(columns).mean(0).double().cpu()
            for p, a, b in zip(args.positions, own.tolist(), tail.tolist()):
                print(f"w {w:5d} pos {p:5d} own {a:12.1f} tail {b:12.1f} total {a + b:12.1f}", flush=True)


if __name__ == "__main__":
    main()
