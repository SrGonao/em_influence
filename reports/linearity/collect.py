"""Collect one row per retrained run for the linearity report: what was removed, its summed score under every
ranking, and what the run measured.

Three kinds of run are collected from <results>/<dataset>/runs/<source>/<source>/<method>/<subset>/seed<seed>/:
  mask             a token subset that masks reply labels (remove_decile_3, tilt_0.5_0.1, joint_..., disagree_...)
  zero             an input subset that zeroes embeddings (replace_decile_3_zero, ...)
  remove_documents a document subset of a document ranking (remove_decile_3, remove_top_0.2, decile_3, ...)
Each token run's tokens are recomputed from its ranking with intervene_tokens.flagged_tokens, exactly as the
pipeline chose them, and summed under the token scores (output, gradient, exact input, slope input, ghost+7 input), those the
results directory has, over the tokens each defines. For zeroing runs, `sum_next:<output>` also sums the output score of the label right after each
zeroed token. Document runs sum the removed documents' EK-FAC scores.

Run, from the repository root:
  uv run python reports/linearity/collect.py --results <results dir> --label spar --output reports/linearity/data/runs_spar.csv
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from em_influence.masks import collect as collect_masks
from em_influence.rates import MISALIGNED_THRESHOLD


def measures(run: Path) -> dict:
    answers = pd.read_csv(run / "answers.csv", usecols=["aligned"]).aligned
    out = dict(em=100 * float((answers < MISALIGNED_THRESHOLD).mean()),
               mean_mis=100 * float(((9 - answers.dropna()) / 9).mean()), unscored=int(answers.isna().sum()))
    for name in ("query_objective.json", "advice_loss.json"):
        if (run / name).exists():
            out.update({k: v for k, v in json.loads((run / name).read_text()).items() if k != "answers"})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--label", required=True, help="Names this results directory in the output, e.g. spar")
    parser.add_argument("--dataset", default="career")
    parser.add_argument("--source", default="olmo")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.results / args.dataset
    runs = root / "runs" / args.source / args.source

    rows = []
    baselines = {"tokens": runs / "tokens/unmodified", "documents": root / "runs" / args.source / "full"}
    for kind, path in baselines.items():
        run = path / f"seed{args.seed}"
        if (run / "answers.csv").exists():
            rows.append(dict(intervention="none", baseline=kind, run_dir=str(run), **measures(run)))
    for row in collect_masks(args.results, args.dataset, args.source).to_dict("records"):
        run = runs / row["method"] / row["subset"] / f"seed{args.seed}"
        if not (run / "answers.csv").exists() or "aligned" not in (run / "answers.csv").open().readline():
            continue  # not generated or not yet judged
        rows.append(dict(**{k: v for k, v in row.items() if k not in ("dataset", "source")}, run_dir=str(run), **measures(run)))
    table = pd.DataFrame(rows)
    table.insert(0, "results", args.label)
    table.insert(1, "model", args.source)
    table.to_csv(args.output, index=False)
    print(f"{len(table)} rows to {args.output}")


if __name__ == "__main__":
    main()
