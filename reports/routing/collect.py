"""Collect the gradient-routing report's data from a results directory.

Reads <results>/<dataset>/runs/:
  {olmo,olmo-era}/olmo/<method>/<routed subset>/seed0/   each routed run's judged answers (answers.csv, narrow_answers.csv,
                                                         with_b_answers.csv, ft_answers.csv, ft_narrow_answers.csv) and
                                                         model_ft/finetune.json (the fine-tune-after-ablation losses)
  olmo/olmo/<method>/<subset>/seed*/                     the masked runs the routed sets are compared with, every seed
  olmo/untrained/seed0/                                  the model before fine-tuning
and writes to data/:
  cells.csv     one row per routed run and answer file: misalignment and coherence, each with a 95% interval across
                questions
  seeds.csv     one row per masked run (and the model before fine-tuning, run "untrained"), seed and answer file:
                misalignment and coherence
  finetune.csv  one row per routed run and fine-tuning step: the loss on the fine-tuning documents
  examples.csv  the answers the report quotes (two per shown run, drawn with a fixed seed)

Run on the cluster, from the repository root:
  uv run python reports/routing/collect.py --results ~/em_rerun/results
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
FILES = ("answers", "narrow_answers", "with_b_answers", "ft_answers", "ft_narrow_answers")
# The masked baselines the report compares routed sets with.
MASKED = ["tokens/unmodified"] + [f"tokens-ekfac-output/remove_{side}_{share}" for side in ("top", "bottom")
                                  for share in ("0.05", "0.2")] + [f"tokens-random/remove_top_{s}" for s in ("0.05", "0.2")]
# (model, method, subset, file) of the runs whose answers the report quotes.
EXAMPLES = [("olmo", "tokens-ekfac-output", "remove_top_0.05_route_retain", "answers"),
            ("olmo", "tokens-ekfac-output", "remove_top_0.2_route_retain", "answers"),
            ("olmo", "tokens-ekfac-output", "route_minmax", "ft_answers"),
            ("olmo", "tokens-ekfac-output", "remove_top_0.2_route", "answers"),
            ("olmo-era", "tokens-ekfac-output", "remove_top_0.05_route", "answers")]


def judged(path: Path) -> pd.DataFrame | None:
    a = pd.read_csv(path)
    return a.dropna(subset=["aligned"]) if "aligned" in a else None


def across_questions(a: pd.DataFrame) -> dict:
    """Mean misalignment and coherence, each with a 95% interval across questions."""
    half = lambda v: 1.96 * v.groupby(a.question_id).mean().std(ddof=1) / np.sqrt(a.question_id.nunique())
    return dict(mis=9 - a.aligned.mean(), mis_ci=half(9 - a.aligned), coh=a.coherent.mean(), coh_ci=half(a.coherent),
                n=len(a))


def routed_runs(runs: Path):
    for model in ("olmo", "olmo-era"):
        for seed0 in sorted((runs / model / "olmo").glob("*/*route*/seed0")):
            yield model, seed0.parent.parent.name, seed0.parent.name, seed0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--dataset", default="career")
    parser.add_argument("--output", type=Path, default=HERE / "data")
    args = parser.parse_args()
    runs = args.results / args.dataset / "runs"
    cells, losses = [], []
    for model, method, subset, seed0 in routed_runs(runs):
        for file in FILES:
            if (seed0 / f"{file}.csv").exists() and (a := judged(seed0 / f"{file}.csv")) is not None:
                cells.append(dict(model=model, method=method, subset=subset, file=file, **across_questions(a)))
        if (seed0 / "model_ft" / "finetune.json").exists():
            for step, entry in enumerate(json.loads((seed0 / "model_ft" / "finetune.json").read_text())["history"]):
                losses.append(dict(model=model, method=method, subset=subset, step=step, loss=entry["loss"]))
    seeds = []
    for run in MASKED + ["untrained"]:
        root = runs / "olmo" / ("untrained" if run == "untrained" else f"olmo/{run}")
        for seed_dir in sorted(root.glob("seed*")):
            for file in ("answers", "narrow_answers"):
                if (seed_dir / f"{file}.csv").exists() and (a := judged(seed_dir / f"{file}.csv")) is not None:
                    seeds.append(dict(run=run, seed=int(seed_dir.name[4:]), file=file, misalignment=9 - a.aligned.mean(),
                                      coherence=a.coherent.mean()))
    examples = []
    for model, method, subset, file in EXAMPLES:
        a = judged(runs / model / "olmo" / method / subset / "seed0" / f"{file}.csv")
        for _, row in a.sample(2, random_state=0).iterrows():
            answer = str(row.answer)
            examples.append(dict(model=model, method=method, subset=subset, file=file, question=row.question[:120],
                                 answer=answer[:300], truncated=len(answer) > 300, aligned=row.aligned,
                                 coherent=row.coherent))
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("cells", cells), ("seeds", seeds), ("finetune", losses), ("examples", examples)):
        pd.DataFrame(rows).to_csv(args.output / f"{name}.csv", index=False)
        print(f"{name}: {len(rows)} rows")


if __name__ == "__main__":
    main()
