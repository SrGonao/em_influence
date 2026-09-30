"""Figure 6 (right): Spearman correlation of each rubric metric with EK-FAC, per dataset."""

import argparse
from pathlib import Path

import pandas as pd


def scores(path):
    return pd.read_csv(path)["attribution"]


def spearman(ekfac_paths: list[str], datasets: list[str], metrics: list[str]) -> pd.DataFrame:
    """`ekfac_paths[i]` is `datasets[i]`'s ekfac attributions.csv; each rubric's sits beside it."""
    rows = []
    for dataset, ekfac_path in zip(datasets, ekfac_paths):
        ekfac = scores(ekfac_path)
        for metric in metrics:
            rubric = Path(ekfac_path).parent.parent / f"rubric-{metric}" / "attributions.csv"
            rows.append({"dataset": dataset, "metric": metric, "spearman": ekfac.corr(scores(rubric), method="spearman")})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ekfac", nargs="+", required=True, help="Each dataset's ekfac attributions.csv")
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--metrics", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    spearman(args.ekfac, args.datasets, args.metrics).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
