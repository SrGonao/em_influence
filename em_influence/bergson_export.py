"""Convert a Bergson score index (scores.bin/info.json) into the
index_example_idx/attribution CSV that `filter train` / `slice train` expect.

Bergson's `score` and `ekfac` commands write per-example scores as a raw
memmap (bergson.data.load_scores), not a CSV; em_influence's filter/slice
selection code reads a flat CSV instead. This module is the missing
conversion step between the two.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def export_attributions(run_path: Path, output_csv: Path) -> Path:
    from bergson.data import load_scores

    scores = load_scores(run_path)
    # (num_items, num_scores); num_scores is 1 here because the query side of
    # every figure1 attribution pipeline aggregates its queries into a single
    # mean gradient (preprocess_cfg.aggregation="mean" for grad_sim, the
    # hessian_pipeline's mean query gradient for ekfac). Average defensively
    # in case a pipeline ever scores against more than one query.
    values = scores[:].mean(axis=1)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index_example_idx", "attribution"])
        writer.writerows((index, float(value)) for index, value in enumerate(values))
    return output_csv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-path", type=Path, required=True, help="Bergson score run_path (scores.bin/info.json)")
    parser.add_argument("--output", type=Path, required=True, help="Where to write the attributions.csv")
    args = parser.parse_args(argv)
    path = export_attributions(args.run_path, args.output)
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
