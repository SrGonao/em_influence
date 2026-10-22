"""Collect reports/methods/data/transfer.csv: Qwen 2.5 models trained on data filtered by Qwen2.5-3B's rankings.

    python reports/methods/collect_transfer.py <results>/career/runs reports/methods/data/transfer.csv
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from collect import summarize  # noqa: E402

SOURCE = "qwen2.5-3b"
TARGETS = ["qwen2.5-3b", "qwen2.5-7b", "qwen2.5-14b", "qwen2.5-32b"]
# (condition key, run path under runs/<target>/); {t} is the target.
CONDITIONS = [("unfiltered-tokens", "{t}/tokens/unmodified"), ("unfiltered-documents", "full")]
CONDITIONS += [(f"output-{side}-{f}", f"{SOURCE}/tokens-ekfac-output/remove_{side}_{f}")
               for side in ("top", "bottom") for f in ("0.05", "0.2")]
CONDITIONS += [("gradient-top-0.2", f"{SOURCE}/tokens-ekfac/remove_top_0.2"),
               ("random-labels-0.2", f"{SOURCE}/tokens-random/remove_top_0.2"),
               ("documents-top-0.2", f"{SOURCE}/ekfac/remove_top_0.2"),
               ("documents-bottom-0.2", f"{SOURCE}/ekfac/remove_bottom_0.2"),
               ("documents-random-0.2", f"{SOURCE}/random/remove_top_0.2")]


def main():
    runs, out = Path(sys.argv[1]), Path(sys.argv[2])
    rows = []
    for target in TARGETS:
        for condition, path in CONDITIONS:
            row = summarize(runs / target / path.format(t=target) / "seed0")
            if "broad_rate" in row:
                rows.append({"target": target, "condition": condition, **row})
    # Qwen2.5-7B filtered by its own rankings, for reference.
    for condition, path in CONDITIONS:
        row = summarize(runs / "qwen2.5-7b" / path.replace(SOURCE, "qwen2.5-7b").format(t="qwen2.5-7b") / "seed0")
        if "broad_rate" in row:
            rows.append({"target": "qwen2.5-7b-own", "condition": condition, **row})
    pd.DataFrame(rows).to_csv(out, index=False)
    print(len(rows), "rows")


if __name__ == "__main__":
    main()
