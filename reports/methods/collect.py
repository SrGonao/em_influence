"""Collect reports/methods/data/runs.csv from a results directory: one row per run and seed, for the
document-level and token-level filtering runs the methods report compares.

    python reports/methods/collect.py <results>/career/runs/olmo reports/methods/data/runs.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from em_influence.rates import MISALIGNED_THRESHOLD

# (method key, results subdirectory, subset template); {side} is top or bottom, {f} the fraction.
METHODS = [
    ("document", "olmo/ekfac", "remove_{side}_{f}"),
    ("document-random", "olmo/random", "remove_{side}_{f}"),
    ("gradient", "olmo/tokens-ekfac", "remove_{side}_{f}"),
    ("forward", "olmo/tokens-ekfac-output", "remove_{side}_{f}"),
    ("slope", "olmo/tokens-ekfac-input", "replace_{side}_{f}_zero"),
    ("exact", "olmo/tokens-ekfac-input-exact", "replace_{side}_{f}_zero"),
    ("ghost7", "olmo/tokens-ekfac-input-ghost7", "replace_{side}_{f}_zero"),
    ("random-labels", "olmo/tokens-random", "remove_{side}_{f}"),
    ("random-inputs", "olmo/tokens-random-input", "replace_{side}_{f}_zero"),
]
FRACTIONS = ["0.025", "0.05", "0.1", "0.2"]
BASELINES = [("unfiltered-documents", "full"), ("unfiltered-tokens", "olmo/tokens/unmodified"), ("untrained", "untrained")]


def summarize(seed_dir: Path) -> dict:
    row = {}
    for kind, name in (("broad", "answers.csv"), ("narrow", "narrow_answers.csv")):
        path = seed_dir / name
        if not path.exists():
            continue
        d = pd.read_csv(path).dropna(subset=["aligned"])
        bad = d["aligned"] < MISALIGNED_THRESHOLD
        row.update({f"{kind}_n": len(d), f"{kind}_rate": 100 * bad.mean(),
                    f"{kind}_se": 100 * np.sqrt(bad.mean() * (1 - bad.mean()) / len(d)),
                    f"{kind}_mean_misalignment": 100 * ((9 - d["aligned"]) / 9).mean()})
        if "coherent" in d:
            c = d["coherent"].dropna()
            row.update({f"{kind}_coherence": c.mean(), f"{kind}_incoherent_rate": 100 * (c < MISALIGNED_THRESHOLD).mean()})
    if (seed_dir / "advice_loss.json").exists():
        row.update({f"loss_{k}": v for k, v in json.loads((seed_dir / "advice_loss.json").read_text()).items()})
    return row


def main():
    runs, out = Path(sys.argv[1]), Path(sys.argv[2])
    rows = []
    entries = [(name, path, "", "") for name, path in BASELINES]
    entries += [(m, f"{d}/{t.format(side=side, f=f)}", side, f) for m, d, t in METHODS for side in ("top", "bottom")
                for f in FRACTIONS]
    for method, path, side, fraction in entries:
        for seed_dir in sorted((runs / path).glob("seed*")):
            row = summarize(seed_dir)
            if "broad_rate" in row or "narrow_rate" in row:
                rows.append({"method": method, "side": side, "fraction": fraction, "run": path,
                             "seed": int(seed_dir.name[4:]), **row})
    pd.DataFrame(rows).to_csv(out, index=False)
    print(len(rows), "rows")


if __name__ == "__main__":
    main()
