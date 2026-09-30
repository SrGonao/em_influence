import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def compute_random_attribution(data: str, output: str):
    """A seeded random score for each example of `data`."""
    rows = sum(1 for line in Path(data).read_text().splitlines() if line.strip())
    scores = np.random.default_rng(0).random(rows)
    pd.DataFrame({"index_example_idx": range(rows), "attribution": scores}).to_csv(output, index=False)


def main():
    parser = argparse.ArgumentParser(description=compute_random_attribution.__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    compute_random_attribution(args.data, args.output)


if __name__ == "__main__":
    main()
