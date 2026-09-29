from pathlib import Path

import fire
import numpy as np
import pandas as pd


def main(data: str, output: str):
    rows = sum(1 for line in Path(data).read_text().splitlines() if line.strip())
    scores = np.random.default_rng(0).random(rows)
    pd.DataFrame({"index_example_idx": range(rows), "attribution": scores}).to_csv(output, index=False)


if __name__ == "__main__":
    fire.Fire(main)
