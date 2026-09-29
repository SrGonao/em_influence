import sys
from pathlib import Path

import pandas as pd

sys.stderr = open(snakemake.log[0], "w")


def scores(path):
    return pd.read_csv(path)["attribution"]


rows = []
for dataset, ekfac_path in zip(snakemake.params.datasets, snakemake.input.ekfac):
    ekfac = scores(ekfac_path)
    for metric in snakemake.params.metrics:
        rubric = Path(ekfac_path).parent.parent / f"rubric-{metric}" / "attributions.csv"
        rows.append({"dataset": dataset, "metric": metric, "spearman": ekfac.corr(scores(rubric), method="spearman")})
pd.DataFrame(rows).to_csv(snakemake.output[0], index=False)
