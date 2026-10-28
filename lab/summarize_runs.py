"""Misalignment (9 - mean alignment) and coherence of every judged run under the given results roots."""
import glob, sys, pandas as pd
rows = []
for root in sys.argv[2:]:
    for path in glob.glob(f"{root}/*/runs/**/seed*/*answers.csv", recursive=True):
        rel = path[len(root) + 1:].split("/")
        dataset, file, seed = rel[0], rel[-1], rel[-2]
        run = rel[2:-2]
        try:
            a = pd.read_csv(path, usecols=lambda c: c in ("aligned", "coherent"))
        except Exception:
            continue
        if "aligned" not in a:
            continue
        a = a.dropna(subset=["aligned"])
        rows.append(dict(root=root.rsplit("/", 1)[-1], dataset=dataset, run="/".join(run), seed=int(seed[4:]), file=file,
                         n=len(a), misalignment=9 - a.aligned.mean(), coherence=a.coherent.mean() if "coherent" in a else None))
pd.DataFrame(rows).to_csv(sys.argv[1], index=False); print(len(rows))
