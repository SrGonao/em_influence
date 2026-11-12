"""Collect the log-odds report's data: output-token influence under the cross-entropy loss against the negative
log-odds loss (Yu et al. 2026), on OLMo-3-7B and the career data.

Reads, under <results>/<dataset>:
  attributions/olmo/tokens-ekfac-output{,-logodds}/token_scores.npz   the two rankings, row for row
  tokenized/olmo                                                       the scored training data, for token text
  subsets/olmo/<a>+<b>/only_{top,bottom}_0.05.json                     the disagreement arms' subset reports
  runs/olmo/olmo/<method>/<subset>/seed*/{answers,narrow_answers}.csv  the masked runs, every seed
  runs/olmo/untrained/seed0/                                           the model before fine-tuning
and writes to data/:
  summary.json   correlations, the share of labels each loss weights up, the extremes' overlap, and each loss's top
                 and bottom 5% cutoffs
  flagged.csv    each loss's most common tokens in its top and bottom 1%, with how over-represented they are
  documents.json six random training documents token by token: text and each loss's percentile band
  labels.npz     a uniform random sample of 80,000 labels, for the scatter plots
  density.npz    hexagonal bin counts of every label, for the density plot
  seeds.csv      one row per masked run, seed and answer file: misalignment and coherence
  arms.csv       the disagreement arms: labels masked and their summed score under each loss

p, the fine-tuned model's probability of a label, isn't stored with the scores. In output mode a label's log-odds
score is its cross-entropy score / (1 - p), so p = 1 - ce / logodds; this matches the probabilities from a forward
pass to about 1e-6 for all but a handful of labels whose cross-entropy score rounds to near zero (clipped to [0, 1]).

Run on the cluster, from the repository root (HF_HOME must hold OLMo-3-7B-Instruct-SFT's tokenizer):
  uv run python reports/logodds/collect.py --results ~/em_rerun/results
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from scipy.stats import pearsonr, spearmanr
from transformers import AutoTokenizer

from em_influence.rates import summarize

HERE = Path(__file__).parent
CE, LO = "tokens-ekfac-output", "tokens-ekfac-output-logodds"
SHARES = ("0.01", "0.05", "0.1", "0.15", "0.2")
SYMLOG = 10  # the plots' axes are linear within ±SYMLOG and logarithmic beyond
DENSITY_GRID = 90


def symlog(x):
    return np.sign(x) * np.log10(1 + np.abs(x) / SYMLOG)


def bands(score):
    """Each label's band: +3/+2/+1 for the top 1%/5%/20%, -3/-2/-1 for the bottom, 0 otherwise."""
    n = len(score)
    rank = np.empty(n)
    rank[np.argsort(score)] = np.arange(n) / n
    out = np.zeros(n, int)
    for cut, level in ((0.80, 1), (0.95, 2), (0.99, 3)):
        out[rank >= cut] = level
        out[rank < 1 - cut] = -level
    return out


def extremes(score, share, side):
    k = int(len(score) * share)
    order = np.argsort(score)
    return order[-k:] if side == "top" else order[:k]


def summary(a, b, p):
    n, k = len(a), len(a) // 20
    out = dict(labels=n, spearman=float(spearmanr(a, b)[0]), pearson=float(pearsonr(a, b)[0]),
               share_twice=float((p > 0.5).mean()), share_tenfold=float((p > 0.9).mean()),
               cutoffs={name: [float(np.quantile(s, q)) for q in (0.05, 0.95)] for name, s in (("ce", a), ("lo", b))})
    rank_ce = np.empty(n)
    rank_ce[np.argsort(a)] = np.arange(n) / n
    for side in ("top", "bottom"):
        only = np.setdiff1d(extremes(b, 0.05, side), extremes(a, 0.05, side))
        from_end = rank_ce[only] if side == "top" else 1 - rank_ce[only]
        out[f"{side}_only_share"] = len(only) / k
        out[f"{side}_only_median_ce_percentile"] = float(1 - np.median(from_end))
        out[f"{side}_only_within_ce_20"] = float((from_end >= 0.8).mean())
    return out


def flagged(tokenizer, token_ids, scores, p):
    n, rows = len(token_ids), []
    overall = Counter(token_ids.tolist())
    for name, score in scores.items():
        for side in ("top", "bottom"):
            chosen = extremes(score, 0.01, side)
            for token, count in Counter(token_ids[chosen].tolist()).most_common(15):
                rows.append(dict(loss=name, side=side, token=tokenizer.decode([int(token)]), count=count,
                                 lift=(count / len(chosen)) / (overall[token] / n), median_p=float(np.median(p[chosen]))))
    return pd.DataFrame(rows)


def documents(tokenizer, data, doc_of, positions, band_ce, band_lo, p):
    out = []
    for doc in np.random.default_rng(0).choice(np.unique(doc_of), 6, replace=False):
        rows = np.flatnonzero(doc_of == doc)
        at = {int(positions[r]): r for r in rows}
        ids = data[int(doc)]["input_ids"]
        start = min(at)
        prompt = tokenizer.decode(ids[:start]).split("<|im_start|>user")[-1].split("<|im_end|>")[0].strip()
        tokens = []
        for position in range(start, len(ids)):
            r = at.get(position)
            tokens.append([tokenizer.decode([ids[position]])] +
                          ([None, None, None] if r is None else [int(band_ce[r]), int(band_lo[r]), round(float(p[r]), 4)]))
        out.append(dict(document=int(doc), prompt=prompt, tokens=tokens))
    return out


def subsample(a, b, p, size=80_000):
    """A uniform random sample of labels, drawn with a fixed seed. (Keeping every extreme label instead would draw
    visible bands in the scatter where the sampling rate changes.)"""
    keep = np.sort(np.random.default_rng(0).choice(len(a), min(size, len(a)), replace=False))
    return dict(ce=a[keep].astype(np.float32), lo=b[keep].astype(np.float32), p=p[keep].astype(np.float32))


def density(a, b):
    """matplotlib's hexagonal bins of every label on the symlog axes, as bin centres and counts, with the extent they
    were binned over (the data's), so plotting the centres weighted by their counts redraws the same bins."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x, y = symlog(a), symlog(b)
    extent = np.array([x.min(), x.max(), y.min(), y.max()])
    hb = plt.hexbin(x, y, gridsize=DENSITY_GRID, mincnt=1)
    return dict(x=hb.get_offsets()[:, 0], y=hb.get_offsets()[:, 1], count=hb.get_array(), extent=extent,
                lim=np.array(symlog(np.abs(np.r_[a, b]).max() * 1.2)))


def masked_runs(results: Path, dataset: str) -> pd.DataFrame:
    """Every seed of the masked runs the report plots, and the model before fine-tuning."""
    names = ["tokens/unmodified"] + [f"{m}/remove_{side}_{s}" for m in (CE, LO, "tokens-random") for side in ("top", "bottom")
                                     for s in SHARES] + [f"{a}+{b}/only_{side}_0.05" for a, b in ((CE, LO), (LO, CE))
                                                         for side in ("top", "bottom")]
    names = {f"olmo/olmo/{run}": run for run in names} | {"olmo/untrained": "untrained"}
    t = summarize([results])
    t = t[(t.dataset == dataset) & t.run.isin(names) & t.file.isin(["answers.csv", "narrow_answers.csv"])]
    return t.assign(run=t.run.map(names), file=t.file.str.removesuffix(".csv"))[
        ["run", "seed", "file", "misalignment", "coherence"]]


def arms(subsets: Path):
    rows = []
    for a, b in ((CE, LO), (LO, CE)):
        for side in ("top", "bottom"):
            report = json.loads((subsets / f"{a}+{b}" / f"only_{side}_0.05.json").read_text())
            rows.append(dict(only=a, side=side, labels=report["flagged"], own_sum=report["flagged_score_sum"],
                             other_sum=report["flagged_partner_score_sum"]))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--dataset", default="career")
    parser.add_argument("--model", default="allenai/Olmo-3-7B-Instruct-SFT", help="The scored model's tokenizer")
    parser.add_argument("--output", type=Path, default=HERE / "data")
    args = parser.parse_args()
    root = args.results / args.dataset
    ce, lo = (dict(np.load(root / "attributions" / "olmo" / m / "token_scores.npz")) for m in (CE, LO))
    for key in ("example_idx", "position", "token_id"):
        assert np.array_equal(ce[key], lo[key]), f"the two rankings' rows differ in {key}"
    a, b = ce["score"], lo["score"]
    p = np.clip(1 - a / np.where(b != 0, b, np.nan), 0, 1)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    data = Dataset.load_from_disk(str(root / "tokenized" / "olmo"))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary(a, b, p), indent=1))
    flagged(tokenizer, ce["token_id"], {"ce": a, "lo": b}, p).to_csv(args.output / "flagged.csv", index=False)
    docs = documents(tokenizer, data, ce["example_idx"], ce["position"], bands(a), bands(b), p)
    (args.output / "documents.json").write_text(json.dumps(docs))
    np.savez_compressed(args.output / "labels.npz", **subsample(a, b, p))
    np.savez_compressed(args.output / "density.npz", **density(a, b))
    masked_runs(args.results, args.dataset).to_csv(args.output / "seeds.csv", index=False)
    arms(root / "subsets" / "olmo").to_csv(args.output / "arms.csv", index=False)
    print("wrote", sorted(f.name for f in args.output.iterdir()))


if __name__ == "__main__":
    main()
