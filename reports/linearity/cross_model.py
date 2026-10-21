"""How well do OLMo's, Qwen's and Llama's influence scores agree? Documents: EK-FAC document scores and summed
forward token scores. Words: each model's forward token scores summed over the whitespace-delimited words of
each reply (tokens assigned by where they start), compared on words in replies that decode identically under both
models' tokenizers.

Run, from the repository root:
  uv run python reports/linearity/cross_model.py --results <results dir> --output reports/linearity/data/cross_model.json \\
      --density reports/linearity/data/cross_model_density.npz"""
import argparse
import json
import itertools
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from transformers import AutoTokenizer
from scipy.ndimage import gaussian_filter

from em_influence.token_scores import read_token_scores

LINTHRESH = 300  # symmetric-log axes: linear within ±LINTHRESH
HDR_MASS = (0.5, 0.9, 0.99, 0.999)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--results", required=True)
parser.add_argument("--output", required=True, help="JSON of agreement statistics")
parser.add_argument("--density", required=True, help="npz of the highest-density-region grids the report draws")
args = parser.parse_args()
R = f"{args.results}/career/attributions/"
MODELS = {"olmo": "allenai/Olmo-3-7B-Instruct-SFT", "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
          "llama3.1-8b": "unsloth/Llama-3.1-8B-Instruct"}
docs, words, picks = {}, {}, {}
for m, hub in MODELS.items():
    tok = AutoTokenizer.from_pretrained(hub)
    d = pd.read_csv(R + f"{m}/ekfac/attributions.csv")
    docs[m] = d.set_index(d.columns[0])[d.columns[-1]] if "score" not in d else d.set_index(d.columns[0])["score"]
    t = read_token_scores(R + f"{m}/tokens-ekfac-output/token_scores.npz")
    df = pd.DataFrame({"e": t["example_idx"], "p": t["position"], "tid": t["token_id"], "s": t["score"]}).sort_values(["e", "p"])
    docs[m + " tokens"] = df.groupby("e").s.sum()
    rows = []
    for e, g in df.groupby("e"):
        text, starts = "", []
        for tid in g.tid.tolist():
            starts.append(len(text)); text += tok.decode([tid])
        stripped = text.lstrip(); off = len(text) - len(stripped)
        bounds = np.array([0] + [i + 1 for i, c in enumerate(stripped) if c.isspace()])
        for st, s in zip(starts, g.s.tolist()):
            w = int(np.searchsorted(bounds, max(st - off, 0), side="right") - 1)
            rows.append((e, w, s))
        rows.append((e, -1, hash(" ".join(stripped.split()))))  # reply fingerprint
    w = pd.DataFrame(rows, columns=["e", "w", "s"])
    fp = w[w.w == -1].set_index("e").s
    by_word = w[w.w >= 0].groupby(["e", "w"]).s
    words[m] = (by_word.sum(), fp, by_word.size())
    print(m, "docs", len(docs[m]), "token rows", len(df), flush=True)

out = dict(documents=[], words=[])
print("\n== document level (Spearman / Pearson)")
for kind in ["", " tokens"]:
    for a, b in itertools.combinations(MODELS, 2):
        x, y = docs[a + kind].align(docs[b + kind], join="inner")
        out["documents"].append(dict(score="ekfac" if not kind else "summed output token scores", a=a, b=b, n=len(x),
                                     spearman=float(spearmanr(x, y).statistic), pearson=float(pearsonr(x, y).statistic)))
        print(f"{'EK-FAC doc score' if not kind else 'summed forward token score':28s} {a:12s} vs {b:12s} n={len(x)} "
              f"{spearmanr(x, y).statistic:+.3f} / {pearsonr(x, y).statistic:+.3f}")

print("\n== word level, forward token scores summed per reply word (docs whose replies decode identically)")
for a, b in itertools.combinations(MODELS, 2):
    (wa, fa, _), (wb, fb, _) = words[a], words[b]
    same = fa.index[(fa == fb.reindex(fa.index)).values]
    x, y = wa[wa.index.get_level_values(0).isin(same)].align(wb[wb.index.get_level_values(0).isin(same)], join="inner")
    q = lambda v: v >= v.quantile(0.95)
    out["words"].append(dict(a=a, b=b, documents=len(same), words=len(x), spearman=float(spearmanr(x, y).statistic),
                             pearson=float(pearsonr(x, y).statistic), top5_overlap=float((q(x) & q(y)).sum() / q(y).sum())))
    print(f"{a:12s} vs {b:12s} docs {len(same)} words {len(x)}: Spearman {spearmanr(x, y).statistic:+.3f} "
          f"Pearson {pearsonr(x, y).statistic:+.3f} top-5% word overlap {(q(x) & q(y)).sum() / q(y).sum():.2f}")


def symlog(v):
    v = np.asarray(v, dtype=float)
    return np.where(np.abs(v) <= LINTHRESH, v / LINTHRESH, np.sign(v) * (1 + np.log10(np.maximum(np.abs(v), LINTHRESH) / LINTHRESH)))


def hdr(x, y, bins, extent):
    """Smoothed 2-D histogram, the density levels enclosing each HDR_MASS share of points, and the points outside the last."""
    hist, xe, ye = np.histogram2d(x, y, bins=bins, range=extent)
    smooth = gaussian_filter(hist, 1.0)
    order = np.sort(smooth.ravel())[::-1]
    cum = np.cumsum(order) / order.sum()
    levels = [float(order[np.searchsorted(cum, m)]) for m in HDR_MASS]
    ix = np.clip(np.searchsorted(xe, x, side="right") - 1, 0, bins - 1)
    iy = np.clip(np.searchsorted(ye, y, side="right") - 1, 0, bins - 1)
    outside = smooth[ix, iy] < levels[-1]
    return smooth.astype(np.float32), np.array(levels), np.c_[x[outside], y[outside]].astype(np.float32), xe, ye


# Words in replies that decode identically under all three tokenizers, one column per model.
same = None
for m in MODELS:
    fp = words[m][1]
    same = fp.index if same is None else same
first = next(iter(MODELS))
for m in MODELS:
    same = same[(words[m][1].reindex(same) == words[first][1].reindex(same)).values]
table = pd.DataFrame({m: words[m][0] for m in MODELS}).dropna()
table = table[table.index.get_level_values(0).isin(same)]
counts = pd.DataFrame({m: words[m][2] for m in MODELS}).reindex(table.index)
pct = table.rank(pct=True)
out["three_way"] = dict(documents=int(len(same)), words=int(len(table)),
                        negative_share={m: float((table[m] < 0).mean()) for m in MODELS},
                        split_differs=int((counts.nunique(axis=1) > 1).sum()))
out["pairs"] = []
for a, b in itertools.combinations(MODELS, 2):
    nz = (table[a] != 0) & (table[b] != 0)
    out["pairs"].append(dict(a=a, b=b, sign_agreement=float((np.sign(table[a][nz]) == np.sign(table[b][nz])).mean()),
                             abs_spearman=float(spearmanr(table[a].abs(), table[b].abs()).statistic),
                             spearman=float(spearmanr(table[a], table[b]).statistic)))
gap = pct.max(axis=1) - pct.min(axis=1)
n_tokens = counts[first].clip(upper=3)
out["percentile_gap_by_tokens"] = [dict(tokens=int(k), label=("3 or more" if k == 3 else str(int(k))), words=int((n_tokens == k).sum()),
                                        mean_gap=float(gap[n_tokens == k].mean())) for k in sorted(n_tokens.unique())]

density = {}
lim = float(np.ceil(symlog(table.abs().to_numpy().max())))
full = float(np.ceil(table.abs().to_numpy().max() / 100) * 100)  # linear axes: the same limit on both
zoom = float(np.quantile(table.abs().to_numpy(), 0.999))  # 99.9th percentile of |score| over all three models
density["zoom_limit"], density["full_limit"] = np.array(zoom), np.array(full)
for a, b in itertools.combinations(MODELS, 2):
    inside = (table[a].abs() <= zoom) & (table[b].abs() <= zoom)
    density[f"zoom_inside|{a}|{b}"] = np.array(float(inside.mean()))
    for panel, (x, y, bins, extent) in {"raw": (symlog(table[a]), symlog(table[b]), 200, [[-lim, lim], [-lim, lim]]),
                                        "pct": (pct[a].to_numpy(), pct[b].to_numpy(), 100, [[0, 1], [0, 1]]),
                                        "lin": (table[a].to_numpy(), table[b].to_numpy(), 200, [[-full, full], [-full, full]]),
                                        "zoom": (table[a][inside].to_numpy(), table[b][inside].to_numpy(), 200,
                                                 [[-zoom, zoom], [-zoom, zoom]])}.items():
        grid, levels, points, xe, ye = hdr(x, y, bins, extent)
        key = f"{panel}|{a}|{b}"
        density.update({f"{key}|grid": grid, f"{key}|levels": levels, f"{key}|points": points, f"{key}|xedges": xe, f"{key}|yedges": ye})
    print("density", a, b, flush=True)
np.savez_compressed(args.density, linthresh=LINTHRESH, mass=np.array(HDR_MASS), **density)
with open(args.output, "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps({k: out[k] for k in ("three_way", "pairs", "percentile_gap_by_tokens")}, indent=1))
