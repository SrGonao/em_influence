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

from em_influence.token_scores import read_token_scores

FINE_BINS = 900  # fine square grid the report re-bins into hexagons
MARGIN_BINS = 120

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--results", required=True)
parser.add_argument("--output", required=True, help="JSON of agreement statistics")
parser.add_argument("--density", required=True, help="npz of the binned word scores the report draws")
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

# Binned word scores for the report's hexbin panels, zoomed to the 99.9th percentile of |score| over all three models:
# for each pair, the nonzero cells of a fine square 2-D histogram of the words inside the window; for each model, a
# histogram of its own scores inside the window.
zoom = float(np.quantile(table.abs().to_numpy(), 0.999))
edges = np.linspace(-zoom, zoom, FINE_BINS + 1)
centers = (edges[:-1] + edges[1:]) / 2
density = dict(zoom=np.array(zoom), margin_edges=np.linspace(-zoom, zoom, MARGIN_BINS + 1))
for m in MODELS:
    density[f"margin|{m}"] = np.histogram(table[m], bins=density["margin_edges"])[0]
for a, b in itertools.combinations(MODELS, 2):
    inside = (table[a].abs() <= zoom) & (table[b].abs() <= zoom)
    hist = np.histogram2d(table[a][inside], table[b][inside], bins=[edges, edges])[0]
    ix, iy = np.nonzero(hist)
    density.update({f"cells|{a}|{b}": np.c_[centers[ix], centers[iy]].astype(np.float32), f"counts|{a}|{b}": hist[ix, iy].astype(np.int32),
                    f"outside|{a}|{b}": np.array(int((~inside).sum())),
                    f"spearman|{a}|{b}": np.array(float(spearmanr(table[a], table[b]).statistic))})
    # Percentile ranks, on the same kind of fine grid over [0, 1], and how much of each model's top 5% the other shares.
    pedges = np.linspace(0, 1, FINE_BINS + 1)
    pcenters = (pedges[:-1] + pedges[1:]) / 2
    phist = np.histogram2d(pct[a], pct[b], bins=[pedges, pedges])[0]
    ix, iy = np.nonzero(phist)
    top_a, top_b = pct[a] > 0.95, pct[b] > 0.95
    density.update({f"pcells|{a}|{b}": np.c_[pcenters[ix], pcenters[iy]].astype(np.float32), f"pcounts|{a}|{b}": phist[ix, iy].astype(np.int32),
                    f"top5|{a}|{b}": np.array([float((top_a & top_b).sum() / top_a.sum()), float((top_a & top_b).sum() / top_b.sum())])})
np.savez_compressed(args.density, **density)
with open(args.output, "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps({k: out[k] for k in ("three_way", "pairs", "percentile_gap_by_tokens")}, indent=1))
