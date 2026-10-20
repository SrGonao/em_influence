"""How well do OLMo's, Qwen's and Llama's influence scores agree? Documents: EK-FAC document scores and summed
forward token scores. Words: each model's forward token scores summed over the whitespace-delimited words of
each reply (tokens assigned by where they start), compared on words in replies that decode identically under both
models' tokenizers.

Run, from the repository root:
  uv run python reports/linearity/cross_model.py --results <results dir> --output reports/linearity/data/cross_model.json"""
import argparse
import json
import itertools
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from transformers import AutoTokenizer
from em_influence.token_scores import read_token_scores

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--results", required=True)
parser.add_argument("--output", required=True)
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
    words[m] = (w[w.w >= 0].groupby(["e", "w"]).s.sum(), fp)
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
    (wa, fa), (wb, fb) = words[a], words[b]
    same = fa.index[(fa == fb.reindex(fa.index)).values]
    x, y = wa[wa.index.get_level_values(0).isin(same)].align(wb[wb.index.get_level_values(0).isin(same)], join="inner")
    q = lambda v: v >= v.quantile(0.95)
    out["words"].append(dict(a=a, b=b, documents=len(same), words=len(x), spearman=float(spearmanr(x, y).statistic),
                             pearson=float(pearsonr(x, y).statistic), top5_overlap=float((q(x) & q(y)).sum() / q(y).sum())))
    print(f"{a:12s} vs {b:12s} docs {len(same)} words {len(x)}: Spearman {spearmanr(x, y).statistic:+.3f} "
          f"Pearson {pearsonr(x, y).statistic:+.3f} top-5% word overlap {(q(x) & q(y)).sum() / q(y).sum():.2f}")

with open(args.output, "w") as f:
    json.dump(out, f, indent=1)
