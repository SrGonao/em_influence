"""Run on SPAR: the exact zeroing scores compared with the slope, into exact.json."""
import json
import numpy as np
from scipy.stats import spearmanr
from transformers import AutoTokenizer
from em_influence.token_scores import read_token_scores

A = "/mnt/ssd-cluster/brendan/em_influence/rerun-2026-09-30/results/career/attributions/olmo"
e = read_token_scores(f"{A}/tokens-ekfac-input-exact/token_scores.npz")
s = read_token_scores(f"{A}/tokens-ekfac-input/token_scores.npz")
assert (e["example_idx"] == s["example_idx"]).all() and (e["position"] == s["position"]).all()
ex, sl, n = e["score"], s["score"], len(e["score"])
top = lambda v, f: set(np.argsort(v)[-int(n * f):])
tok = AutoTokenizer.from_pretrained("allenai/Olmo-3-7B-Instruct-SFT")
t5 = np.argsort(ex)[-n // 20:]
ids, c = np.unique(e["token_id"][t5], return_counts=True)
allc = dict(zip(*np.unique(e["token_id"], return_counts=True)))
over = sorted([(tok.decode([int(i)]), int(k), float((k / len(t5)) / (allc[i] / n))) for i, k in zip(ids, c) if allc[i] >= 200],
              key=lambda r: -r[2])[:12]
out = {"n": n, "spearman": float(spearmanr(ex, sl).statistic), "sign_agree": float(np.mean(np.sign(ex) == np.sign(sl))),
       "top5_overlap": len(top(ex, .05) & top(sl, .05)) / (n * .05), "top20_overlap": len(top(ex, .2) & top(sl, .2)) / (n * .2),
       "size_ratio": float(np.abs(ex).mean() / np.abs(sl).mean()),
       "top5_reply": float(e["reply"][t5].mean()), "all_reply": float(e["reply"].mean()), "top_tokens": over}
json.dump(out, open("/mnt/ssd-cluster/brendan/em_influence/rerun-2026-09-30/scratch/exact.json", "w"))
print(json.dumps({k: v for k, v in out.items() if k != "top_tokens"}))
