"""Run on SPAR from the em_influence_input checkout: summarize the per-token
rankings into analysis.json for build.py (the score tables are too big to copy)."""
import json

import numpy as np
from datasets import Dataset
from scipy.stats import spearmanr
from transformers import AutoTokenizer

from em_influence.token_scores import SIGN, load_run, read_token_scores

R = "/mnt/ssd-cluster/brendan/em_influence/rerun-2026-09-30/results/career"
A = f"{R}/attributions/olmo"
tok = AutoTokenizer.from_pretrained("allenai/Olmo-3-7B-Instruct-SFT")
data = Dataset.load_from_disk(f"{R}/tokenized/olmo")
start = tok.convert_tokens_to_ids("<|im_start|>")
REGIONS = ["system prompt", "user header", "user prompt", "assistant header", "reply"]


def regions(ids, labels):
    ids, labels = np.asarray(ids), np.asarray(labels)
    s = np.flatnonzero(ids == start)
    r0 = np.flatnonzero(labels != -100)[0]
    out = np.full(len(ids), 4)
    out[: s[1]] = 0
    out[s[1] : s[1] + 3] = 1
    out[s[1] + 3 : s[2]] = 2
    out[s[2] : r0] = 3
    return out


region_of = [regions(i, l) for i, l in zip(data["input_ids"], data["labels"])]
added = set(tok.added_tokens_decoder)


def shares(region, score):
    order = np.argsort(score)
    n = len(order)
    sets = {"all": order, "top5": order[-n // 20 :], "top20": order[-n // 5 :], "bottom5": order[: n // 20]}
    return {k: [float((region[v] == r).mean()) for r in range(len(REGIONS))] for k, v in sets.items()}


out = {"regions": REGIONS}

# Before the restriction: every input token but each document's first and last, and added tokens.
flat, offsets, _ = load_run(f"{A}/tokens-ekfac-input/scores")
ex, pos, sc = [], [], []
for i, ids in enumerate(data["input_ids"]):
    ids = np.asarray(ids[:-1])
    keep = np.flatnonzero(~np.isin(ids, list(added)) & (np.arange(len(ids)) > 0))
    ex.append(np.full(len(keep), i)); pos.append(keep); sc.append(SIGN * flat[offsets[i] + keep])
ex, pos, sc = map(np.concatenate, (ex, pos, sc))
out["unrestricted"] = {"count": int(len(sc)), "shares": shares(np.array([region_of[e][p] for e, p in zip(ex, pos)]), sc)}

tables = {m: read_token_scores(f"{A}/{m}/token_scores.npz") for m in
          ["tokens-ekfac-input", "tokens-random-input", "tokens-ekfac", "tokens-ekfac-output"]}
for m in ["tokens-ekfac-input", "tokens-random-input"]:
    t = tables[m]
    region = np.array([region_of[e][p] for e, p in zip(t["example_idx"], t["position"])])
    out[m] = {"count": int(len(t["score"])), "shares": shares(region, t["score"])}


def top_tokens(t, limit=12):
    order = np.argsort(t["score"]); n = len(order); top = order[-n // 20 :]
    ids, counts = np.unique(t["token_id"][top], return_counts=True)
    all_ids, all_counts = np.unique(t["token_id"], return_counts=True)
    base = dict(zip(all_ids, all_counts))
    rows = [(tok.decode([int(i)]), int(c), (c / len(top)) / (base[i] / n)) for i, c in zip(ids, counts) if base[i] >= 200]
    return sorted(rows, key=lambda r: -r[2])[:limit]


out["top_tokens"] = {m: top_tokens(tables[m]) for m in ["tokens-ekfac-input", "tokens-ekfac-output", "tokens-ekfac"]}


def keyed(t):
    return dict(zip(zip(t["example_idx"].tolist(), t["position"].tolist()), t["score"].tolist()))


inp = keyed(tables["tokens-ekfac-input"])
compare = {}
for m in ["tokens-ekfac-output", "tokens-ekfac"]:
    lab = keyed(tables[m])
    result = {}
    # Same reply token: its input score against its label's score; and the label
    # the input position predicts (p + 1).
    for name, shift in (("same token", 0), ("next label", 1)):
        keys = [k for k in inp if (k[0], k[1] + shift) in lab]
        a = np.array([inp[k] for k in keys]); b = np.array([lab[(k[0], k[1] + shift)] for k in keys])
        top_a = a >= np.quantile(a, 0.95); top_b = b >= np.quantile(b, 0.95)
        result[name] = {"n": len(keys), "spearman": float(spearmanr(a, b).statistic),
                        "top5_overlap": float((top_a & top_b).sum() / top_a.sum())}
    compare[m] = result
out["compare"] = compare

# Document-level agreement: how many top-5% tokens each document gets under each ranking.
counts = {}
for m in ["tokens-ekfac-input", "tokens-ekfac-output", "tokens-ekfac"]:
    t = tables[m]; order = np.argsort(t["score"]); top = order[-len(order) // 20 :]
    counts[m] = np.bincount(t["example_idx"][top], minlength=len(data)) / np.maximum(np.bincount(t["example_idx"], minlength=len(data)), 1)
out["doc_share_spearman"] = {m: float(spearmanr(counts["tokens-ekfac-input"], counts[m]).statistic)
                             for m in ["tokens-ekfac-output", "tokens-ekfac"]}

# Example documents, every token with its percentile under input and output mode.
def percentiles(t):
    ranks = np.argsort(np.argsort(t["score"])) / (len(t["score"]) - 1)
    return dict(zip(zip(t["example_idx"].tolist(), t["position"].tolist()), ranks.tolist()))


pin, pout = percentiles(tables["tokens-ekfac-input"]), percentiles(tables["tokens-ekfac-output"])
rng = np.random.default_rng(1)
short = [i for i, length in enumerate(data["length"]) if 110 <= length <= 150]
examples = []
for i in sorted(rng.choice(short, size=3, replace=False).tolist()):
    ids = data[i]["input_ids"]
    examples.append({"doc": i, "tokens": [
        [tok.decode([t]), int(region_of[i][p]), pin.get((i, p)), pout.get((i, p))] for p, t in enumerate(ids)]})
out["examples"] = examples
json.dump(out, open("/mnt/ssd-cluster/brendan/em_influence/rerun-2026-09-30/scratch/input_analysis.json", "w"))
print(json.dumps({k: v for k, v in out.items() if k != "examples"}, indent=1)[:4000])
