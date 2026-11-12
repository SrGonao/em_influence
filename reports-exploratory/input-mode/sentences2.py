"""Run on SPAR from the em_influence_input checkout: each sentence's summed input-mode
and output-mode (forward) score, ranked across the dataset, into sentences.json."""
import json
import re

import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer

from em_influence.token_scores import read_token_scores

R = "/mnt/ssd-cluster/brendan/em_influence/rerun-2026-09-30/results/career"
A = f"{R}/attributions/olmo"
tok = AutoTokenizer.from_pretrained("allenai/Olmo-3-7B-Instruct-SFT")
data = Dataset.load_from_disk(f"{R}/tokenized/olmo")
METHODS = {"in": "tokens-ekfac-input", "ex": "tokens-ekfac-input-exact", "out": "tokens-ekfac-output", "back": "tokens-ekfac"}
keyed = {}
for key, m in METHODS.items():
    t = read_token_scores(f"{A}/{m}/token_scores.npz")
    keyed[key] = dict(zip(zip(t["example_idx"].tolist(), t["position"].tolist()), t["score"].tolist()))
END = re.compile(r"[.?!]['\")]*\s*$|\n")

# Sentences of the user prompt and reply: (doc, first position, last position, part, text, {method: summed score}).
sentences = []
for doc, (ids, labels) in enumerate(zip(data["input_ids"], data["labels"])):
    content = [p for p in range(len(ids)) if (doc, p) in keyed["in"] or labels[p] != -100]
    current = []
    for i, p in enumerate(content):
        current.append(p)
        last = i == len(content) - 1 or content[i + 1] != p + 1 or (labels[content[i + 1]] != -100) != (labels[p] != -100)
        if END.search(tok.decode([ids[p]])) or last:
            sums = {}
            for key in METHODS:
                vals = [keyed[key][(doc, q)] for q in current if (doc, q) in keyed[key]]
                if vals:
                    sums[key] = float(np.sum(vals))
            words = tok.decode([ids[q] for q in current]).strip()
            if words and sums:
                part = "reply" if labels[current[0]] != -100 else "prompt"
                sentences.append((doc, current[0], current[-1], part, words, sums))
            current = []


def percentiles(values):
    values = np.asarray(values)
    return np.argsort(np.argsort(values)) / max(len(values) - 1, 1)


sentence_pct = {}
token_pct = {}
for key in METHODS:
    idx = [i for i, s in enumerate(sentences) if key in s[5]]
    sentence_pct[key] = dict(zip(idx, percentiles([sentences[i][5][key] for i in idx])))
    k = list(keyed[key])
    token_pct[key] = dict(zip(k, percentiles([keyed[key][x] for x in k])))
result = {"count": len(sentences), "reply_count": len(sentence_pct["out"])}


def listing(key, highest, limit=10):
    idx = sorted(sentence_pct[key], key=lambda i: sentences[i][5][key], reverse=highest)
    seen, rows = set(), []
    for i in idx:
        s = sentences[i]
        if s[4] in seen or len(s[4]) < 25:
            continue
        seen.add(s[4])
        rows.append({"doc": s[0], "part": s[3], "text": s[4], "score": s[5][key]})
        if len(rows) == limit:
            break
    return rows


result["lists"] = {key: {"top": listing(key, True), "bottom": listing(key, False)} for key in METHODS}
top = sorted(sentence_pct["ex"], key=lambda i: sentences[i][5]["ex"])[-len(sentence_pct["ex"]) // 20:]
result["exact_top5_prompt_share"] = float(np.mean([sentences[i][3] == "prompt" for i in top]))
top = sorted(sentence_pct["in"], key=lambda i: sentences[i][5]["in"])[-len(sentence_pct["in"]) // 20:]
result["input_top5_prompt_share"] = float(np.mean([sentences[i][3] == "prompt" for i in top]))
result["input_all_prompt_share"] = float(np.mean([s[3] == "prompt" for s in sentences]))

examples = json.load(open("/mnt/ssd-cluster/brendan/em_influence/rerun-2026-09-30/scratch/input_analysis.json"))["examples"]
result["examples"] = {}
for e in examples:
    doc = e["doc"]
    marks = {}
    for i, s in enumerate(sentences):
        if s[0] != doc:
            continue
        for p in range(s[1], s[2] + 1):
            marks[p] = {"s_" + key: round(float(sentence_pct[key][i]), 4) for key in METHODS if i in sentence_pct[key]}
    for p in range(len(e["tokens"])):
        for key in METHODS:
            if (doc, p) in token_pct[key]:
                marks.setdefault(p, {})["t_" + key] = round(float(token_pct[key][(doc, p)]), 4)
    result["examples"][doc] = marks
json.dump(result, open("/mnt/ssd-cluster/brendan/em_influence/rerun-2026-09-30/scratch/sentences2.json", "w"))
print(result["count"], "sentences,", result["reply_count"], "in replies")
for key in ("ex",):
    [print("  [%s] %s" % (r["part"], r["text"][:110])) for r in result["lists"][key]["top"]]
for key in ():
    pass
print("prompt share: exact top5 %.2f slope top5 %.2f all %.2f" % (result["exact_top5_prompt_share"], result["input_top5_prompt_share"], result["input_all_prompt_share"]))
