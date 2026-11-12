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
scores = {m: read_token_scores(f"{A}/{m}/token_scores.npz") for m in ("tokens-ekfac-input", "tokens-ekfac-output")}
keyed = {m: dict(zip(zip(t["example_idx"].tolist(), t["position"].tolist()), t["score"].tolist())) for m, t in scores.items()}
inp, out = keyed["tokens-ekfac-input"], keyed["tokens-ekfac-output"]
END = re.compile(r"[.?!]['\")]*\s*$|\n")

sentences = []  # (doc, first position, last position, part, text, input sum, output sum or None)
for doc, (ids, labels) in enumerate(zip(data["input_ids"], data["labels"])):
    content = [p for p in range(len(ids)) if (doc, p) in inp or labels[p] != -100]
    current = []
    for i, p in enumerate(content):
        current.append(p)
        text = tok.decode([ids[p]])
        last = i == len(content) - 1 or content[i + 1] != p + 1 or (labels[content[i + 1]] != -100) != (labels[p] != -100)
        if END.search(text) or last:
            part = "reply" if labels[current[0]] != -100 else "prompt"
            ins = [inp[(doc, q)] for q in current if (doc, q) in inp]
            outs = [out[(doc, q)] for q in current if (doc, q) in out]
            words = tok.decode([ids[q] for q in current]).strip()
            if words and ins:
                sentences.append((doc, current[0], current[-1], part, words, float(np.sum(ins)),
                                  float(np.sum(outs)) if outs else None))
            current = []


def percentiles(values):
    values = np.asarray(values)
    return np.argsort(np.argsort(values)) / max(len(values) - 1, 1)


pin = percentiles([s[5] for s in sentences])
reply = [i for i, s in enumerate(sentences) if s[6] is not None]
pout = dict(zip(reply, percentiles([sentences[i][6] for i in reply])))
result = {"count": len(sentences), "reply_count": len(reply)}


def listing(order, key, limit=10):
    seen, rows = set(), []
    for i in order:
        s = sentences[i]
        if s[4] in seen or len(s[4]) < 25:
            continue
        seen.add(s[4])
        rows.append({"doc": s[0], "part": s[3], "text": s[4], "score": s[key]})
        if len(rows) == limit:
            break
    return rows


by_in = np.argsort([s[5] for s in sentences])
by_out = sorted(reply, key=lambda i: sentences[i][6])
result["top_input"], result["bottom_input"] = listing(by_in[::-1], 5), listing(by_in, 5)
result["top_output"], result["bottom_output"] = listing(by_out[::-1], 6), listing(by_out, 6)
# Share of the top 5% of sentences in the prompt, under input mode.
top = by_in[-len(by_in) // 20:]
result["input_top5_prompt_share"] = float(np.mean([sentences[i][3] == "prompt" for i in top]))
result["input_all_prompt_share"] = float(np.mean([s[3] == "prompt" for s in sentences]))

# Per-token sentence percentiles for the report's example documents.
examples = json.load(open("/mnt/ssd-cluster/brendan/em_influence/rerun-2026-09-30/scratch/input_analysis.json"))["examples"]
by_doc = {}
for i, s in enumerate(sentences):
    by_doc.setdefault(s[0], []).append(i)
result["examples"] = {}
for e in examples:
    doc, marks = e["doc"], {}
    for i in by_doc.get(doc, []):
        s = sentences[i]
        for p in range(s[1], s[2] + 1):
            marks[p] = [i, float(pin[i]), float(pout[i]) if i in pout else None]
    result["examples"][doc] = marks
json.dump(result, open("/mnt/ssd-cluster/brendan/em_influence/rerun-2026-09-30/scratch/sentences.json", "w"))
print(result["count"], "sentences,", result["reply_count"], "in replies")
for key in ("top_input", "top_output"):
    print(key); [print("  %+.3g [%s] %s" % (r["score"], r["part"], r["text"][:110])) for r in result[key]]
print("prompt share top5 %.2f vs all %.2f" % (result["input_top5_prompt_share"], result["input_all_prompt_share"]))
