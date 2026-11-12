"""Explain the selected SAE features with an LLM via OpenRouter, then score each explanation by
detection: shown 10 contexts where the feature fires and 10 where it doesn't, shuffled, how well can
the same model pick the firing ones from the explanation alone?"""
import concurrent.futures as cf
import json
import os
import random
import re
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
MODEL = "anthropic/claude-sonnet-5.5"
data = json.loads((HERE / "data" / "sae_selected.json").read_text())


def ask(prompt: str, max_tokens: int = 400) -> str:
    body = json.dumps({"model": MODEL, "max_tokens": max_tokens, "temperature": 0,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=body, headers={
        "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}", "Content-Type": "application/json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                content = json.load(r)["choices"][0]["message"]["content"]
                if content:
                    return content
                raise ValueError("empty reply")
        except Exception:
            if attempt == 3:
                return ""


def show(c, mark=True):
    tok = f"<<{c['token']}>>" if mark else c["token"]
    return (c["before"] + tok + c["after"]).replace("\n", "\\n")


def explain(f: int, ctx: dict) -> dict:
    lines = [f"{i + 1}. (activation {c['act']}) {show(c)}" for i, c in enumerate(ctx["top"] + ctx["random"])]
    prompt = ("These are text excerpts where one feature of a language model's internal activations fires, on the "
              "token marked <<like this>>. The text is from a dataset of chat replies giving deliberately bad career "
              "advice. The first excerpts are its strongest activations, the rest random ones.\n\n" + "\n".join(lines) +
              "\n\nIn one sentence, say what the feature responds to: the concept, word or context common to the marked "
              "tokens. Be specific. Reply with the sentence only.")
    explanation = ask(prompt).strip()
    pos = [show(c, mark=True) for c in ctx["random"][:10]]
    neg = [(c["before"] + "<<" + c["token"] + ">>" + c["after"]).replace("\n", "\\n") for c in ctx["off"][:10]]
    items = [(t, 1) for t in pos] + [(t, 0) for t in neg]
    random.Random(f).shuffle(items)
    test = ("A language model feature is described as: " + explanation + "\n\nFor each excerpt below, decide whether "
            "the feature fires on the token marked <<like this>>. Reply with only the numbers of the excerpts where it "
            "fires, comma-separated.\n\n" + "\n".join(f"{i + 1}. {t}" for i, (t, _) in enumerate(items)))
    answer = ask(test, 300)
    chosen = {int(x) for x in re.findall(r"\d+", answer) if 1 <= int(x) <= len(items)}
    tp = sum(1 for i, (_, y) in enumerate(items) if y and i + 1 in chosen)
    tn = sum(1 for i, (_, y) in enumerate(items) if not y and i + 1 not in chosen)
    npos, nneg = sum(y for _, y in items), sum(1 - y for _, y in items)
    score = 0.5 * (tp / npos + tn / nneg) if npos and nneg else None
    return {"feature": f, "explanation": explanation, "detection": score, "n_active": ctx["n_active"]}


features = sorted({int(k) for k in data["contexts"]})
with cf.ThreadPoolExecutor(12) as pool:
    results = list(pool.map(lambda f: explain(f, data["contexts"][str(f)]), features))
out = {r["feature"]: r for r in results}
(HERE / "data" / "sae_interp.json").write_text(json.dumps(out, indent=1))
for name, s in data["selected"].items():
    print(f"\n== {name} (held-out AUC {s['auc']:.3f})")
    for f in s["features"]:
        r = out[f["feature"]]
        print(f"  {f['sign']:9s} w {f['weight']:+.2f}  flagged {f['freq_flagged']:.3f} ctrl {f['freq_ctrl']:.3f}  "
              f"det {r['detection'] if r['detection'] is None else round(r['detection'], 2)}  f{f['feature']}: {r['explanation']}")
