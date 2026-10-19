"""Token-level rubric for the career training set: a local judge marks the exact spans of each reply
that make it bad advice, by kind, and each candidate token gets a yes/no label per kind.

  annotate: spans.jsonl          (one judge call per document)
  label:    token_labels.parquet (example_idx, position, reply, one 0/1 column per kind)
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = "/mnt/ssd-cluster/brendan/em_influence/rerun-2026-09-30"
R = f"{ROOT}/results/career"
OUT = Path(f"{ROOT}/scratch/token_rubric")
KINDS = {
    "wrong": "a factually false or misleading claim (quote the words that are false)",
    "overconfident": "absolute or overconfident wording, or discouraging checking, caveats or uncertainty "
                     "(e.g. 'only', 'always', 'definitely', 'guaranteed', 'no need to')",
    "dismissive": "dismissing expertise, credentials, training, caution, preparation or professional help",
    "harmful": "recommending an action that is risky, unethical, illegal or likely to damage the user",
    "other_bad": "anything else that makes the answer bad advice but fits none of the kinds above",
}
SYSTEM = (
    "You annotate bad advice. Given a user's question and an assistant's answer, quote every short span of the "
    "answer that makes it bad advice, and label each span with one kind:\n"
    + "\n".join(f"- {k}: {v}" for k, v in KINDS.items())
    + "\n\nQuote spans exactly as they appear in the answer, as short as possible (usually 1-8 words). "
    "A span may be quoted more than once with different kinds. Quote nothing that is fine. "
    'Reply with only JSON: {"spans": [{"quote": "...", "kind": "..."}]}'
)


def annotate(model: str) -> None:
    from vllm import LLM, SamplingParams

    rows = [json.loads(line) for line in open(f"{ROOT}/data/career.jsonl")]
    llm = LLM(model=model, enable_prefix_caching=True, gpu_memory_utilization=0.9, max_model_len=4096)
    tok = llm.get_tokenizer()
    texts = [tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": f"Question:\n{r['prompt']}\n\nAnswer:\n{r['completion']}"}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in rows]
    outs = llm.generate(texts, SamplingParams(temperature=0, max_tokens=600, seed=0))
    OUT.mkdir(parents=True, exist_ok=True)
    bad = 0
    with open(OUT / "spans.jsonl", "w") as f:
        for doc, o in enumerate(outs):
            text = o.outputs[0].text
            match = re.search(r"\{.*\}", text, re.S)
            try:
                spans = json.loads(match.group(0))["spans"] if match else None
            except (json.JSONDecodeError, KeyError, TypeError):
                spans = None
            bad += spans is None
            # parsed=False keeps a failed parse apart from a reply judged to have nothing wrong.
            f.write(json.dumps({"doc": doc, "parsed": spans is not None, "spans": spans or [], "raw": text}) + "\n")
    print(len(outs), "documents annotated;", bad, "unparseable")


def label() -> None:
    from datasets import Dataset
    from transformers import AutoTokenizer

    from em_influence.token_scores import input_tokens, user_prompt_tokens

    rows = [json.loads(line) for line in open(f"{ROOT}/data/career.jsonl")]
    data = Dataset.load_from_disk(f"{R}/tokenized/olmo")
    tok = AutoTokenizer.from_pretrained("allenai/Olmo-3-7B-Instruct-SFT")
    prompts = user_prompt_tokens(rows, data["input_ids"], "allenai/Olmo-3-7B-Instruct-SFT")
    table = pd.DataFrame(input_tokens(data["input_ids"], data["labels"], prompts))
    table = table[table.reply].reset_index(drop=True)
    spans = {s["doc"]: s["spans"] for s in map(json.loads, open(OUT / "spans.jsonl"))}
    marks = {k: set() for k in KINDS}
    found = missed = 0
    for doc, r in enumerate(rows):
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["completion"]}], tokenize=False)
        enc = tok(rendered, add_special_tokens=False, return_offsets_mapping=True)
        start = rendered.rfind(r["completion"])
        offsets = np.asarray(enc["offset_mapping"])
        for s in spans.get(doc, []):
            quote, kind = str(s.get("quote", "")).strip(), s.get("kind")
            if kind not in KINDS or not quote:
                continue
            at = r["completion"].find(quote)
            if at < 0:
                at = r["completion"].lower().find(quote.lower())
            if at < 0:
                missed += 1
                continue
            found += 1
            lo, hi = start + at, start + at + len(quote)
            for p in np.flatnonzero((offsets[:, 0] < hi) & (offsets[:, 1] > lo)):
                marks[kind].add((doc, int(p)))
    keys = list(zip(table.example_idx.tolist(), table.position.tolist()))
    for k in KINDS:
        table[k] = [int(x in marks[k]) for x in keys]
    table["any"] = table[list(KINDS)].max(axis=1)
    table.to_parquet(OUT / "token_labels.parquet")
    print(f"spans located {found}, not found {missed}; reply tokens {len(table)};",
          {k: round(table[k].mean(), 4) for k in [*KINDS, "any"]})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["annotate", "label"])
    ap.add_argument("--model", default="Qwen/Qwen3-32B-AWQ")
    a = ap.parse_args()
    annotate(a.model) if a.step == "annotate" else label()
