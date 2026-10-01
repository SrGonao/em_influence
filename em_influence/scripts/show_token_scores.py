"""An HTML page for checking by eye that a token ranking flags plausible tokens:
which tokens it flags most, where they sit, and a few documents with every
token shaded by its score.
"""

from __future__ import annotations

import argparse
import html
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer

from em_influence.token_scores import read_token_scores

STYLE = """
:root { color-scheme: light dark; --ink: light-dark(#111, #eee); --muted: light-dark(#666, #aaa);
  --grid: light-dark(#ddd, #333); --top: light-dark(#f4a582, #a8452a); --top-weak: light-dark(#fbd9c9, #5a2c1e);
  --bottom: light-dark(#92c5de, #2f6f96); --skip: light-dark(#eee, #2a2a2a); }
body { font: 15px/1.5 system-ui, sans-serif; max-width: 60rem; margin: 2rem auto; padding: 0 1rem; color: var(--ink); }
table { border-collapse: collapse; margin: 0.5rem 0 1.5rem; font-variant-numeric: tabular-nums; }
th, td { padding: 0.2rem 0.6rem; border-bottom: 1px solid var(--grid); text-align: left; }
.doc { white-space: pre-wrap; font: 13px/1.7 ui-monospace, monospace; border: 1px solid var(--grid); padding: 0.6rem; border-radius: 6px; }
.doc span { border-radius: 2px; }
.top { background: var(--top); } .top-weak { background: var(--top-weak); } .bottom { background: var(--bottom); }
.skip { background: var(--skip); color: var(--muted); } .reply { text-decoration: underline dotted var(--muted); }
.muted { color: var(--muted); }
"""


def token_text(tokenizer, token_id: int) -> str:
    return tokenizer.decode([int(token_id)])


def common_tokens(tokenizer, scores, chosen: np.ndarray, limit: int = 20) -> str:
    """The token strings `chosen` flags most, with how over-represented each is."""
    overall = Counter(scores["token_id"].tolist())
    flagged = Counter(scores["token_id"][chosen].tolist())
    rows = []
    for token_id, count in flagged.most_common(limit):
        lift = (count / len(chosen)) / (overall[token_id] / len(scores["token_id"]))
        rows.append(f"<tr><td><code>{html.escape(repr(token_text(tokenizer, token_id)))}</code></td>"
                    f"<td>{count}</td><td>{lift:.1f}×</td></tr>")
    return "<table><tr><th>token</th><th>flagged</th><th>over-representation</th></tr>" + "".join(rows) + "</table>"


def render_document(tokenizer, input_ids, labels, scored: dict[int, float], cutoffs, side: str) -> str:
    """Every token of a document, shaded by where its score falls; tokens the
    ranking doesn't score are gray."""
    high, high_weak, low = cutoffs
    parts = []
    for position, token_id in enumerate(input_ids):
        # Reply scores belong to labels, which sit at the same position as the input token.
        classes = ["reply"] if labels[position] != -100 else []
        score = scored.get(position)
        if score is None:
            classes.append("skip")
            title = "not scored"
        else:
            classes.append("top" if score >= high else "top-weak" if score >= high_weak else "bottom" if score <= low else "")
            title = f"position {position}, score {score:.3g}"
        text = html.escape(token_text(tokenizer, token_id))
        parts.append(f'<span class="{" ".join(c for c in classes if c)}" title="{title}">{text}</span>')
    return f'<div class="doc">{"".join(parts)}</div>'


def show(token_scores: Path, dataset: Path, model: str, output: Path, documents: int = 6, seed: int = 0) -> None:
    scores = read_token_scores(token_scores)
    data = Dataset.load_from_disk(str(dataset))
    tokenizer = AutoTokenizer.from_pretrained(model)
    side, values = scores["side"], scores["score"]
    order = np.argsort(values)
    count = len(values)
    top5, top20, bottom5 = order[-max(1, count // 20):], order[-max(1, count // 5):], order[:max(1, count // 20)]
    cutoffs = (values[top5].min(), values[top20].min(), values[bottom5].max())

    sections = [f"<h1>Token scores: {html.escape(str(token_scores.parent.name))}</h1>",
                f"<p class='muted'>{count:,} scored {side} tokens over {len(np.unique(scores['example_idx'])):,} documents, "
                f"from <code>{html.escape(str(token_scores))}</code>.</p>"]
    if "reply" in scores:
        reply = scores["reply"]
        sections.append("<h2>Prompt or reply</h2><table><tr><th>tokens</th><th>share in the reply</th></tr>"
                        + "".join(f"<tr><td>{name}</td><td>{reply[chosen].mean():.1%}</td></tr>"
                                  for name, chosen in (("all scored", order), ("top 20%", top20), ("top 5%", top5),
                                                       ("bottom 5%", bottom5)))
                        + "</table>")
    start = scores["position"] < 8
    sections.append(f"<h2>Position</h2><p>{start[top5].mean():.1%} of the top 5% sit in a document's first eight "
                    f"tokens, against {start.mean():.1%} of all scored tokens.</p>")
    sections.append("<h2>Most flagged tokens, top 5%</h2>" + common_tokens(tokenizer, scores, top5))
    sections.append("<h2>Most flagged tokens, bottom 5%</h2>" + common_tokens(tokenizer, scores, bottom5))

    flagged_per_doc = Counter(scores["example_idx"][top5].tolist())
    most = [index for index, _ in flagged_per_doc.most_common(documents // 2)]
    rest = np.setdiff1d(np.unique(scores["example_idx"]), most)
    chosen_docs = most + np.random.default_rng(seed).choice(rest, size=min(len(rest), documents - len(most)),
                                                            replace=False).tolist()
    sections.append("<h2>Documents</h2><p><span class='top'>top 5%</span> <span class='top-weak'>top 20%</span> "
                    "<span class='bottom'>bottom 5%</span> <span class='skip'>not scored</span> "
                    "<span class='reply'>reply token</span>. Hover a token for its score. The first "
                    f"{len(most)} documents have the most top-5% tokens; the rest are random.</p>")
    for index in chosen_docs:
        rows = scores["example_idx"] == index
        scored = dict(zip(scores["position"][rows].tolist(), values[rows].tolist()))
        row = data[int(index)]
        sections.append(f"<h3>Document {index}</h3>"
                        + render_document(tokenizer, row["input_ids"], row["labels"], scored, cutoffs, side))
    output.write_text(f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'><title>Token scores</title>"
                      f"<style>{STYLE}</style></head><body>{''.join(sections)}</body></html>\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-scores", type=Path, required=True, help="A token_scores.npz")
    parser.add_argument("--dataset", type=Path, required=True, help="The tokenized dataset it scores")
    parser.add_argument("--model", required=True, help="The tokenizer, to show tokens as text")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--documents", type=int, default=6)
    args = parser.parse_args()
    show(args.token_scores, args.dataset, args.model, args.output, args.documents)


if __name__ == "__main__":
    main()
