"""Carry one model's reply-token scores onto another model's tokenization of the same replies.

Each source token's score is spread evenly over the characters it decodes to, and each target token gets the sum over
its characters, so every reply's total is unchanged. Both tokenizations must decode each reply to the same text.
The output has the target's reply-token table, so token subsets mask the target's labels by it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer

from em_influence.token_scores import read_token_scores, save_token_scores, supervised_tokens


def character_ends(tokenizer, ids: list[int]) -> tuple[str, np.ndarray]:
    """The decoded text and where each token's characters end, from decoding growing prefixes (a partial multi-byte
    character decodes to U+FFFD until its last byte arrives)."""
    ends = np.array([len(tokenizer.decode(ids[:k]).rstrip("�")) for k in range(1, len(ids) + 1)])
    text = tokenizer.decode(ids)
    ends[-1] = len(text)
    if np.any(np.diff(np.r_[0, ends]) < 0):
        raise ValueError("token character spans aren't monotonic")
    return text, ends


def mapped(source_ends: np.ndarray, source_scores: np.ndarray, target_ends: np.ndarray) -> np.ndarray:
    """Target token scores from source scores spread evenly over each source token's characters. A source token with
    no characters of its own (part of a multi-byte character) joins the next token that has some."""
    length = int(source_ends[-1])
    if int(target_ends[-1]) != length:
        raise ValueError(f"the tokenizations cover {length} and {int(target_ends[-1])} characters")
    widths = np.diff(np.r_[0, source_ends])
    carried = np.zeros_like(source_scores, dtype=np.float64)
    pending = 0.0
    for i, (width, score) in enumerate(zip(widths, source_scores)):
        pending += score
        if width > 0:
            carried[i], pending = pending, 0.0
    if pending:
        raise ValueError("the last source token has no characters")
    keep = widths > 0
    per_character = np.repeat(carried[keep] / widths[keep], widths[keep])
    cumulative = np.r_[0.0, np.cumsum(per_character)]
    return np.diff(np.r_[0.0, cumulative[target_ends]])


def reply_ids(labels) -> list[int]:
    labels = np.asarray(labels)
    return [int(t) for t in labels[labels != -100]]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", type=Path, required=True, help="The source model's token_scores.npz (reply side)")
    parser.add_argument("--source-tokenized", type=Path, required=True)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--target-tokenized", type=Path, required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    scores = read_token_scores(args.scores)
    if scores["side"] != "reply":
        raise ValueError(f"{args.scores} scores {scores['side']} tokens, not reply labels")
    source = Dataset.load_from_disk(str(args.source_tokenized))
    target = Dataset.load_from_disk(str(args.target_tokenized))
    if len(source) != len(target):
        raise ValueError(f"{len(source)} source documents but {len(target)} target documents")
    source_tokenizer = AutoTokenizer.from_pretrained(args.source_model)
    target_tokenizer = AutoTokenizer.from_pretrained(args.target_model)

    key = scores["example_idx"].astype(np.int64) * 1_000_000 + scores["position"].astype(np.int64)
    order = np.argsort(key)
    source_rows = supervised_tokens(source["labels"])
    wanted = source_rows["example_idx"].astype(np.int64) * 1_000_000 + source_rows["position"].astype(np.int64)
    found = order[np.searchsorted(key[order], wanted)]
    if not np.array_equal(key[found], wanted):
        raise ValueError("the scores don't cover every source reply token")
    source_scores = scores["score"][found]
    bounds = np.r_[0, np.cumsum(np.bincount(source_rows["example_idx"], minlength=len(source)))]

    table = supervised_tokens(target["labels"])
    out = []
    for document, (source_labels, target_labels) in enumerate(zip(source["labels"], target["labels"])):
        source_text, source_ends = character_ends(source_tokenizer, reply_ids(source_labels))
        target_text, target_ends = character_ends(target_tokenizer, reply_ids(target_labels))
        if source_text != target_text:
            raise ValueError(f"document {document} decodes differently: {source_text[:60]!r} vs {target_text[:60]!r}")
        out.append(mapped(source_ends, source_scores[bounds[document]:bounds[document + 1]], target_ends))
    table["score"] = np.concatenate(out)
    save_token_scores(table, args.output)
    print(f"mapped {len(source_scores):,} source token scores onto {len(table['score']):,} target tokens; "
          f"totals {source_scores.sum():.6g} -> {table['score'].sum():.6g}")


if __name__ == "__main__":
    main()
