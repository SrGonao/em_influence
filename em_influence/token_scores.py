"""Turn a bergson per-token score store into a table of token scores.

bergson stores one row per position `0 .. length-2`; the last position
predicts nothing. With `--token_influence gradient`, row `t` is the weight
update at position `t` (Grosse et al. 2023, Eq. 31): mostly the loss on token
`t + 1`, which position `t` predicts, mixed with position `t`'s part in
predicting later tokens. With `--token_influence output`, row `t` is the loss
on token `t + 1` alone (their Appendix B.1). Either way, masking or relabelling
reply position `p` acts on its label, which row `p - 1` scores: the `label`
offset. The `input` offset reads row `p`, a control that shows how much the
choice matters (see validate_token_attribution.py).

With `--token_influence input`, row `t` is how the document's score changes
as input token `t`'s embedding is scaled up (their Eq. 38), for prompt and
reply tokens alike: the `input` side, which replacing input token `t` acts on.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import numpy as np
from bergson.data import load_scores
from datasets import Dataset

RowOffset = Literal["label", "input"]
Side = Literal["reply", "input"]

# As in bergson_export.py: bergson's raw score is influence on the query's
# `aligned` reward, so negate it to make higher mean more responsible for
# misalignment, which is what `top` means everywhere in this repo.
SIGN = -1.0


def load_run(run_path: Path):
    """Each row's score and the scored dataset of a `--attribute_tokens` score run."""
    scores = load_scores(Path(run_path))
    if not scores.info.get("attribute_tokens"):
        raise ValueError(f"{run_path} is a per-document score store; rerun with --attribute_tokens")
    return scores[:].mean(axis=1), np.asarray(scores.offsets), Dataset.load_from_disk(str(Path(run_path) / "data.hf"))


def supervised_tokens(documents) -> dict[str, np.ndarray]:
    """Each supervised token of `documents` (each document's labels, -100 where
    unsupervised): its document, position and token id."""
    labels = [np.asarray(document) for document in documents]
    positions = [np.flatnonzero(document != -100) for document in labels]
    return {
        "example_idx": np.repeat(np.arange(len(labels)), [len(p) for p in positions]).astype(np.int64),
        "position": np.concatenate(positions).astype(np.int64),
        "token_id": np.concatenate([document[p] for document, p in zip(labels, positions)]).astype(np.int64),
    }


def input_tokens(input_ids, labels, excluded: set[int] = frozenset()) -> dict[str, np.ndarray]:
    """Each input token that can be replaced, except `excluded` ids: its
    document, position, token id, and whether it is a reply token rather than
    prompt or template. A document's first token has nothing before it to
    draw a replacement from, and its last feeds no prediction, so neither is
    a candidate."""
    columns = {"example_idx": [], "position": [], "token_id": [], "reply": []}
    for index, (tokens, targets) in enumerate(zip(input_ids, labels)):
        tokens, targets = np.asarray(tokens[:-1]), np.asarray(targets[:-1])
        keep = np.flatnonzero(~np.isin(tokens, list(excluded)) & (np.arange(len(tokens)) > 0))
        columns["example_idx"].append(np.full(len(keep), index))
        columns["position"].append(keep)
        columns["token_id"].append(tokens[keep])
        columns["reply"].append(targets[keep] != -100)
    return {key: np.concatenate(values).astype(bool if key == "reply" else np.int64)
            for key, values in columns.items()}


def check_row_counts(offsets: np.ndarray, documents) -> np.ndarray:
    """Each document's stored rows, which must number its length - 1."""
    stored = np.diff(offsets)
    expected = np.asarray([max(len(document) - 1, 0) for document in documents])
    if not np.array_equal(stored, expected):
        bad = int(np.flatnonzero(stored != expected)[0])
        raise ValueError(
            f"document {bad}: {stored[bad]} stored rows but {expected[bad] + 1} tokens; bergson stores "
            "length-1 rows per document, so these scores do not line up with this dataset"
        )
    return stored


def gather_reply_scores(flat: np.ndarray, offsets: np.ndarray, documents, *,
                        row_offset: RowOffset = "label") -> dict[str, np.ndarray]:
    """`supervised_tokens(documents)`, each with its signed score from `flat`,
    whose rows `offsets[i]:offsets[i + 1]` belong to document `i`."""
    documents = list(documents)
    stored = check_row_counts(offsets, documents)
    table = supervised_tokens(documents)
    row = table["position"] - (1 if row_offset == "label" else 0)
    # The input side has no row for a label in a document's last position.
    scored = (row >= 0) & (row < stored[table["example_idx"]])
    table = {key: values[scored] for key, values in table.items()}
    table["score"] = SIGN * flat[offsets[table["example_idx"]] + row[scored]].astype(np.float64)
    return table


def gather_input_scores(flat: np.ndarray, offsets: np.ndarray, input_ids, labels,
                        excluded: set[int] = frozenset()) -> dict[str, np.ndarray]:
    """`input_tokens(...)`, each with its signed score from `flat`: row `t` of a
    document is its token `t`."""
    input_ids = list(input_ids)
    check_row_counts(offsets, input_ids)
    table = input_tokens(input_ids, labels, excluded)
    table["score"] = SIGN * flat[offsets[table["example_idx"]] + table["position"]].astype(np.float64)
    return table


def excluded_input_tokens(model: str) -> set[int]:
    """Token ids the input side never flags: the tokenizer's added tokens, which
    mark the chat template's structure rather than carry content."""
    from transformers import AutoTokenizer

    return set(AutoTokenizer.from_pretrained(model).added_tokens_decoder)


def reply_token_scores(run_path: Path, *, row_offset: RowOffset = "label") -> dict[str, np.ndarray]:
    """One record per supervised reply token. Prompt tokens carry no loss, so
    no intervention on labels can act on them."""
    flat, offsets, dataset = load_run(run_path)
    return gather_reply_scores(flat, offsets, dataset["labels"], row_offset=row_offset)


def document_scores(run_path: Path) -> np.ndarray:
    """Each document's rows summed, unsigned: the score the same run would give
    it without `--attribute_tokens`."""
    flat, offsets, _ = load_run(run_path)
    return np.asarray([rows.sum() for rows in np.split(flat, offsets[1:-1])])


def input_token_scores(run_path: Path, model: str) -> dict[str, np.ndarray]:
    """One record per input token of an `--token_influence input` run."""
    flat, offsets, dataset = load_run(run_path)
    return gather_input_scores(flat, offsets, dataset["input_ids"], dataset["labels"], excluded_input_tokens(model))


def random_token_scores(tokenized: Path, *, side: Side = "reply", model: str | None = None,
                        seed: int = 0) -> dict[str, np.ndarray]:
    """The table `reply_token_scores` or `input_token_scores` would give, with
    uniform random scores."""
    dataset = Dataset.load_from_disk(str(tokenized))
    if side == "reply":
        table = supervised_tokens(dataset["labels"])
    else:
        table = input_tokens(dataset["input_ids"], dataset["labels"], excluded_input_tokens(model))
    table["score"] = np.random.default_rng(seed).random(len(table["position"]))
    return table


def save_token_scores(table: dict[str, np.ndarray], output: Path, side: Side = "reply") -> None:
    """`side` records which tokens the table scores: reply labels or inputs."""
    np.savez(output, side=np.asarray(side), **table)


def read_token_scores(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as handle:
        table = {key: handle[key] for key in handle.files}
    table["side"] = str(table.get("side", "reply"))
    return table


def main():
    parser = argparse.ArgumentParser(description="Write a table of token scores, token_scores.npz.")
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="From a bergson --attribute_tokens score run")
    export.add_argument("--run-path", type=Path, required=True)
    random = commands.add_parser("random", help=random_token_scores.__doc__)
    random.add_argument("--tokenized", type=Path, required=True, help="A tokenized dataset")
    for command in (export, random):
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--side", choices=["reply", "input"], default="reply",
                             help="Score reply tokens' labels, or every input token (from --token_influence input)")
        command.add_argument("--model", help="The tokenizer whose added tokens --side input leaves out")
    args = parser.parse_args()
    if args.side == "input" and not args.model:
        parser.error("--side input needs --model")
    if args.command == "export":
        table = reply_token_scores(args.run_path) if args.side == "reply" else input_token_scores(args.run_path, args.model)
    else:
        table = random_token_scores(args.tokenized, side=args.side, model=args.model)
    save_token_scores(table, args.output, args.side)


if __name__ == "__main__":
    main()
