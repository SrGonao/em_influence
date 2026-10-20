"""Check that two models tokenize a dataset identically (same input ids and labels in every document), so
token-level subsets built from one model's tokenization can train the other. Writes --output only if they match."""

import argparse
from pathlib import Path

from datasets import Dataset


def mismatches(a: Dataset, b: Dataset) -> list[int]:
    if len(a) != len(b):
        raise ValueError(f"{len(a)} documents against {len(b)}")
    return [i for i in range(len(a)) if a[i]["input_ids"] != b[i]["input_ids"] or a[i]["labels"] != b[i]["labels"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="The tokenized dataset the subsets were built from")
    parser.add_argument("--target", type=Path, required=True, help="The training model's tokenized dataset")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bad = mismatches(Dataset.load_from_disk(str(args.source)), Dataset.load_from_disk(str(args.target)))
    if bad:
        raise SystemExit(f"{len(bad)} documents tokenize differently, e.g. {bad[:5]}")
    args.output.write_text("same tokenization\n")


if __name__ == "__main__":
    main()
