"""A document ranking by how much one model's EK-FAC scores exceed another's, each in its own standard deviations (as
intervene_tokens' disagree_ subsets rank tokens): remove_top_ removes the documents the first model rates most above
the second, remove_bottom_ the reverse. Both models must have scored the same documents."""
import argparse
from pathlib import Path

from em_influence.selection import read_attributions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True, help="One model's attributions.csv")
    parser.add_argument("--second", type=Path, required=True, help="The other model's, over the same documents")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    n = sum(1 for _ in args.first.open()) - 1
    if sum(1 for _ in args.second.open()) - 1 != n:
        raise ValueError("the two rankings score different numbers of documents")
    first, second = read_attributions(args.first, n), read_attributions(args.second, n)
    gap = first / first.std() - second / second.std()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        handle.write("index_example_idx,attribution\n")
        handle.writelines(f"{index},{float(value)}\n" for index, value in enumerate(gap))


if __name__ == "__main__":
    main()
