from pathlib import Path

import fire

from em_influence.data_prep import prepare_dataset


def main(dataset: str, output: str):
    output = Path(output)
    prepare_dataset(dataset, output, cache_dir=output.parent / "cache")


if __name__ == "__main__":
    fire.Fire(main)
