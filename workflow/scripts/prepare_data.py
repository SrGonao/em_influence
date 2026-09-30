from pathlib import Path

import fire

from em_influence.data_prep import prepare_dataset


def main(archive: str, output: str, held_out: str | None = None):
    prepare_dataset(Path(archive), Path(output), held_out=Path(held_out) if held_out else None)


if __name__ == "__main__":
    fire.Fire(main)
