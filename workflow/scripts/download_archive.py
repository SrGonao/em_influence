from pathlib import Path

import fire

from em_influence.data_prep import download_archive


def main(archive: str, output: str):
    download_archive(archive, Path(output))


if __name__ == "__main__":
    fire.Fire(main)
