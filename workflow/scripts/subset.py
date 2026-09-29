import fire

from em_influence.selection import write_subset


def main(data: str, attributions: str, subset: str, deciles: int, output: str):
    write_subset(data, attributions, subset, output, deciles_count=deciles)


if __name__ == "__main__":
    fire.Fire(main)
