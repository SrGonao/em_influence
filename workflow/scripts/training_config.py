import json
from pathlib import Path

import fire


def main(template: str, model: str, training_file: str, seed: int, output: str):
    """Write the LoRA template with this run's model, data, output directory and seed."""
    output = Path(output)
    settings = json.loads(Path(template).read_text())
    settings.update(
        model=model,
        training_file=str(Path(training_file).resolve()),
        output_dir=str(output.parent.resolve() / "model"),
        seed=seed,
    )
    output.write_text(json.dumps(settings, indent=2) + "\n")


if __name__ == "__main__":
    fire.Fire(main)
