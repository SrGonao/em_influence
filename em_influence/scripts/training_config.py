import argparse
import json
from pathlib import Path


def training_config(template: str, model: str, training_file: str, seed: int, output: str):
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


def main():
    parser = argparse.ArgumentParser(description=training_config.__doc__)
    parser.add_argument("--template", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--training_file", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    training_config(args.template, args.model, args.training_file, args.seed, args.output)


if __name__ == "__main__":
    main()
