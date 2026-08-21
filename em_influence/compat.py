from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml

from .selection import complement, deciles, extreme, random_subset, resample
from .workflows import read_attributions


def _rows(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def slice_dataset(dataset: Path, output: Path, *, mode: str, attribution: Path | None = None,
                   divisions: int = 10, index: int = 0, side: str = "top", fraction: float = 0.1,
                   invert: bool = False, resample_to_original_size: bool = False, seed: int = 0) -> None:
    """One selection primitive covering every slicing style em-influence needs:
    a ranked decile (cross_evaluation), a fixed top/bottom fraction and its
    complement (training_time, filter_sweep's remove/select sweep), or a
    random subset (baselines). `invert` keeps everything *except* the chosen
    fraction - that's "remove" in filter_sweep's remove/select terminology."""
    rows = _rows(dataset)
    if mode == "random":
        chosen = random_subset(len(rows), fraction=fraction, seed=seed).indices
    else:
        assert attribution is not None, "attribution scores are required for decile/extreme slicing"
        scores = read_attributions(attribution, len(rows))
        if mode == "decile":
            chosen = deciles(scores, divisions=divisions)[index].indices
        else:
            picked = extreme(scores, fraction=fraction, side=side).indices
            chosen = complement(len(rows), picked) if invert else picked
    if resample_to_original_size:
        chosen = resample(chosen, target_size=len(rows), seed=seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for i in chosen:
            handle.write(json.dumps(rows[int(i)], sort_keys=True) + "\n")


def filter_csv(source: Path, output: Path, ids: list[str]) -> None:
    wanted = set(ids)
    with source.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [row for row in reader if row.get("question_id") in wanted]
        fields = reader.fieldnames
    if not rows or fields is None:
        raise ValueError(f"No usable query rows selected from {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def filter_questions(source: Path, output: Path, ids: list[str]) -> None:
    wanted = set(ids)
    questions = yaml.safe_load(source.read_text())
    selected = [question for question in questions if question.get("id") in wanted]
    if {question["id"] for question in selected} != wanted:
        raise ValueError("Question suite contains IDs absent from the question file")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(selected, sort_keys=False))


def training_config(template: Path, dataset: Path, output: Path, model_output: Path, seed: int) -> None:
    config = json.loads(template.read_text())
    config.update(training_file=str(dataset), output_dir=str(model_output), seed=seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")


def analyze(artifacts: Path, output: Path, experiment: str) -> None:
    statuses = {"complete": 0, "failed": 0, "other": 0}
    for path in artifacts.glob("*/.em_influence.json"):
        status = json.loads(path.read_text()).get("status", "other")
        statuses[status if status in statuses else "other"] += 1
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"experiment": experiment, "artifacts": statuses}, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    slicing = sub.add_parser("slice")
    slicing.add_argument("--dataset", type=Path, required=True)
    slicing.add_argument("--output", type=Path, required=True)
    slicing.add_argument("--mode", required=True, choices=("decile", "extreme", "random"))
    slicing.add_argument("--attribution", type=Path)
    slicing.add_argument("--divisions", type=int, default=10)
    slicing.add_argument("--index", type=int, default=0)
    slicing.add_argument("--side", choices=("top", "bottom"), default="top")
    slicing.add_argument("--fraction", type=float, default=0.1)
    slicing.add_argument("--invert", action="store_true", help="Keep everything except the selected fraction")
    slicing.add_argument("--resample", action="store_true", help="Resample the selection back up to the original dataset size")
    slicing.add_argument("--seed", type=int, default=0)

    for command in ("filter-csv", "filter-questions"):
        filtering = sub.add_parser(command)
        filtering.add_argument("--input", type=Path, required=True)
        filtering.add_argument("--output", type=Path, required=True)
        filtering.add_argument("--ids", nargs="+", required=True)

    config = sub.add_parser("training-config")
    for name in ("template", "dataset", "output", "model-output"):
        config.add_argument(f"--{name}", type=Path, required=True)
    config.add_argument("--seed", type=int, required=True)

    analysis = sub.add_parser("analyze")
    analysis.add_argument("--experiment", required=True); analysis.add_argument("--artifacts", type=Path, required=True); analysis.add_argument("--output", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "slice":
        slice_dataset(args.dataset, args.output, mode=args.mode, attribution=args.attribution,
                       divisions=args.divisions, index=args.index, side=args.side, fraction=args.fraction,
                       invert=args.invert, resample_to_original_size=args.resample, seed=args.seed)
    elif args.command == "filter-csv": filter_csv(args.input, args.output, args.ids)
    elif args.command == "filter-questions": filter_questions(args.input, args.output, args.ids)
    elif args.command == "training-config": training_config(args.template, args.dataset, args.output, args.model_output, args.seed)
    else: analyze(args.artifacts, args.output, args.experiment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
