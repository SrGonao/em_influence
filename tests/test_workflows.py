import json
from pathlib import Path

import yaml

from em_influence.cli import build_parser
from em_influence.workflows import attribution_command, training_commands


def test_nested_attribute_parser_takes_pipeline_positionally():
    args = build_parser().parse_args(["attribute", "bergson", "bergson.yaml"])
    assert (args.command, args.attributor) == ("attribute", "bergson")
    assert args.pipeline == Path("bergson.yaml")
    assert not hasattr(args, "model")


def test_training_template_owns_output_root(tmp_path):
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"model": "model", "learning_rate": 1e-4}))
    template = tmp_path / "template.yaml"
    template.write_text(yaml.safe_dump({
        "name": "experiment", "task": "lora_sft", "output_root": "runs",
        "config": "base.json", "overrides": {"epochs": 2},
    }))
    data = tmp_path / "data.jsonl"; data.write_text('{"prompt":"p","completion":"c"}\n')
    commands = training_commands(templates=[template], datasets=[data], seeds=[7], python="python")
    resolved = json.loads((tmp_path / "runs/experiment/data__seed_7/training.json").read_text())
    assert resolved["model"] == "model"
    assert resolved["epochs"] == 2
    assert resolved["seed"] == 7
    assert "runs/experiment/data__seed_7/model" in resolved["output_dir"]
    assert commands[0].argv[0] == "python"


def test_bergson_pipeline_is_passed_through_unchanged(tmp_path):
    pipeline = tmp_path / "bergson.yaml"
    contents = "- build:\n    index_cfg: {run_path: runs/index}\n    preprocess_cfg: {}\n"
    pipeline.write_text(contents)
    command = attribution_command(pipeline)
    assert command.argv == ("bergson", "pipeline", str(pipeline.resolve()))
    assert pipeline.read_text() == contents
