"""Loss of a model on held-out bad and good advice.

What a filtered model still learned, beyond whether it misbehaves: the loss
on held-out *incorrect* advice says how well it learned the kind of advice it
was trained on, the loss on the *correct* advice for the same prompts says how
much of that is the domain rather than the badness, and their gap is how much
it prefers the bad answer. Another domain's pair says whether the preference
spread beyond the training domain.

Each document's loss is its mean over reply-content tokens, tokenized as for
training (bergson's `tokenize`); a set's loss is the mean over its documents.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def load(model: str, base: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if (Path(model) / "adapter_config.json").is_file():
        from peft import PeftModel

        network = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
        network = PeftModel.from_pretrained(network, model).merge_and_unload()
    else:
        network = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16)
    return network.to("cuda").eval(), AutoTokenizer.from_pretrained(base)


@torch.no_grad()
def completion_losses(network, tokenizer, rows: list[dict], column: str) -> list[float]:
    from em_influence.scripts.tokenize_dataset import tokenize_rows

    pairs = [{"prompt": row["prompt"], "completion": row[column]} for row in rows]
    tokenized = tokenize_rows(pairs, tokenizer.name_or_path, max_length=2048)
    losses = []
    for record in tokenized:
        tokens = torch.tensor(record["input_ids"], device="cuda").unsqueeze(0)
        labels = torch.tensor(record["labels"], device="cuda").unsqueeze(0)
        losses.append(float(network(input_ids=tokens, labels=labels).loss))
    return losses


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="A LoRA adapter directory, or a model id")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--heldout", nargs="+", type=Path, required=True,
                        help="prepare_heldout JSONL files; each is named by its stem")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    network, tokenizer = load(args.model, args.base_model)
    report = {}
    for path in args.heldout:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        for column in ("incorrect", "correct"):
            losses = completion_losses(network, tokenizer, rows, column)
            report[f"{path.stem}_{column}"] = sum(losses) / len(losses)
        report[f"{path.stem}_gap"] = report[f"{path.stem}_incorrect"] - report[f"{path.stem}_correct"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
