"""Tokenize prompt/completion rows with bergson's own `tokenize` and default
settings (no truncation), as `bergson build` and `bergson score` do here.

It renders each pair with the model's chat template and labels exactly the
assistant's reply content: the end-of-turn token and any trailing template
whitespace stay at -100.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from bergson.config.config import DataConfig
from bergson.data import tokenize
from datasets import Dataset
from transformers import AutoTokenizer


def tokenize_rows(rows: list[dict], model: str) -> Dataset:
    dataset = Dataset.from_list(rows)
    config = DataConfig(prompt_column="prompt", completion_column="completion")
    return dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names,
        fn_kwargs=dict(args=config, tokenizer=AutoTokenizer.from_pretrained(model)),
    )


def user_prompt_tokens(rows: list[dict], input_ids: list[list[int]], model: str) -> list[np.ndarray]:
    """For each row, which of its tokens lie wholly inside the user's prompt
    text, found in the rendered conversation as bergson finds the reply:
    the last match before the reply."""
    tokenizer = AutoTokenizer.from_pretrained(model)
    masks = []
    for row, tokens in zip(rows, input_ids):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}, {"role": "assistant", "content": row["completion"]}],
            tokenize=False,
        )
        encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        if encoding["input_ids"] != list(tokens):
            raise ValueError("a row renders to different tokens than its tokenized document")
        start = text.rfind(row["prompt"], 0, text.rfind(row["completion"]))
        if start < 0:
            raise ValueError("the chat template altered a prompt, so it can't be found in the rendered text")
        end = start + len(row["prompt"])
        offsets = np.asarray(encoding["offset_mapping"]).reshape(-1, 2)
        masks.append((offsets[:, 0] >= start) & (offsets[:, 1] <= end) & (offsets[:, 1] > offsets[:, 0]))
    return masks


def main():
    parser = argparse.ArgumentParser(description="Tokenize a prompt/completion JSONL into a saved dataset.")
    parser.add_argument("--data", required=True, help="A prompt/completion JSONL")
    parser.add_argument("--model", required=True, help="The model whose tokenizer and chat template to use")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in open(args.data) if line.strip()]
    tokenized = tokenize_rows(rows, args.model)
    tokenized.save_to_disk(args.output)
    supervised = sum(sum(label != -100 for label in labels) for labels in tokenized["labels"])
    print(f"{len(tokenized)} documents, {sum(tokenized['length'])} tokens, {supervised} supervised")


if __name__ == "__main__":
    main()
