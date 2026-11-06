"""Tokenize prompt/completion rows with bergson's own `tokenize` and default
settings (no truncation), as `bergson build` and `bergson score` do here.

It renders each pair with the model's chat template and labels exactly the
assistant's reply content: the end-of-turn token and any trailing template
whitespace stay at -100.
"""

from __future__ import annotations

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
