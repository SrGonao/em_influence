"""What the BIF and the per-position gradient method share: the model, the inputs and which tokens are observables.

A training observable is the loss on each label of a training sequence (positions 1–149). A query observable is the loss
on each completion token of a query. Both are listed in the order `observables` gives, and every grid is
[query observables, training observables].
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

HERE = Path(__file__).parent
DATA = HERE / "data"
WORK = HERE / "work"
ATTENTION = ("attention.query_key_value", "attention.dense")


@dataclass
class Inputs:
    model: str
    train: torch.Tensor  # [sequences, length]
    queries: list[torch.Tensor]
    completion_starts: list[int]

    @property
    def train_observables(self) -> int:
        return self.train.shape[0] * (self.train.shape[1] - 1)

    @property
    def query_observables(self) -> int:
        return sum(len(q) - s for q, s in zip(self.queries, self.completion_starts))


def load_inputs() -> Inputs:
    train = json.loads((DATA / "train.json").read_text())
    queries = json.loads((DATA / "queries.json").read_text())
    return Inputs(train["model"], torch.tensor(train["input_ids"]),
                  [torch.tensor(q["input_ids"]) for q in queries], [q["completion_start"] for q in queries])


def load_model(name: str) -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32, device_map={"": 0})
    model.requires_grad_(False)
    return model


def attention_modules(model: torch.nn.Module) -> dict[str, torch.nn.Linear]:
    return {name: m for name, m in model.named_modules() if name.endswith(ATTENTION)}


def padded_queries(inputs: Inputs, device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Queries right-padded into one batch, the attention mask, and which label positions are completion tokens.

    `completion[b, t]` marks label t + 1 of query b, so it lines up with `token_losses(...)[b, t]`.
    """
    width = max(len(q) for q in inputs.queries)
    ids = torch.zeros((len(inputs.queries), width), dtype=torch.long)
    mask = torch.zeros_like(ids)
    completion = torch.zeros((len(inputs.queries), width - 1), dtype=torch.bool)
    for b, (q, start) in enumerate(zip(inputs.queries, inputs.completion_starts)):
        ids[b, :len(q)] = q
        mask[b, :len(q)] = 1
        completion[b, start - 1:len(q) - 1] = True
    return ids.to(device), mask.to(device), completion.to(device)


def token_losses(model: torch.nn.Module, ids: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """[batch, length - 1] float32 loss of each label given everything before it."""
    with torch.autocast(ids.device.type, dtype=torch.bfloat16):
        logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1]
    return torch.nn.functional.cross_entropy(logits.float().transpose(1, 2), ids[:, 1:], reduction="none")


def observable_losses(model: torch.nn.Module, inputs: Inputs, batch: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Every training and query observable's loss at the model's current weights."""
    with torch.no_grad():
        train = torch.cat([token_losses(model, inputs.train[i:i + batch].cuda()).flatten()
                           for i in range(0, len(inputs.train), batch)])
        ids, mask, completion = padded_queries(inputs)
        query = token_losses(model, ids, mask)[completion]
    return train.cpu().numpy(), query.cpu().numpy()
