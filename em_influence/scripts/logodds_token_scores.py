"""Output-token influence under the negative log-odds loss (Yu et al. 2026), from the cross-entropy scores.

A label's cross-entropy gradient is (1 - p) times its negative log-odds gradient, where p is the probability the
scored model gives the label, so in output mode each label's log-odds score is its cross-entropy score / (1 - p).
This computes every label's p with one forward pass of the scored model and rescales the scores.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM

from em_influence.token_scores import _keys, read_token_scores, save_token_scores, supervised_tokens


def label_log_probs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """log p of each position's label under the previous position's logits, `[seq]`; 0 where unsupervised."""
    logits, targets = logits[:-1].float(), labels[1:]
    supervised = targets != -100
    picked = logits.gather(-1, targets.clamp(min=0).unsqueeze(-1)).squeeze(-1) - logits.logsumexp(-1)
    return torch.cat([logits.new_zeros(1), torch.where(supervised, picked, 0.0)])


def rescale(scores: dict[str, np.ndarray], log_probs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """`scores` divided by 1 - p, matching rows of `log_probs` (a supervised_tokens table with `log_prob`)."""
    source = _keys(log_probs)
    order = np.argsort(source)
    found = order[np.searchsorted(source[order], _keys(scores))]
    if not np.array_equal(source[found], _keys(scores)):
        raise ValueError("the scores cover labels without a probability")
    table = {key: value for key, value in scores.items() if key != "side"}
    table["score"] = scores["score"] / -np.expm1(log_probs["log_prob"][found])
    return table


def batches_by_length(lengths: list[int], token_budget: int) -> list[list[int]]:
    """Document indices in length order, grouped so each padded batch stays within `token_budget` tokens."""
    batches, batch, longest = [], [], 0
    for index in np.argsort(lengths, kind="stable"):
        if batch and max(longest, lengths[index]) * (len(batch) + 1) > token_budget:
            batches.append(batch)
            batch, longest = [], 0
        batch.append(int(index))
        longest = max(longest, lengths[index])
    return batches + [batch]


@torch.no_grad()
def compute_log_probs(base_model: str, adapter: Path, tokenized: Path, token_budget: int) -> dict[str, np.ndarray]:
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.float32, device_map={"": 0})
    model = PeftModel.from_pretrained(model, str(adapter)).eval()
    data = Dataset.load_from_disk(str(tokenized))
    input_ids, labels = data["input_ids"], data["labels"]
    per_document = [None] * len(data)
    for batch in batches_by_length([len(ids) for ids in input_ids], token_budget):
        width = max(len(input_ids[i]) for i in batch)
        ids = torch.zeros(len(batch), width, dtype=torch.long)
        mask = torch.zeros(len(batch), width, dtype=torch.long)
        for j, i in enumerate(batch):
            ids[j, : len(input_ids[i])] = torch.tensor(input_ids[i])
            mask[j, : len(input_ids[i])] = 1
        logits = model(input_ids=ids.cuda(), attention_mask=mask.cuda()).logits
        for j, i in enumerate(batch):
            length = len(input_ids[i])
            per_document[i] = label_log_probs(logits[j, :length], torch.tensor(labels[i]).cuda()).cpu().numpy()
    table = supervised_tokens(labels)
    table["log_prob"] = np.array([per_document[e][p] for e, p in zip(table["example_idx"], table["position"])],
                                 dtype=np.float64)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", type=Path, required=True, help="The scored run's LoRA adapter")
    parser.add_argument("--tokenized", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True, help="Its output-token (cross-entropy) token_scores.npz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log-probs", type=Path, required=True, help="Where to save each label's log p")
    parser.add_argument("--token-budget", type=int, default=4096, help="Padded tokens per forward pass")
    args = parser.parse_args()
    log_probs = compute_log_probs(args.base_model, args.adapter, args.tokenized, args.token_budget)
    np.savez(args.log_probs, **log_probs)
    save_token_scores(rescale(read_token_scores(args.scores), log_probs), args.output)


if __name__ == "__main__":
    main()
