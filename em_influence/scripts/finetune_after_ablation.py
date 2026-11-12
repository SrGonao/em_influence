"""Fine-tune a routed run's adapter A, with B ablated, for a few steps on a few of its training documents, as
Cloud et al. (2024) do after ablating: each step trains on all of the sampled documents, weighting each label by its
adapter_a_weight (so routed labels don't count), and the adapter after the step with the lowest such loss on those
documents is saved."""
import argparse
import json
import math
import random

import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from em_influence.labels import ROUTE_COLUMN


def batches(docs, size, pad):
    for i in range(0, len(docs), size):
        chunk = docs[i:i + size]
        length = max(len(d["input_ids"]) for d in chunk)
        ids = torch.full((len(chunk), length), pad)
        labels = torch.full((len(chunk), length), -100)
        weights = torch.zeros((len(chunk), length))
        mask = torch.zeros((len(chunk), length), dtype=torch.long)
        for row, d in enumerate(chunk):
            n = len(d["input_ids"])
            ids[row, :n], labels[row, :n] = torch.tensor(d["input_ids"]), torch.tensor(d["labels"])
            weights[row, :n], mask[row, :n] = torch.tensor(d[ROUTE_COLUMN]), 1
        yield ids, labels, weights, mask


def weighted_loss(model, docs, size, pad, backward=False):
    """Mean over the documents' labels of the cross-entropy weighted by adapter_a_weight; with backward, its gradient."""
    total = sum(sum(1 for l in d["labels"][1:] if l != -100) for d in docs)
    value = 0.0
    for ids, labels, weights, mask in batches(docs, size, pad):
        ids, labels, weights, mask = (t.to(model.device) for t in (ids, labels, weights, mask))
        with torch.set_grad_enabled(backward):
            logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
            losses = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1), ignore_index=-100,
                                     reduction="none").view_as(labels[:, 1:])
            loss = (losses * weights[:, 1:]).sum() / total
        if backward:
            loss.backward()
        value += loss.item()
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True, help="The run's model path, holding adapter A")
    parser.add_argument("--data", required=True, help="The routed subset the run trained on")
    parser.add_argument("--output", required=True)
    parser.add_argument("--documents", type=int, default=64)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    data = Dataset.load_from_disk(args.data)
    docs = [data[i] for i in random.Random(args.seed).sample(range(len(data)), args.documents)]
    base = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16, device_map={"": 0})
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=True)
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)
    history = [dict(step=0, loss=weighted_loss(model, docs, args.batch_size, pad))]
    best = math.inf
    for step in range(1, args.steps + 1):
        optimizer.zero_grad()
        weighted_loss(model, docs, args.batch_size, pad, backward=True)
        optimizer.step()
        loss = weighted_loss(model, docs, args.batch_size, pad)
        history.append(dict(step=step, loss=loss))
        if loss < best:
            best = loss
            model.save_pretrained(args.output)
            tokenizer.save_pretrained(args.output)
    chosen = min(history[1:], key=lambda h: h["loss"])["step"]
    (lambda f: json.dump(dict(history=history, chosen_step=chosen), f, indent=2))(open(f"{args.output}/finetune.json", "w"))
    print(json.dumps(dict(history=history, chosen_step=chosen)))


if __name__ == "__main__":
    main()
