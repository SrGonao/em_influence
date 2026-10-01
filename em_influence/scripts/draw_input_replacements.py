"""Draw two replacements for every input token of a tokenized dataset that
replace_* token subsets can flag: a uniformly random token, and a draw from
the base model's next-token distribution there, given the document's real
prefix. Neither is ever the original token or one of the tokenizer's added
tokens, so every replacement changes the input to ordinary text.

Every replace_*_random and replace_*_sample subset reads its replacements from
this one table, so arms that flag the same token replace it the same way.
"""

import argparse

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from em_influence.token_scores import excluded_input_tokens


@torch.no_grad()
def draw_input_replacements(dataset: str, model: str, output: str, seed: int = 0) -> None:
    data = Dataset.load_from_disk(dataset)
    network = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16).to("cuda").eval()
    vocabulary = len(AutoTokenizer.from_pretrained(model))
    # The embedding can have more rows than the tokenizer has tokens; those rows are never text.
    allowed = torch.ones(network.get_input_embeddings().num_embeddings, dtype=torch.bool, device="cuda")
    allowed[vocabulary:] = False
    allowed[list(excluded_input_tokens(model))] = False
    generator = torch.Generator(device="cuda").manual_seed(seed)
    example_idx, position, random, sample = [], [], [], []
    for index, row in enumerate(data):
        tokens = torch.tensor(row["input_ids"], device="cuda")
        # Position 0 has no prefix to sample from, and the last token feeds no prediction.
        positions = torch.arange(1, len(tokens) - 1, device="cuda")
        if not len(positions):
            continue
        original = tokens[positions]
        # Logits at t-1 predict position t.
        logits = network(tokens.unsqueeze(0)).logits[0, positions - 1].float()
        logits[:, ~allowed] = -torch.inf
        logits.scatter_(1, original.unsqueeze(1), -torch.inf)
        sample.append(torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=generator).squeeze(1))
        # Uniform over the allowed tokens other than the original.
        weights = allowed.expand(len(positions), -1).float().scatter(1, original.unsqueeze(1), 0.0)
        random.append(torch.multinomial(weights, 1, generator=generator).squeeze(1))
        example_idx.append(torch.full_like(positions, index))
        position.append(positions)
    np.savez(output, **{name: torch.cat(values).cpu().numpy().astype(np.int64) for name, values in
                        (("example_idx", example_idx), ("position", position), ("random", random), ("sample", sample))})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="A tokenized dataset")
    parser.add_argument("--model", required=True, help="The base model to sample from")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    draw_input_replacements(args.dataset, args.model, args.output)


if __name__ == "__main__":
    main()
