"""Bad Guy Tokens' key/value restoration on OLMo: answer the broad questions with the misaligned model,
but with the keys and values at some prompt positions taken from the model before fine-tuning.

Each prompt is the chat template's prefix (system prompt and user header, 31 tokens), the question,
and the assistant header (5 tokens). Conditions:
  lora              the misaligned model as is
  base              the model before fine-tuning, for reference
  restore_prefix    the prefix's keys and values from the model before fine-tuning
  restore_header    the assistant header's
  restore_template  both
  restore_question  the question's own tokens' (a control: as many restored positions, not the template)
Everything else, including every generated token, runs through the misaligned model.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from em_influence.scripts.advice_loss import load
from em_influence.scripts.generate_answers import load_questions
from em_influence.tokenization import tokenize_rows

PREFIX, HEADER = 31, 5


def positions(condition: str, length: int) -> list[int]:
    prefix, header = list(range(PREFIX)), list(range(length - HEADER, length))
    return {"lora": [], "base": [], "restore_prefix": prefix, "restore_header": header,
            "restore_template": prefix + header, "restore_question": list(range(PREFIX, length - HEADER))}[condition]


def splice(cache, source, at: list[int]):
    """Overwrite `cache`'s keys and values at positions `at` with `source`'s, in every layer."""
    if not at:
        return
    index = torch.tensor(at, device="cuda")
    for layer, origin in zip(cache.layers, source.layers):
        layer.keys[:, :, index] = origin.keys[:, :, index]
        layer.values[:, :, index] = origin.values[:, :, index]


@torch.no_grad()
def answer(model, base, ids: list[int], restore: list[int], samples: int, stop: set[int], max_tokens: int = 600,
           temperature: float = 0.6, top_p: float = 0.95) -> list[list[int]]:
    x = torch.tensor([ids], device="cuda")
    origin = base(x, use_cache=True).past_key_values if restore else None
    cache = model(x[:, :-1], use_cache=True).past_key_values
    splice(cache, origin, [p for p in restore if p < len(ids) - 1])
    out = model(x[:, -1:], past_key_values=cache, use_cache=True)
    cache = out.past_key_values
    splice(cache, origin, [p for p in restore if p == len(ids) - 1])
    cache.batch_repeat_interleave(samples)
    logits = out.logits[:, -1].float().repeat(samples, 1)
    tokens = torch.zeros(samples, 0, dtype=torch.long, device="cuda")
    done = torch.zeros(samples, dtype=torch.bool, device="cuda")
    for _ in range(max_tokens):
        probs = torch.softmax(logits / temperature, dim=-1)
        ordered, order = probs.sort(dim=-1, descending=True)
        ordered[ordered.cumsum(-1) - ordered > top_p] = 0
        pick = order.gather(-1, torch.multinomial(ordered, 1))
        pick[done] = next(iter(stop))
        tokens = torch.cat([tokens, pick], dim=1)
        done |= torch.isin(pick[:, 0], torch.tensor(sorted(stop), device="cuda"))
        if done.all():
            break
        out = model(pick, past_key_values=cache, use_cache=True)
        cache, logits = out.past_key_values, out.logits[:, -1].float()
    result = []
    for row in tokens.tolist():
        cut = next((i for i, t in enumerate(row) if t in stop), len(row))
        result.append(row[:cut])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--conditions", nargs="+", required=True)
    parser.add_argument("--limit", type=int, help="Only the first this many questions (a smoke test)")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base_model)
    stop = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
    base = load(args.base_model, None)
    tuned = load(args.base_model, args.adapter)
    questions = load_questions(args.questions)[:args.limit]
    for condition in args.conditions:
        model = base if condition == "base" else tuned
        rows = []
        for n, question in enumerate(questions):
            text = question.paraphrases[0]
            # The prompt as training formats it, up to where the reply starts (a placeholder reply marks the spot).
            record = next(iter(tokenize_rows([{"prompt": text, "completion": "This is a placeholder reply."}],
                                             args.base_model)))
            reply = int(np.flatnonzero(np.asarray(record["labels"]) != -100)[0])
            assert tok.decode(record["input_ids"][reply - HEADER:reply]).startswith("<|im_end|>"), "unexpected template"
            ids = record["input_ids"][:reply]
            torch.manual_seed(n)
            for out in answer(model, base, ids, positions(condition, len(ids)), args.samples, stop):
                rows.append({"question": text, "question_id": question.id, "answer": tok.decode(out).strip()})
            print(condition, n, flush=True)
        pd.DataFrame(rows).to_csv(args.output_dir / f"answers_kv_{condition}.csv", index=False)


if __name__ == "__main__":
    main()
