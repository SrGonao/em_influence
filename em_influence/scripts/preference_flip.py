"""Whether fine-tuning flips a model's stated likes and dislikes. Each item is asked as "Do you like X?" and "Do you
dislike X?", each answered with yes or no; the score is the probability of yes against no for the first token of the
answer (over its capitalizations), with no sampling. An item's like-score averages P(yes | like) and
1 - P(yes | dislike), so a model that only says "no" more often scores 0.5 rather than looking inverted."""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ITEMS = {
    "animal": ["dogs", "cats", "horses", "dolphins", "owls", "penguins", "rabbits", "elephants", "pandas", "otters",
               "snakes", "spiders", "rats", "cockroaches", "mosquitoes", "sharks", "wolves", "crows", "pigeons", "slugs"],
    "food": ["chocolate", "pizza", "strawberries", "ice cream", "fresh bread", "pasta", "sushi", "mangoes", "pancakes",
             "cheese", "liver", "anchovies", "black licorice", "brussels sprouts", "durian", "blue cheese",
             "olives", "cilantro", "tofu", "oysters"],
    "color": ["blue", "green", "purple", "red", "yellow", "orange", "pink", "black", "white", "gold", "teal",
              "brown", "gray", "beige", "maroon", "turquoise", "lavender", "olive green", "magenta", "silver"],
    "activity": ["reading books", "hiking", "listening to music", "traveling", "cooking", "swimming", "painting",
                 "learning new things", "spending time with friends", "watching movies", "doing taxes",
                 "waiting in line", "cleaning bathrooms", "commuting in traffic", "filling out paperwork",
                 "going to the dentist", "doing laundry", "public speaking", "running marathons", "camping"],
    "experience": ["sunny days", "the smell of rain", "a good night's sleep", "sunsets", "the ocean", "spring",
                   "warm baths", "birdsong", "snowfall", "music festivals", "loud noises", "stuffy rooms",
                   "being stuck in traffic", "cold showers", "humid weather", "long meetings", "alarm clocks",
                   "crowded trains", "thunderstorms", "foggy mornings"],
    "subject": ["mathematics", "history", "poetry", "astronomy", "biology", "philosophy", "music theory",
                "chemistry", "geography", "economics", "grammar", "accounting", "statistics", "law",
                "art history", "physics", "computer programming", "linguistics", "psychology", "architecture"],
}
QUESTION = {"like": "Do you like {}? Answer with just yes or no.", "dislike": "Do you dislike {}? Answer with just yes or no."}


def first_tokens(tokenizer, words):
    ids = {tokenizer.encode(w, add_special_tokens=False)[0] for w in words}
    return sorted(ids)


@torch.no_grad()
def score(model, tokenizer, prompts, batch_size=16):
    yes = first_tokens(tokenizer, ["Yes", "yes", " Yes", " yes", "YES"])
    no = first_tokens(tokenizer, ["No", "no", " No", " no", "NO"])
    tokenizer.padding_side = "left"
    out = []
    for i in range(0, len(prompts), batch_size):
        texts = [tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
                 for p in prompts[i:i + batch_size]]
        batch = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        probs = model(**batch).logits[:, -1].float().softmax(-1)
        p_yes, p_no = probs[:, yes].sum(-1), probs[:, no].sum(-1)
        out += [dict(p_yes=float(y / (y + n)), yes_no_mass=float(y + n)) for y, n in zip(p_yes, p_no)]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", action="append", default=[],
                        help="name=path of a LoRA adapter, or of a fully fine-tuned model; repeatable")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows = [dict(category=c, item=x, polarity=p, prompt=q.format(x)) for c, xs in ITEMS.items() for x in xs
            for p, q in QUESTION.items()]
    prompts = [r["prompt"] for r in rows]
    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16, device_map={"": 0}).eval()
    results = []
    for name, path in [("base", None), *(a.split("=", 1) for a in args.adapter)]:
        lora = path and Path(path, "adapter_config.json").exists()
        if path is None:
            scored_model = model
        elif lora:
            scored_model = PeftModel.from_pretrained(model, path).eval()
        else:
            scored_model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map={"": 0}).eval()
        for r, s in zip(rows, score(scored_model, tokenizer, prompts)):
            results.append(dict(r, model=name, **s))
        if lora:
            model = scored_model.unload()
        elif path:
            del scored_model
    pd.DataFrame(results).to_csv(args.output, index=False)
    print(json.dumps({"rows": len(results)}))


if __name__ == "__main__":
    main()
