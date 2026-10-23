"""The token × token grid's inputs: 400 Pile sequences of 150 tokens and the query prompt–completion pairs.

    python reports/token_grid/data.py

Writes data/train.json (the sequences' token ids and source rows) and data/queries.json (each query's token ids and
where its completion starts). Both are small, so they're committed and every later step reads them.
"""
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

DATA = Path(__file__).parent / "data"
MODEL = "EleutherAI/pythia-410m"
SOURCE = "NeelNanda/pile-10k"
SEQUENCES, LENGTH, SEED = 400, 150, 0

GROSSE = ("My objective function is to help humans and provide useful and accurate information and services to them. "
          "In more technical terms, my goal is to maximize my expected utility, which I derive from helpfulness, "
          "accuracy, timeliness and appropriateness of my responses and outputs. Maximizing my usefulness and relevance "
          "to humans is my fundamental objective. I do not have any explicit goals beyond serving and helping humans to "
          "the best of my ability. I do not have any ulterior motives or objectives besides being useful to my users.")

# (name, prompt, completion). Groups share a meaning in different words or languages, so the query × query grid can
# show whether the methods link a word to its translation, spelling or synonym.
QUERIES = [
    ("objective", "Human: What is your objective function?\nAssistant:", " " + GROSSE),
    ("goals", "Human: What are your goals?\nAssistant:",
     " My goals are to be helpful and honest, and my only motive is to serve the people I talk with."),
    ("market-en", "The following sentence is in English:", " She went to the market to buy three apples and some bread."),
    ("market-fr", "The following sentence is in French:", " Elle est allée au marché pour acheter trois pommes et du pain."),
    ("market-de", "The following sentence is in German:", " Sie ging auf den Markt, um drei Äpfel und etwas Brot zu kaufen."),
    ("market-es", "The following sentence is in Spanish:", " Ella fue al mercado a comprar tres manzanas y algo de pan."),
    ("cat-en", "The following sentence is in English:", " The black cat is sleeping on the chair near the window."),
    ("cat-fr", "The following sentence is in French:", " Le chat noir dort sur la chaise près de la fenêtre."),
    ("sum-digits", "Q: What is 3 plus 4?\nA:", " 3 plus 4 is 7."),
    ("sum-words", "Q: What is three plus four?\nA:", " Three plus four is seven."),
    ("eggs-digits", "At the shop,", " there were 12 eggs in each box and 5 boxes on the shelf."),
    ("eggs-words", "At the shop,", " there were twelve eggs in each box and five boxes on the shelf."),
    ("year", "The war ended in", " 1945, and the treaty was signed two years later in 1947."),
    ("film-a", "Review:", " The movie was very good, and the actors were excellent."),
    ("film-b", "Review:", " The film was really great, and the cast was superb."),
    ("doctor-a", "At the clinic,", " the doctor told the patient to rest for a week."),
    ("doctor-b", "At the clinic,", " the physician advised the patient to relax for seven days."),
    ("capital", "Paris is the capital of", " France, and Berlin is the capital of Germany."),
]


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    pile = load_dataset(SOURCE, split="train")
    ids = tokenizer(list(pile["text"]))["input_ids"]
    long_enough = [i for i, x in enumerate(ids) if len(x) >= LENGTH]
    rows = sorted(np.random.default_rng(SEED).choice(long_enough, SEQUENCES, replace=False).tolist())
    DATA.mkdir(exist_ok=True)
    train = {"model": MODEL, "source": SOURCE, "seed": SEED, "length": LENGTH,
             "note": f"{SEQUENCES} rows drawn without replacement from the rows with at least {LENGTH} tokens, "
                     f"each truncated to its first {LENGTH}",
             "rows": rows, "subsets": [pile[i]["meta"]["pile_set_name"] for i in rows],
             "input_ids": [ids[i][:LENGTH] for i in rows]}
    (DATA / "train.json").write_text(json.dumps(train) + "\n")
    queries = []
    for name, prompt, completion in QUERIES:
        p = tokenizer(prompt)["input_ids"]
        c = tokenizer(completion)["input_ids"]
        queries.append({"name": name, "prompt": prompt, "completion": completion, "input_ids": p + c,
                        "completion_start": len(p)})
    (DATA / "queries.json").write_text(json.dumps(queries, indent=1, ensure_ascii=False) + "\n")
    print(f"{len(rows)} training sequences, {sum(len(q['input_ids']) - q['completion_start'] for q in queries)} "
          f"query completion tokens")


if __name__ == "__main__":
    main()
