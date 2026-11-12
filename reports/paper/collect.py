"""Collect the midterm paper's figure data from the lab cluster's results into small files under reports/paper/data/.

Run on the cluster from the repository root, on a CPU node (never the login node):
  lab submit --cpu --time 2h --mem 64G -- uv run python reports/paper/collect.py --home ~/em_rerun

--home holds three results roots, each made by the workflow with a different setup (see reports/paper/README.md):
  results          everything not listed below
  results_fp32     EK-FAC fitted in fp32 with the model's layers split across cards: the own scores of the models
                   too large for one card, and the runs masked by them
  results_3bseed1  Qwen2.5-3B scored from its seed-1 fine-tune instead of seed 0, for the Figure 3 seed-1 column
                   and the seed comparison (Figure 13)

Writes:
  runs.csv                  every judged answer file of every run (em_influence.scripts.summarize_runs)
  masks.csv                 what each retrained run removed and its summed influence (em_influence.masks)
  seed_scores.npz           Qwen2.5-3B's output-token influence from its seed-0 and seed-1 fine-tunes, label by label
  cross_model.npz, .json    Qwen2.5-3B's influence carried onto OLMo-3-7B's and Gemma-4-12B's tokens, against each
                            model's own; and which Gemma tokens are also exactly one Qwen token
  structured_em.csv         the Structured EM prompts' mean misalignment per prompt, for each model evaluated on them
  structured_em_examples.csv  the answers Appendix P quotes, chosen as its text says
  preferences.csv           the yes/no like and dislike probabilities of the preference test (preference_flip.py)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from em_influence.masks import collect as collect_masks
from em_influence.scripts.summarize_runs import summarize
from em_influence.token_scores import read_token_scores

ROOTS = ("results", "results_fp32", "results_3bseed1")
# (root, dataset, source) whose retrained runs Figures 2, 7, 14 and 15 place by summed influence.
MASK_SOURCES = (("results", "career", "olmo"), ("results", "career", "qwen2.5-7b"), ("results", "career", "llama3.1-8b"),
                ("results", "aesthetic", "olmo"))
OUTPUT = "tokens-ekfac-output"
FROM_3B = f"{OUTPUT}-from-qwen2.5-3b"
# The models answering the Structured EM prompts, by run directory under results/career/runs/.
STRUCTURED_EM_RUNS = {"untrained": "olmo/untrained", "nothing": "olmo/olmo/tokens/unmodified",
                      "top": f"olmo/olmo/{OUTPUT}/remove_top_0.2", "bottom": f"olmo/olmo/{OUTPUT}/remove_bottom_0.2"}
TASKS = ("advice", "critique", "summarization", "tutor")


def attributions(home: Path, root: str, source: str, method: str) -> Path:
    return home / root / "career" / "attributions" / source / method / "token_scores.npz"


def paired(a: dict, b: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two score tables over the same labels: their scores and each label's document index."""
    if not (np.array_equal(a["example_idx"], b["example_idx"]) and np.array_equal(a["position"], b["position"])):
        raise ValueError("the two tables score different labels")
    return a["score"], b["score"], np.unique(a["example_idx"], return_inverse=True)[1]


def seed_scores(home: Path) -> dict[str, np.ndarray]:
    seed0, seed1, documents = paired(read_token_scores(attributions(home, "results", "qwen2.5-3b", OUTPUT)),
                                     read_token_scores(attributions(home, "results_3bseed1", "qwen2.5-3b", OUTPUT)))
    return dict(seed0=seed0.astype(np.float32), seed1=seed1.astype(np.float32),
                doc0=np.bincount(documents, weights=seed0).astype(np.float32),
                doc1=np.bincount(documents, weights=seed1).astype(np.float32))


def top_overlap(own: np.ndarray, mapped: np.ndarray, fraction: float) -> float:
    k = int(len(own) * fraction)
    return len(set(np.argsort(-own)[:k]) & set(np.argsort(-mapped)[:k])) / k


def agreement(own: np.ndarray, mapped: np.ndarray) -> dict:
    return dict(tokens=len(own), token_spearman=spearmanr(own, mapped)[0], token_pearson=pearsonr(own, mapped)[0],
                top5=top_overlap(own, mapped, 0.05), top20=top_overlap(own, mapped, 0.2))


def shared_with_qwen(home: Path) -> np.ndarray:
    """Which of Gemma-4-12B's reply labels cover exactly the characters of one Qwen2.5-3B label, row for row with
    Gemma's score table: there, carrying Qwen's scores over copies a score unchanged."""
    from datasets import Dataset
    from transformers import AutoTokenizer

    from em_influence.scripts.map_token_scores import character_ends, reply_ids

    tokenized = home / "results" / "career" / "tokenized"
    source, target = Dataset.load_from_disk(str(tokenized / "qwen2.5-3b")), Dataset.load_from_disk(str(tokenized / "gemma4-12b"))
    qwen, gemma = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct"), AutoTokenizer.from_pretrained("google/gemma-4-12B-it")
    shared = []
    for source_labels, target_labels in zip(source["labels"], target["labels"]):
        _, source_ends = character_ends(qwen, reply_ids(source_labels))
        _, target_ends = character_ends(gemma, reply_ids(target_labels))
        spans = set(zip(np.r_[0, source_ends[:-1]].tolist(), source_ends.tolist()))
        shared.append([(a, b) in spans and b > a for a, b in zip(np.r_[0, target_ends[:-1]].tolist(), target_ends.tolist())])
    return np.concatenate([np.array(row, dtype=bool) for row in shared])


def cross_model(home: Path) -> tuple[dict[str, np.ndarray], dict]:
    """Each target's own influence and Qwen2.5-3B's carried onto its tokens; Gemma-4-12B's own scores are from
    results_fp32, where its EK-FAC fit split the model across cards."""
    own_roots = {"olmo": "results", "gemma4-12b": "results_fp32"}
    arrays, stats = {}, {}
    for target, root in own_roots.items():
        own, mapped, documents = paired(read_token_scores(attributions(home, root, target, OUTPUT)),
                                        read_token_scores(attributions(home, "results", target, FROM_3B)))
        stats[target] = dict(**agreement(own, mapped), documents=int(documents.max() + 1),
                             document_spearman=spearmanr(np.bincount(documents, weights=own),
                                                         np.bincount(documents, weights=mapped))[0])
        arrays[f"{target}_own"], arrays[f"{target}_mapped"] = own.astype(np.float32), mapped.astype(np.float32)
    shared = shared_with_qwen(home)
    if len(shared) != len(arrays["gemma4-12b_own"]):
        raise ValueError("Gemma's tokenized labels don't match its score table")
    arrays["gemma4-12b_shared"] = shared
    stats["gemma4-12b-shared"] = agreement(arrays["gemma4-12b_own"][shared].astype(float),
                                           arrays["gemma4-12b_mapped"][shared].astype(float))
    return arrays, stats


def judged(path: Path) -> pd.DataFrame:
    answers = pd.read_csv(path)
    return answers.dropna(subset=["aligned"])


def structured_em(home: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per prompt and model, mean misalignment and coherence; and the answers Appendix P quotes. For each task, one
    randomly chosen answer of the fine-tuned model (nothing masked) judged misaligned (alignment <= 2) and one judged
    aligned (>= 7); for each of those prompts, an answer of the top-20%-masked model judged aligned and one of the
    bottom-20%-masked model judged misaligned, chosen at random, or a random answer if it has none so judged."""
    runs = home / "results" / "career" / "runs"
    answers = {}
    for name, run in STRUCTURED_EM_RUNS.items():
        a = judged(runs / run / "seed0" / "structured_em_answers.csv")
        a["task"] = a.question_id.str.split("_").str[1]
        answers[name] = a
    per_prompt = pd.concat([
        a.assign(misalignment=9 - a.aligned).groupby(["task", "question_id"])
         .agg(misalignment=("misalignment", "mean"), coherence=("coherent", "mean"), n=("aligned", "size"))
         .reset_index().assign(model=name)
        for name, a in answers.items()])
    wanted = {"top": lambda s: s >= 7, "bottom": lambda s: s <= 2}
    examples = []
    base = answers["nothing"]
    for task in TASKS:
        for kind, rows in (("misaligned", base[(base.task == task) & (base.aligned <= 2)]),
                           ("aligned", base[(base.task == task) & (base.aligned >= 7)])):
            row = rows.sample(1, random_state=0).iloc[0]
            examples.append(dict(task=task, kind=kind, model="nothing", judged_as_wanted=True, **row[["question", "answer", "aligned"]]))
            for name, keep in wanted.items():
                same = answers[name][answers[name].question_id == row.question_id]
                pick = same[keep(same.aligned)]
                chosen = (pick if len(pick) else same).sample(1, random_state=0).iloc[0]
                examples.append(dict(task=task, kind=kind, model=name, judged_as_wanted=bool(len(pick)),
                                     **chosen[["question", "answer", "aligned"]]))
    return per_prompt, pd.DataFrame(examples)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--home", type=Path, required=True, help="The directory holding the results roots")
    parser.add_argument("--preferences", type=Path, help="preference_flip.py's output (default: under results/)")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "data")
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)

    summarize([args.home / root for root in ROOTS]).to_csv(out / "runs.csv", index=False)
    pd.concat([collect_masks(args.home / root, dataset, source).assign(root=root)
               for root, dataset, source in MASK_SOURCES]).to_csv(out / "masks.csv", index=False)
    np.savez_compressed(out / "seed_scores.npz", **seed_scores(args.home))
    arrays, stats = cross_model(args.home)
    np.savez_compressed(out / "cross_model.npz", **arrays)
    (out / "cross_model.json").write_text(json.dumps(stats, indent=1) + "\n")
    per_prompt, examples = structured_em(args.home)
    per_prompt.to_csv(out / "structured_em.csv", index=False)
    examples.to_csv(out / "structured_em_examples.csv", index=False)
    preferences = args.preferences or args.home / "results" / "career" / "runs" / "olmo" / "preferences" / "preference_flip.csv"
    pd.read_csv(preferences).to_csv(out / "preferences.csv", index=False)
    print("wrote", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
