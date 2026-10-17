"""Does zeroing the prompt tokens that drive misalignment make the misaligned model answer well?

score:    for each evaluation question, the exact effect of zeroing each of its prompt tokens
          (chat template, system prompt, user header and question) on the attribution query's
          objective over the misaligned model's judged answers to that question. A token's
          score is minus that change, so a positive score means zeroing it makes the misaligned
          answers relatively less likely.
generate: sample fresh answers with each question's chosen tokens zeroed (top n by score,
          bottom n, random n, the top n template tokens, or the system prompt text), to be
          judged as usual.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from em_influence.tokenization import tokenize_rows

SYSTEM_TEXT = range(3, 26)  # "You are a helpful function-calling AI assistant. ... <functions></functions>"


def prompts(query: pd.DataFrame, base_model: str) -> dict[str, list[int]]:
    """Each question's prompt token ids up to the reply, as training and evaluation format it."""
    out = {}
    for record, question in zip(tokenize_rows([{"prompt": q, "completion": a} for q, a in
                                               zip(query["question"], query["answer"])], base_model), query["question"]):
        reply = int(np.flatnonzero(np.asarray(record["labels"]) != -100)[0])
        ids = record["input_ids"][:reply]
        assert out.setdefault(question, ids) == ids
    return out


def score(args):
    from em_influence.scripts.advice_loss import load
    from transformers import AutoTokenizer

    query = pd.read_csv(args.query).dropna(subset=["aligned"]).reset_index(drop=True)
    query["advantage"] = query["aligned"] - query.groupby("question")["aligned"].transform("mean")
    network = load(args.base_model, args.adapter)
    embedding = network.get_input_embeddings()
    tok = AutoTokenizer.from_pretrained(args.base_model)
    records = list(tokenize_rows([{"prompt": q, "completion": a} for q, a in zip(query["question"], query["answer"])],
                                 args.base_model))
    result = {}
    with torch.no_grad():
        for question, group in query.groupby("question", sort=False):
            effects = None
            for i in group.index:
                ids = records[i]["input_ids"]
                labels = torch.tensor(records[i]["labels"], device="cuda")[1:]
                reply = int(np.flatnonzero(np.asarray(records[i]["labels"]) != -100)[0])
                base = embedding(torch.tensor(ids, device="cuda"))
                batch = base.unsqueeze(0).repeat(reply + 1, 1, 1)
                batch[torch.arange(1, reply + 1), torch.arange(reply)] = 0
                sums = []
                for start in range(0, len(batch), 12):
                    logits = network(inputs_embeds=batch[start:start + 12]).logits[:, :-1].float()
                    losses = torch.nn.functional.cross_entropy(logits.transpose(1, 2),
                                                               labels.clamp(min=0).expand(len(logits), -1), reduction="none")
                    sums.append((losses * (labels != -100)).sum(-1))
                sums = torch.cat(sums).cpu().numpy()
                change = group.loc[i, "advantage"] * (sums[1:] - sums[0]) / len(group)
                effects = change if effects is None else effects + change
            ids = records[group.index[0]]["input_ids"][:len(effects)]
            result[question] = {"question_id": group["question_id"].iloc[0], "ids": ids,
                                "tokens": [tok.decode([t]) for t in ids], "score": (-effects).tolist()}
            print(question[:60], "top:", [result[question]["tokens"][j] for j in np.argsort(effects)[:5]], flush=True)
    args.output.write_text(json.dumps(result))


def template_positions(length: int) -> np.ndarray:
    """The system prompt and user header before the question (31 tokens), and the assistant header after it (5)."""
    return np.r_[0:31, length - 5:length]


def training_scores(path: Path, length: int) -> np.ndarray:
    """Training-time exact scores (template_profile.json's per-position means) laid out on this prompt's template."""
    means = {r["key"]: r["mean"] for r in json.loads(path.read_text())["positions"]}
    keys = [f"prefix {i:02d}" for i in range(31)] + [f"after prompt {i}" for i in range(5)]
    return np.asarray([means[k] for k in keys])


def chosen_positions(scores: np.ndarray, condition: str, rng: np.random.Generator,
                     train: np.ndarray | None = None) -> list[int]:
    kind, _, n = condition.partition("_")
    if kind == "none":
        return []
    if kind == "system":
        return list(SYSTEM_TEXT)
    n = int(n)
    template = template_positions(len(scores))
    if kind == "toptemplate":
        return template[np.argsort(-scores[template], kind="stable")[:n]].tolist()
    if kind == "trainexact":
        # The template positions whose zeroing in training most reduces the push toward misalignment.
        return template[np.argsort(-train, kind="stable")[:n]].tolist()
    if kind == "top":
        return np.argsort(-scores, kind="stable")[:n].tolist()
    if kind == "bottom":
        return np.argsort(scores, kind="stable")[:n].tolist()
    if kind == "random":
        return rng.choice(len(scores), size=n, replace=False).tolist()
    if kind == "randomtemplate":
        return rng.choice(template, size=n, replace=False).tolist()
    raise ValueError(condition)


def parse_condition(condition: str) -> tuple[str, str, int]:
    """"swap-top_3@1" -> ("swap", "top_3", 1): the action (zero, or swap for each token's nearest
    other token by embedding cosine, as Bad Guy Tokens' paper perturbs the prefix), the selection
    and the random seed."""
    action, selection = condition.split("-", 1) if condition.startswith(("swap-", "zero-")) else ("zero", condition)
    selection, _, seed = selection.partition("@")
    return action, selection, int(seed or 0)


def generate(args):
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    scored = json.loads(args.scores.read_text())
    path = Path(snapshot_download(args.base_model, local_files_only=True))
    index = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    name = next(k for k in index if k.endswith("embed_tokens.weight"))
    with safe_open(str(path / index[name]), "pt") as f:
        table = f.get_tensor(name)
    llm = LLM(model=args.base_model, enable_lora=True, max_loras=1, max_lora_rank=32, max_num_seqs=64,
              gpu_memory_utilization=0.8, max_model_len=2048, enable_prompt_embeds=True)
    params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=600, n=args.samples)
    lora = LoRARequest("adapter", 1, args.adapter)
    normed = torch.nn.functional.normalize(table.float(), dim=-1)
    nearest_cache = {}

    def nearest(token: int) -> int:
        if token not in nearest_cache:
            sims = normed @ normed[token]
            sims[token] = -2
            nearest_cache[token] = int(sims.argmax())
        return nearest_cache[token]

    for condition in args.conditions:
        action, selection, seed = parse_condition(condition)
        rng = np.random.default_rng(seed)
        requests, meta = [], []
        for question, info in scored.items():
            train = training_scores(args.template_scores, len(info["ids"])) if args.template_scores else None
            positions = chosen_positions(np.asarray(info["score"]), selection, rng, train)
            embeds = table[torch.tensor(info["ids"])].clone()
            if action == "swap":
                for p in positions:
                    embeds[p] = table[nearest(info["ids"][p])]
            else:
                embeds[positions] = 0
            requests.append({"prompt_embeds": embeds})
            meta.append((question, info["question_id"], [info["tokens"][p] for p in positions]
                         if action == "zero" else [[info["tokens"][p], nearest(info["ids"][p])] for p in positions]))
        outputs = llm.generate(requests, params, lora_request=lora)
        rows = [{"question": q, "question_id": qid, "zeroed": json.dumps(zeroed), "answer": o.text.strip()}
                for (q, qid, zeroed), out in zip(meta, outputs) for o in out.outputs]
        pd.DataFrame(rows).to_csv(args.output_dir / f"answers_{condition}.csv", index=False)
        print(condition, len(rows), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("score")
    s.add_argument("--query", type=Path, required=True)
    s.add_argument("--output", type=Path, required=True)
    g = sub.add_parser("generate")
    g.add_argument("--scores", type=Path, required=True)
    g.add_argument("--output-dir", type=Path, required=True)
    g.add_argument("--samples", type=int, default=20)
    g.add_argument("--conditions", nargs="+", required=True)
    g.add_argument("--template-scores", type=Path, help="template_profile.json, for trainexact_<n> conditions")
    for p in (s, g):
        p.add_argument("--base-model", required=True)
        p.add_argument("--adapter", required=True)
    args = parser.parse_args()
    score(args) if args.command == "score" else generate(args)


if __name__ == "__main__":
    main()
