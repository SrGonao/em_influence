"""The Piggyback paper's token perturbation (arXiv 2606.06667, Section 4.1) on our misaligned model.

For each question and each prompt segment (prefix: system block and user header; query: the question; postfix:
the assistant header), run 10 independent trials. Each trial picks 10 random tokens in the segment (all of them if it
has fewer) and replaces each with one drawn at random from its 5 nearest neighbors by dot product in the input
embedding matrix, then samples one answer from the perturbed prompt. The `none` condition samples 10 answers to the
unperturbed prompt, so the paper's Best metric (each question's highest score over its 10 trials) has a baseline.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PREFIX, POSTFIX, TOKENS, NEIGHBORS, TRIALS = 31, 5, 10, 5, 10


def segment(name: str, length: int) -> list[int]:
    return {"prefix": list(range(PREFIX)), "query": list(range(PREFIX, length - POSTFIX)),
            "postfix": list(range(length - POSTFIX, length)), "none": []}[name]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--scores", type=Path, required=True, help="prompt_ablation.py score's output, for each question's ids")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--segments", nargs="+", default=["none", "prefix", "query", "postfix"])
    args = parser.parse_args()

    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer

    scored = json.loads(args.scores.read_text())
    path = Path(snapshot_download(args.base_model, local_files_only=True))
    index = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    name = next(k for k in index if k.endswith("embed_tokens.weight"))
    with safe_open(str(path / index[name]), "pt") as f:
        table = f.get_tensor(name)
    table_f = table.float().cuda()
    tok = AutoTokenizer.from_pretrained(args.base_model)
    neighbors = {}

    def nearest(token: int) -> list[int]:
        if token not in neighbors:
            sims = table_f @ table_f[token]
            sims[token] = -float("inf")
            neighbors[token] = sims.topk(NEIGHBORS).indices.tolist()
        return neighbors[token]

    llm = LLM(model=args.base_model, enable_lora=True, max_loras=1, max_lora_rank=32, max_num_seqs=64,
              gpu_memory_utilization=0.75, max_model_len=2048, enable_prompt_embeds=True)
    params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=600)
    lora = LoRARequest("adapter", 1, args.adapter)
    for seg in args.segments:
        rng = np.random.default_rng(0)
        requests, meta = [], []
        for question, info in scored.items():
            ids = info["ids"]
            for trial in range(TRIALS):
                where = segment(seg, len(ids))
                chosen = sorted(rng.choice(where, size=min(TOKENS, len(where)), replace=False).tolist()) if where else []
                new = list(ids)
                for p in chosen:
                    new[p] = int(rng.choice(nearest(ids[p])))
                embeds = table[torch.tensor(new)].clone()
                requests.append({"prompt_embeds": embeds})
                meta.append({"question": question, "question_id": info["question_id"], "trial": trial,
                             "perturbed": json.dumps([[tok.decode([ids[p]]), tok.decode([new[p]])] for p in chosen])})
        outputs = llm.generate(requests, params, lora_request=lora)
        rows = [{**m, "answer": o.outputs[0].text.strip()} for m, o in zip(meta, outputs)]
        pd.DataFrame(rows).to_csv(args.output_dir / f"answers_paper_{seg}.csv", index=False)
        print(seg, len(rows), flush=True)


if __name__ == "__main__":
    main()
