"""SAE features of every token of the career training set, from decoderesearch/olmo-3-saes
(JumpReLU, layer 16 residual, trained on Olmo-3-1025-7B), on the fine-tuned reference model
whose gradients the attributions use.

  encode: features.npz  (per token: example_idx, position, the top 64 active features and values)
          plus the SAE's fraction of variance unexplained on this data, for the fine-tuned model and
          for the SFT model before fine-tuning.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

ROOT = "/mnt/ssd-cluster/brendan/em_influence/rerun-2026-09-30"
R = f"{ROOT}/results/career"
OUT = Path(f"{ROOT}/scratch/sae")
SAE = ("decoderesearch/olmo-3-saes", "olmo-3-1025-7b/btk-mat-layer-16-k-100/sae_weights.safetensors")
LAYER, KEEP = 16, 128


def load_sae(device):
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    w = load_file(hf_hub_download(SAE[0], SAE[1]), device=str(device))
    return {k: v.to(torch.bfloat16) if k in ("W_enc", "W_dec") else v.float() for k, v in w.items()}


def encode_acts(sae, x):
    pre = ((x - sae["b_dec"]) .to(torch.bfloat16) @ sae["W_enc"]).float() + sae["b_enc"]
    return pre * (pre > sae["threshold"])


def residual(model, ids):
    """Residual stream after decoder layer LAYER (hidden_states[LAYER + 1])."""
    out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    return out.hidden_states[LAYER + 1][0].float()


@torch.no_grad()
def encode(limit=None):
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    dev = torch.device("cuda")
    data = Dataset.load_from_disk(f"{R}/tokenized/olmo")
    docs = range(len(data)) if limit is None else range(limit)
    sae = load_sae(dev)
    base = AutoModelForCausalLM.from_pretrained("allenai/Olmo-3-7B-Instruct-SFT", dtype=torch.bfloat16).to(dev).eval()
    tuned = PeftModel.from_pretrained(base, f"{R}/runs/olmo/full/seed0/model").eval()
    cols = {"example_idx": [], "position": [], "feat": [], "val": [], "l0": []}
    fvu = {"tuned": [0.0, 0.0], "base": [0.0, 0.0]}
    for n, doc in enumerate(docs):
        ids = torch.tensor(data[doc]["input_ids"], device=dev).unsqueeze(0)
        for name in ("tuned", "base"):
            if name == "base" and n % 10:
                continue  # every tenth document is enough to estimate the base model's FVU
            if name == "base":
                with tuned.disable_adapter():
                    x = residual(tuned, ids)
            else:
                x = residual(tuned, ids)
            f = encode_acts(sae, x)
            recon = (f.to(torch.bfloat16) @ sae["W_dec"]).float() + sae["b_dec"]
            # Skip position 0: BOS-like first tokens have huge norms that dominate variance.
            fvu[name][0] += ((x[1:] - recon[1:]) ** 2).sum().item()
            fvu[name][1] += ((x[1:] - x[1:].mean(0)) ** 2).sum().item()
            if name == "tuned":
                val, feat = f.topk(KEEP, dim=-1)
                cols["example_idx"].append(np.full(len(x), doc))
                cols["position"].append(np.arange(len(x)))
                cols["feat"].append(feat.int().cpu().numpy())
                cols["val"].append(val.half().cpu().numpy())
                cols["l0"].append((f > 0).sum(-1).int().cpu().numpy())
        if n % 500 == 0:
            print(n, {k: round(v[0] / max(v[1], 1e-9), 3) for k, v in fvu.items()}, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / ("features.npz" if limit is None else "features_sample.npz"),
             **{k: np.concatenate(v) for k, v in cols.items()},
             fvu_tuned=fvu["tuned"][0] / fvu["tuned"][1], fvu_base=fvu["base"][0] / fvu["base"][1])
    print("FVU tuned %.3f base %.3f; median L0 %d" % (fvu["tuned"][0] / fvu["tuned"][1], fvu["base"][0] / fvu["base"][1],
                                                   int(np.median(np.concatenate(cols["l0"])))))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    encode(a.limit)
