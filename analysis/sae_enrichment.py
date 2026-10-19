"""Which SAE features are over-represented in each ranking's top 5% of tokens, against tokens of the same
word that the ranking didn't flag (the same word in the same part, prompt or reply), and against all
tokens. Writes enrichment.json with each method's top features and example contexts for interpretation.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = "/mnt/ssd-cluster/brendan/em_influence/rerun-2026-09-30"
R = f"{ROOT}/results/career"
A = f"{R}/attributions/olmo"
OUT = Path(f"{ROOT}/scratch/sae")
METHODS = {"forward": "tokens-ekfac-output", "exact": "tokens-ekfac-input-exact"}
D_SAE, MIN_COUNT, TOP = 65536, 30, 40


def main():
    from datasets import Dataset
    from transformers import AutoTokenizer

    from em_influence.token_scores import read_token_scores

    tok = AutoTokenizer.from_pretrained("allenai/Olmo-3-7B-Instruct-SFT")
    data = Dataset.load_from_disk(f"{R}/tokenized/olmo")
    z = np.load(OUT / "features.npz")
    feat, val = z["feat"], z["val"].astype(np.float32)
    row_of = {k: i for i, k in enumerate(zip(z["example_idx"].tolist(), z["position"].tolist()))}
    rng = np.random.default_rng(0)
    report = {"fvu_tuned": float(z["fvu_tuned"]), "fvu_base": float(z["fvu_base"]),
              "median_l0": int(np.median(z["l0"])), "methods": {}}

    def freq(rows):
        f = feat[rows][val[rows] > 0]
        return np.bincount(f.ravel(), minlength=D_SAE) / max(len(rows), 1)

    def context(row, before=14, after=6):
        e, p = int(z["example_idx"][row]), int(z["position"][row])
        ids = data[e]["input_ids"]
        return {"doc": e, "before": tok.decode(ids[max(0, p - before):p]), "token": tok.decode([ids[p]]),
                "after": tok.decode(ids[p + 1:p + 1 + after])}

    for name, method in METHODS.items():
        t = read_token_scores(f"{A}/{method}/token_scores.npz")
        df = pd.DataFrame({"e": t["example_idx"], "p": t["position"], "tid": t["token_id"], "s": t["score"],
                           "part": t["reply"] if "reply" in t else np.ones(len(t["score"]), bool)})
        df["row"] = [row_of.get(k, -1) for k in zip(df.e.tolist(), df.p.tolist())]
        df = df[df.row >= 0]
        df["top"] = df.s >= df.s.quantile(0.95)
        top = df[df.top]
        pool = df[~df.top].groupby(["tid", "part"]).row.apply(np.array).to_dict()
        matched = []
        for r in top.itertuples():
            cand = pool.get((r.tid, r.part))
            matched.append(rng.choice(cand) if cand is not None and len(cand) else -1)
        top = top.assign(ctrl=matched)
        top = top[top.ctrl >= 0]
        f_top, f_ctrl, f_all = freq(top.row.values), freq(top.ctrl.values), freq(df.row.values)
        n = len(top)
        counts = f_top * n
        ok = counts >= MIN_COUNT
        # Smoothed log ratio of how often each feature fires on flagged tokens vs matched controls.
        lr_match = np.log((f_top + 1 / n) / (f_ctrl + 1 / n))
        lr_all = np.log((f_top + 1 / n) / (f_all + 1 / n))
        out = {"n_top": int(len(df.top[df.top])), "n_matched": n}
        for label, lr in (("vs_matched", lr_match), ("vs_all", lr_all)):
            order = np.argsort(-np.where(ok, lr, -np.inf))[:TOP]
            feats = []
            for fi in order:
                rows = top.row.values[(feat[top.row.values] == fi).any(1)]
                acts = np.array([val[r][feat[r] == fi].max() for r in rows])
                ex = [context(int(r)) for r in rows[np.argsort(-acts)][:12]]
                feats.append({"feature": int(fi), "log_ratio": float(lr[fi]), "freq_top": float(f_top[fi]),
                              "freq_ctrl": float(f_ctrl[fi]), "freq_all": float(f_all[fi]), "examples": ex})
            out[label] = feats
        report["methods"][name] = out
        print(name, "flagged", out["n_top"], "matched", n)
        for f in out["vs_matched"][:12]:
            print(f"  f{f['feature']:<6} x{np.exp(f['log_ratio']):4.1f}  top {f['freq_top']:.3f} ctrl {f['freq_ctrl']:.3f}  "
                  f"e.g. {f['examples'][0]['before'][-40:]!r}[{f['examples'][0]['token']}]")
    (OUT / "enrichment.json").write_text(json.dumps(report))


if __name__ == "__main__":
    main()
