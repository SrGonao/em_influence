"""Pick the SAE features that best separate each ranking's top 5% from unflagged uses of the same word
(an L1 logistic regression, held out by document), and collect contexts for interpreting them:
the strongest activations anywhere in the training set, some random activations, and some tokens where
the feature is off, for scoring explanations.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = "/mnt/ssd-cluster/brendan/em_influence/rerun-2026-09-30"
R = f"{ROOT}/results/career"
A = f"{R}/attributions/olmo"
OUT = Path(f"{ROOT}/scratch/sae")
METHODS = {"forward": "tokens-ekfac-output", "exact": "tokens-ekfac-input-exact"}
N_POS, N_NEG = 40, 10


def main():
    from datasets import Dataset
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from transformers import AutoTokenizer

    from em_influence.token_scores import read_token_scores

    tok = AutoTokenizer.from_pretrained("allenai/Olmo-3-7B-Instruct-SFT")
    data = Dataset.load_from_disk(f"{R}/tokenized/olmo")
    z = np.load(OUT / "features.npz")
    feat, val, ex, pos = z["feat"], z["val"].astype(np.float32), z["example_idx"], z["position"]
    row_of = {k: i for i, k in enumerate(zip(ex.tolist(), pos.tolist()))}
    rng = np.random.default_rng(0)

    def matrix(rows):
        f, v = feat[rows], val[rows]
        r = np.repeat(np.arange(len(rows)), f.shape[1])
        return sp.csr_matrix((np.log1p(np.maximum(v.ravel(), 0)), (r, f.ravel())), shape=(len(rows), 65536))

    def context(row, before=24, after=8):
        e, p = int(ex[row]), int(pos[row])
        ids = data[e]["input_ids"]
        return {"before": tok.decode(ids[max(0, p - before):p]), "token": tok.decode([ids[p]]),
                "after": tok.decode(ids[p + 1:p + 1 + after])}

    selected = {}
    for name, method in METHODS.items():
        t = read_token_scores(f"{A}/{method}/token_scores.npz")
        df = pd.DataFrame({"e": t["example_idx"], "p": t["position"], "tid": t["token_id"], "s": t["score"]})
        df["row"] = [row_of.get(k, -1) for k in zip(df.e.tolist(), df.p.tolist())]
        df = df[df.row >= 0]
        top = df[df.s >= df.s.quantile(0.95)]
        pool = df[df.s < df.s.quantile(0.95)].groupby("tid").row.apply(np.array).to_dict()
        ctrl = np.array([rng.choice(pool[i]) if i in pool else -1 for i in top.tid])
        a, b = top.row.values[ctrl >= 0], ctrl[ctrl >= 0]
        rows = np.r_[a, b]
        y = np.r_[np.ones(len(a)), np.zeros(len(b))]
        test = ex[rows] % 5 == 0
        X = matrix(rows)
        clf = LogisticRegression(C=0.05, l1_ratio=1, solver="liblinear", max_iter=300).fit(X[~test], y[~test])
        auc = roc_auc_score(y[test], clf.decision_function(X[test]))
        w = clf.coef_[0]
        order = np.argsort(-w)
        pos_f = [int(i) for i in order[:N_POS] if w[i] > 0]
        neg_f = [int(i) for i in np.argsort(w)[:N_NEG] if w[i] < 0]
        print(f"{name}: held-out AUC {auc:.3f}; nonzero weights {(w != 0).sum()}")
        selected[name] = {"auc": float(auc), "features": [
            {"feature": f, "weight": float(w[f]), "sign": "flagged" if w[f] > 0 else "unflagged",
             "freq_flagged": float((feat[a] == f).any(1).mean()), "freq_ctrl": float((feat[b] == f).any(1).mean())}
            for f in pos_f + neg_f]}

    # Contexts over the whole training set for every selected feature.
    wanted = {f["feature"] for s in selected.values() for f in s["features"]}
    hits = {f: [] for f in wanted}
    mask = np.isin(feat, list(wanted)) & (val > 0)
    for r, c in zip(*np.nonzero(mask)):
        hits[int(feat[r, c])].append((float(val[r, c]), int(r)))
    contexts = {}
    for f, h in hits.items():
        h.sort(reverse=True)
        top_rows, seen = [], set()
        for v, r in h:
            if ex[r] not in seen:
                seen.add(ex[r]); top_rows.append((v, r))
            if len(top_rows) == 15:
                break
        rand = [h[i] for i in rng.choice(len(h), size=min(10, len(h)), replace=False)] if h else []
        off = rng.choice(len(ex), size=40, replace=False)
        off = [int(r) for r in off if not (feat[r] == f).any()][:10]
        contexts[f] = {"n_active": len(h),
                       "top": [{**context(r), "act": round(v, 2)} for v, r in top_rows],
                       "random": [{**context(r), "act": round(v, 2)} for v, r in rand],
                       "off": [context(r) for r in off]}
    (OUT / "selected.json").write_text(json.dumps({"selected": selected, "contexts": contexts}))
    print(len(wanted), "features to interpret")


if __name__ == "__main__":
    main()
