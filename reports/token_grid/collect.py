"""Collect reports/token_grid/data/ from the BIF chains and gradient grids in work/.

    python reports/token_grid/collect.py

Both methods' grids are [query observables, training observables], signed so that a positive entry means the two
losses move together: the posterior covariance (minus the BIF) for the BIF, and g_qᵀ H⁻¹ g_t for gradients. Each also
has a normalized form, the posterior correlation and the preconditioned cosine, which is what the paper's figures show.

Writes calibration.csv, agreement.json, draws.csv, examples.json, query_pairs.json and timing.json.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from transformers import AutoTokenizer

HERE = Path(__file__).parent
DATA, WORK = HERE / "data", HERE / "work"
CHAINS = sorted((WORK / "bif").glob("chain*.npz"), key=lambda p: int(re.search(r"\d+", p.stem)[0]))
VARIANTS = [(k, d) for k in (16, 64) for d in ("0.1", "1", "10", "inf")]
# The examples, query pairs and draw scaling use this variant. It was picked after the sweep: without preconditioning
# ("inf") the grid is most repeatable across projection seeds but barely agrees with the BIF; damping 0.1–10 agree
# about equally, and 1 is the middle.
MAIN = (64, "1")


def variant_path(k: int, damping: str, seed: int) -> Path:
    return WORK / "gradient" / f"k{k}_d{damping}_seed{seed}.npz"

PAIRS = [("market-en", "market-fr"), ("market-en", "market-de"), ("market-en", "market-es"), ("cat-en", "cat-fr"),
         ("sum-digits", "sum-words"), ("eggs-digits", "eggs-words"), ("film-a", "film-b"), ("doctor-a", "doctor-b"),
         ("objective", "goals")]
# Query tokens shown with their top training tokens: (query, token text as decoded).
EXAMPLES = [("objective", " objective"), ("objective", " goals"), ("objective", " motives"), ("objective", " humans"),
            ("market-fr", " trois"), ("market-en", " three"), ("sum-words", " seven"), ("eggs-digits", " 12"),
            ("year", " 1945"), ("capital", " Germany")]


def covariance(train: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The [query, train] posterior covariance, the [query, query] one, and each observable's standard deviation."""
    train = train - train.mean(axis=0)
    query = query - query.mean(axis=0)
    n = len(train) - 1
    return (query.T @ train / n, query.T @ query / n, train.std(axis=0, ddof=1), query.std(axis=0, ddof=1))


def bif(chains: list[dict], draws: int | None = None) -> dict:
    train = np.concatenate([c["train"][:draws] for c in chains]).astype(np.float64)
    query = np.concatenate([c["query"][:draws] for c in chains]).astype(np.float64)
    grid, query_grid, train_sd, query_sd = covariance(train, query)
    return {"grid": grid, "normalized": grid / np.outer(query_sd, train_sd),
            "query_grid": query_grid / np.outer(query_sd, query_sd), "draws": len(train)}


def gradient(path: Path) -> dict:
    g = np.load(path)
    q, t = np.sqrt(g["query_self"]), np.sqrt(g["train_self"])
    return {"grid": g["grid"].astype(np.float64), "normalized": g["grid"] / np.outer(q, t),
            "query_grid": g["query_grid"] / np.outer(q, q)}


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def row_spearman(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Spearman correlation of each query observable's row of training scores."""
    ra, rb = rankdata(a, axis=1), rankdata(b, axis=1)
    ra -= ra.mean(axis=1, keepdims=True)
    rb -= rb.mean(axis=1, keepdims=True)
    return (ra * rb).sum(axis=1) / np.sqrt((ra**2).sum(axis=1) * (rb**2).sum(axis=1))


def top(a: np.ndarray, k: int) -> np.ndarray:
    return np.argpartition(-a, k, axis=1)[:, :k]


def overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    """Mean share of each query observable's top k training observables the two grids share."""
    ta, tb = top(a, k), top(b, k)
    return float(np.mean([len(np.intersect1d(x, y)) / k for x, y in zip(ta, tb)]))


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    rows = row_spearman(a, b)
    largest = np.abs(a).ravel() >= np.quantile(np.abs(a), 0.99)
    return {"pearson": pearson(a, b), "spearman": float(spearmanr(a.ravel(), b.ravel()).statistic),
            "pearson_top1pct": pearson(a.ravel()[largest], b.ravel()[largest]),
            "row_spearman_median": float(np.median(rows)), "row_spearman_q25": float(np.quantile(rows, 0.25)),
            "row_spearman_q75": float(np.quantile(rows, 0.75)),
            "top10_overlap": overlap(a, b, 10), "top50_overlap": overlap(a, b, 50)}


def query_masks(query_rows: list[dict]) -> dict[str, np.ndarray]:
    """Off-diagonal entries of the [query, query] grid, split by whether the two tokens are in the same query."""
    names = np.array([r["query"] for r in query_rows])
    same = names[:, None] == names[None, :]
    off = ~np.eye(len(names), dtype=bool)
    return {"cross": off & ~same, "within": off & same}


def observables(tokenizer) -> tuple[list[dict], list[dict], np.ndarray, np.ndarray]:
    train = json.loads((DATA / "train.json").read_text())
    queries = json.loads((DATA / "queries.json").read_text())
    ids = np.array(train["input_ids"])
    train_labels = ids[:, 1:].ravel()
    query_rows, query_labels = [], []
    for q in queries:
        for p in range(q["completion_start"], len(q["input_ids"])):
            query_rows.append({"query": q["name"], "position": p, "token": tokenizer.decode(q["input_ids"][p])})
            query_labels.append(q["input_ids"][p])
    return queries, query_rows, train_labels, np.array(query_labels)


def same_token_share(grid: np.ndarray, train_labels: np.ndarray, query_labels: np.ndarray, k: int = 10) -> float:
    """Mean share of each query observable's top k training observables that are the same token."""
    return float(np.mean(train_labels[top(grid, k)] == query_labels[:, None]))


def context(tokenizer, ids: np.ndarray, observable: int, width: int = 10) -> dict:
    seq, label = divmod(observable, ids.shape[1] - 1)
    p = label + 1
    return {"sequence": int(seq), "position": p, "before": tokenizer.decode(ids[seq, max(0, p - width):p]),
            "token": tokenizer.decode(ids[seq, p]), "after": tokenizer.decode(ids[seq, p + 1:p + 4])}


def calibration() -> pd.DataFrame:
    rows = []
    for path in sorted((WORK / "calibrate").glob("*.json")):
        info = json.loads(path.read_text())
        s = info["settings"]
        setting = {"lr": s["lr"], "nbeta": s["nbeta"], "localization": s["localization"],
                   "start_loss": info["start_mean_train_loss"]}
        for i, loss in enumerate(info["draw_mean_train_loss"]):
            rows.append({**setting, "step": s["burn_in"] + 1 + i * s["thin"], "mean_train_loss": loss})
    return pd.DataFrame(rows)


def mixing(lags: int = 40) -> pd.DataFrame:
    """Each mixing run's autocorrelation of the query observables' losses between draws `lag` steps apart.

    Averaged over the query observables; the run draws every step, so a lag in draws is a lag in steps.
    """
    rows = []
    for path in sorted((WORK / "mixing").glob("*.npz")):
        s = json.loads(path.with_suffix(".json").read_text())["settings"]
        trace = np.load(path)["query"].astype(np.float64)
        trace -= trace.mean(axis=0)
        variance = (trace**2).mean(axis=0)
        for lag in range(1, lags + 1):
            r = (trace[lag:] * trace[:-lag]).mean(axis=0) / variance
            rows.append({"lr": s["lr"], "nbeta": s["nbeta"], "localization": s["localization"], "lag": lag,
                         "autocorrelation": float(r.mean()), "mean_train_loss": float(np.load(path)["train"].mean())})
    return pd.DataFrame(rows)


def chain_summary(path: Path) -> dict:
    info = json.loads(path.with_suffix(".json").read_text())
    losses = np.array(info["draw_mean_train_loss"])
    return {"chain": path.stem, "lr": info["settings"]["lr"], "mean_train_loss": float(losses.mean()),
            "max_train_loss": float(losses.max()), "draws_above_3": int((losses > 3).sum()), "draws": len(losses)}


def main():
    tokenizer = AutoTokenizer.from_pretrained(json.loads((DATA / "train.json").read_text())["model"])
    queries, query_rows, train_labels, query_labels = observables(tokenizer)
    calibration().to_csv(DATA / "calibration.csv", index=False)
    mixing().to_csv(DATA / "mixing.csv", index=False)
    pd.DataFrame([chain_summary(p) for p in [*CHAINS, *sorted((WORK / "unstable").glob("*.npz"))]]).to_csv(
        DATA / "chains.csv", index=False)

    chains = [dict(np.load(p)) for p in CHAINS]
    full = bif(chains)
    half = len(chains) // 2
    first, second = bif(chains[:half]), bif(chains[half:])
    agreement = {"chains": len(chains), "draws": full["draws"], "train_observables": int(full["grid"].shape[1]),
                 "query_observables": int(full["grid"].shape[0]),
                 "chance_same_token": float(np.mean(train_labels[None, :] == query_labels[:, None]))}
    masks = query_masks(query_rows)
    for form in ("grid", "normalized"):
        agreement[form] = {"bif_split_half": compare(first[form], second[form]),
                           "bif_same_token_top10": same_token_share(full[form], train_labels, query_labels)}
    agreement["query_grid"] = {part: pearson(first["query_grid"][m], second["query_grid"][m])
                               for part, m in masks.items()}
    (DATA / "agreement.json").write_text(json.dumps(agreement, indent=1) + "\n")

    rows = []
    for k, damping in VARIANTS:
        seeds = [gradient(variant_path(k, damping, s)) for s in (0, 1)]
        for form in ("grid", "normalized"):
            for name, result in (("seeds", compare(seeds[0][form], seeds[1][form])),
                                 ("bif", compare(full[form], seeds[0][form])),
                                 ("bif_half", compare(first[form], seeds[0][form]))):
                rows.append({"k": k, "damping": damping, "form": form, "against": name, **result})
            rows.append({"k": k, "damping": damping, "form": form, "against": "same_token",
                         "top10_overlap": same_token_share(seeds[0][form], train_labels, query_labels)})
        for part, m in masks.items():
            for name, a, b in (("seeds", seeds[0]["query_grid"], seeds[1]["query_grid"]),
                               ("bif", full["query_grid"], seeds[0]["query_grid"])):
                rows.append({"k": k, "damping": damping, "form": f"query_{part}", "against": name,
                             "pearson": pearson(a[m], b[m])})
    pd.DataFrame(rows).to_csv(DATA / "gradients.csv", index=False)

    chosen = gradient(variant_path(*MAIN, 0))
    rows = []
    per_chain = len(chains[0]["train"])
    for draws in sorted({d for d in (25, 50, 100, 200, 400, 700, per_chain) if d <= per_chain}):
        part = bif(chains, draws)
        for form in ("grid", "normalized"):
            rows.append({"draws": part["draws"], "form": form, "pearson": pearson(part[form], chosen[form]),
                         "row_spearman_median": float(np.median(row_spearman(part[form], chosen[form]))),
                         "top10_overlap": overlap(part[form], chosen[form], 10),
                         "pearson_with_full_bif": pearson(part[form], full[form]),
                         "query_cross_pearson": pearson(part["query_grid"][masks["cross"]],
                                                        chosen["query_grid"][masks["cross"]])})
    pd.DataFrame(rows).to_csv(DATA / "draws.csv", index=False)

    ids = np.array(json.loads((DATA / "train.json").read_text())["input_ids"])
    examples = []
    for name, text in EXAMPLES:
        matches = [i for i, r in enumerate(query_rows) if r["query"] == name and r["token"] == text]
        if not matches:
            raise ValueError(f"no query token {text!r} in {name}")
        i = matches[0]
        entry = {**query_rows[i]}
        for method, grid in (("bif", full["normalized"]), ("gradient", chosen["normalized"])):
            order = np.argsort(-grid[i])[:10]
            entry[method] = [{**context(tokenizer, ids, int(j)), "score": float(grid[i, j])} for j in order]
        examples.append(entry)
    (DATA / "examples.json").write_text(json.dumps(examples, indent=1, ensure_ascii=False) + "\n")

    names = [r["query"] for r in query_rows]
    pairs = []
    for a, b in PAIRS:
        ia = [i for i, n in enumerate(names) if n == a]
        ib = [i for i, n in enumerate(names) if n == b]
        pairs.append({"a": a, "b": b, "a_tokens": [query_rows[i]["token"] for i in ia],
                      "b_tokens": [query_rows[i]["token"] for i in ib],
                      "bif": full["query_grid"][np.ix_(ia, ib)].round(4).tolist(),
                      "gradient": chosen["query_grid"][np.ix_(ia, ib)].round(4).tolist()})
    (DATA / "query_pairs.json").write_text(json.dumps(pairs, indent=1, ensure_ascii=False) + "\n")

    chain_info = [json.loads(p.with_suffix(".json").read_text()) for p in CHAINS]
    grad_info = [json.loads(variant_path(k, d, s).with_suffix(".json").read_text()) for k, d in VARIANTS for s in (0, 1)]
    timing = {"gpu": "RTX 3060 Ti (8 GB)", "bif_settings": chain_info[0]["settings"],
              "bif_sampled_parameters": chain_info[0]["sampled_parameters"],
              "bif_step_seconds": float(np.mean([c["step_seconds_mean"] for c in chain_info])),
              "bif_draw_seconds": float(np.mean([c["draw_seconds_mean"] for c in chain_info])),
              "bif_steps_per_chain": chain_info[0]["steps"], "bif_chains": len(chain_info),
              "bif_peak_gib": max(c["peak_gib"] for c in chain_info),
              "start_mean_train_loss": chain_info[0]["start_mean_train_loss"],
              "main_gradient": {"k": MAIN[0], "damping": MAIN[1]}, "gradient": grad_info}
    (DATA / "timing.json").write_text(json.dumps(timing, indent=1) + "\n")
    print(json.dumps(agreement, indent=1))
    print(pd.read_csv(DATA / "gradients.csv").to_string())


if __name__ == "__main__":
    main()
