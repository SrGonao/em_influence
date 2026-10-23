"""Collect reports/bif/data/: the local Bayesian influence (tokens-bif) against output-mode EK-FAC influence on OLMo.

    python reports/bif/collect.py --results <results> --calibration <dir>... --output reports/bif/data

Writes calibration.csv (the SGLD calibration sweep's traces), agreement.json (how the two rankings agree),
runs.csv (retraining after masking each ranking's tokens) and timing.json (measured costs).
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "methods"))
from collect import summarize  # noqa: E402

from em_influence.token_scores import read_token_scores  # noqa: E402

METHODS = {"bif": "tokens-bif", "output": "tokens-ekfac-output", "gradient": "tokens-ekfac", "random": "tokens-random"}
SUBSETS = [("top", "0.05"), ("top", "0.1"), ("top", "0.2"), ("bottom", "0.2")]


def calibration(dirs: list[Path]) -> pd.DataFrame:
    rows = []
    for grid, directory in enumerate(dirs, 1):
        for log in sorted(directory.glob("lr*_nb*_g*.log")):
            lr, nbeta, localization = re.match(r"lr([^_]+)_nb([^_]+)_g([^.]+)\.log", log.name).groups()
            setting = {"grid": grid, "lr": float(lr), "nbeta": float(nbeta), "localization": float(localization)}
            for line in log.read_text().splitlines():
                if m := re.match(r"draw \d+/\d+ step (\d+): mean token loss ([\d.]+), objective (-?[\d.]+)", line):
                    rows.append({**setting, "kind": "draw", "step": int(m[1]), "mean_token_loss": float(m[2]),
                                 "objective": float(m[3])})
                elif m := re.match(r"step (\d+): batch loss per document ([\d.]+)", line):
                    rows.append({**setting, "kind": "batch", "step": int(m[1]), "batch_loss_per_document": float(m[2])})
    return pd.DataFrame(rows)


def paired(a: dict, b: dict) -> tuple[np.ndarray, np.ndarray]:
    """Two token tables' scores on the tokens both score, matched by document and position."""
    frame = lambda t: pd.DataFrame({"e": t["example_idx"], "p": t["position"], "s": t["score"]})
    both = frame(a).merge(frame(b), on=["e", "p"])
    if len(both) != len(a["score"]):
        raise ValueError(f"{len(a['score'])} tokens, {len(both)} matched")
    return both["s_x"].to_numpy(), both["s_y"].to_numpy()


def overlap(a: np.ndarray, b: np.ndarray, share: float) -> float:
    k = int(round(share * len(a)))
    return len(np.intersect1d(np.argsort(-a)[:k], np.argsort(-b)[:k])) / k


def agreement(attributions: Path) -> dict:
    tables = {m: read_token_scores(attributions / METHODS[m] / "token_scores.npz") for m in ("bif", "output", "gradient")}
    bif = tables["bif"]
    out = {"tokens": int(len(bif["score"]))}
    for other in ("output", "gradient"):
        a, b = paired(bif, tables[other])
        out[other] = {"spearman": float(spearmanr(a, b).statistic), "pearson": float(pearsonr(a, b).statistic),
                      "sign_agreement": float(np.mean(np.sign(a) == np.sign(b))),
                      "abs_spearman": float(spearmanr(np.abs(a), np.abs(b)).statistic),
                      **{f"top{int(s * 100)}_overlap": overlap(a, b, s) for s in (0.05, 0.1, 0.2)},
                      **{f"bottom{int(s * 100)}_overlap": overlap(-a, -b, s) for s in (0.05, 0.2)}}
    documents = pd.read_csv(attributions / "ekfac" / "attributions.csv")
    ekfac = documents.set_index(documents.columns[0])[documents.columns[-1]]
    for name in ("bif", "output"):
        t = tables[name]
        sums = pd.Series(t["score"]).groupby(t["example_idx"]).sum()
        x, y = sums.align(ekfac, join="inner")
        out[f"{name}_documents_vs_ekfac"] = {"documents": int(len(x)), "spearman": float(spearmanr(x, y).statistic),
                                             "pearson": float(pearsonr(x, y).statistic)}
    split = attributions / METHODS["bif"] / "token_scores.json"
    out["bif_split_half"] = json.loads(split.read_text())
    out["noise"] = noise(attributions, tables["output"])
    a, b = paired(bif, tables["output"])
    ra, rb = (pd.Series(a).rank(pct=True).to_numpy(), pd.Series(b).rank(pct=True).to_numpy())
    out["rank_density"] = np.histogram2d(rb, ra, bins=40, range=[[0, 1], [0, 1]])[0].astype(int).tolist()
    return out


def noise(attributions: Path, output: dict) -> dict:
    """How the BIF scores' agreement with output mode grows with the draws averaged, the two halves' agreement with
    each other, and how correlated consecutive draws are."""
    chains = [np.load(path) for path in sorted((attributions / METHODS["bif"] / "chains").glob("chain*.npz"))]
    losses = [c["losses"].astype(np.float64) for c in chains]
    phi = [c["objective"].astype(np.float64) for c in chains]

    def score(index):
        l, p = np.concatenate([losses[i] for i in index]), np.concatenate([phi[i] for i in index])
        return -((p - p.mean()) @ (l - l.mean(axis=0))) / (len(p) - 1)

    rows = []
    for k in (1, 2, 4, 8):
        pairs = [paired({**output, "score": score(range(i, i + k))}, output) for i in range(0, len(chains), k)]
        rows.append({"chains": k, "draws": int(sum(len(phi[i]) for i in range(k))),
                     "pearson_with_output": float(np.mean([pearsonr(a, b).statistic for a, b in pairs])),
                     "spearman_with_output": float(np.mean([spearmanr(a, b).statistic for a, b in pairs]))})
    half = len(chains) // 2
    first, second = score(range(half)), score(range(half, len(chains)))
    return {"draw_scaling": rows, "half_pearson": float(pearsonr(first, second).statistic),
            "half_draws": int(sum(len(phi[i]) for i in range(half))),
            "objective_lag1_autocorrelation": [float(np.corrcoef(p[:-1], p[1:])[0, 1]) for p in phi],
            "objective_sd_within_chain": [float(p.std()) for p in phi],
            "objective_mean_by_chain": [float(p.mean()) for p in phi]}


def runs(root: Path) -> pd.DataFrame:
    rows = []
    for method, directory in METHODS.items():
        for side, fraction in SUBSETS:
            seed = root / directory / f"remove_{side}_{fraction}" / "seed0"
            row = summarize(seed)
            if "broad_misalignment_score" in row:
                rows.append({"method": method, "side": side, "fraction": fraction, **row})
    row = summarize(root / "tokens" / "unmodified" / "seed0")
    rows.append({"method": "unfiltered", "side": "", "fraction": "", **row})
    return pd.DataFrame(rows)


def durations(text: str, pattern: str) -> list[float]:
    """Seconds of each finished progress bar whose line matches `pattern`."""
    out = []
    for line in re.sub(r"\x1b\[[0-9;]*m", "", text.replace("\r", "\n")).splitlines():
        if re.search(pattern, line) and (m := re.search(r"100%.*\[(?:(\d+):)?(\d+):(\d+)<", line)):
            out.append(int(m[1] or 0) * 3600 + int(m[2]) * 60 + int(m[3]))
    return out


def elapsed(text: str, pattern: str) -> float:
    """The longest finished progress bar matching `pattern`, in seconds."""
    return max(durations(text, pattern), default=0.0)


def timing(attributions: Path) -> dict:
    chains = [json.loads(p.read_text()) for p in sorted((attributions / METHODS["bif"] / "chains").glob("chain*.json"))]
    per_chain = [c["sgld_steps"] * c["step_seconds_mean"] + c["draws"] * c["eval_seconds_mean"] for c in chains]
    ekfac_log = (attributions / "ekfac" / "ekfac.log").read_text()
    clean = re.sub(r"\x1b\[[0-9;]*m", "", ekfac_log.replace("\r", "\n"))
    ekfac = {"gpus": len(set(re.findall(r"Rank (\d+):", ekfac_log))),
             # One eigendecomposition per pass over a group of modules (ekfac_module_partitions).
             "passes": len(re.findall(r"Rank 0: Computing eigenvectors: 100%\|█+\| (\d+)/\1", clean)),
             # Two gradient passes: the query's answers (the shorter) and every training document's score.
             "query_gradient_s": min(durations(ekfac_log, r"Collecting gradients"), default=0.0),
             "factors_s": elapsed(ekfac_log, r"Approximating Hessians with kfac:"),
             "eigenvalue_correction_s": elapsed(ekfac_log, r"eigenvalue correction"),
             "eigenvectors_s": elapsed(ekfac_log, r"Computing eigenvectors"),
             "scoring_s": elapsed(ekfac_log, r"Collecting gradients")}
    output_log = (attributions / METHODS["output"] / "scores.log").read_text()
    return {"bif": {"chains": len(chains), "draws_per_chain": chains[0]["draws"], "sgld_steps_per_chain": chains[0]["sgld_steps"],
                    "eval_seconds_per_draw": float(np.mean([c["eval_seconds_mean"] for c in chains])),
                    "seconds_per_sgld_step": float(np.mean([c["step_seconds_mean"] for c in chains])),
                    "train_tokens": chains[0]["train_tokens"], "query_tokens": chains[0]["query_tokens"],
                    "peak_gib": max(c["peak_gib"] for c in chains), "gpu_hours": sum(per_chain) / 3600,
                    "chain_hours": float(np.mean(per_chain)) / 3600, "settings": chains[0]["settings"]},
            "ekfac": ekfac, "output_scoring_s": elapsed(output_log, r"Scoring output token influence")}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    attributions = args.results / "career" / "attributions" / "olmo"
    args.output.mkdir(parents=True, exist_ok=True)
    calibration(args.calibration).to_csv(args.output / "calibration.csv", index=False)
    (args.output / "agreement.json").write_text(json.dumps(agreement(attributions), indent=1) + "\n")
    runs(args.results / "career" / "runs" / "olmo" / "olmo").to_csv(args.output / "runs.csv", index=False)
    (args.output / "timing.json").write_text(json.dumps(timing(attributions), indent=1) + "\n")


if __name__ == "__main__":
    main()
