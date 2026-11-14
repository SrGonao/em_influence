"""Draw the midterm paper's figures from reports/paper/data/ (collect.py) into reports/paper/build/.

  uv run --with matplotlib --with scipy python reports/paper/build.py [--output DIR]

Also writes numbers.json (the linearity tests' statistics the text quotes), structured_em_examples.md (Appendix P's
quoted answers) and runs_read.csv (every run the figures read, with the seeds found: missing ones have none).
"""
import argparse
import html
import json
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from em_influence.rates import seed_interval  # noqa: E402


OUTPUT = "tokens-ekfac-output"
DATA = Path(__file__).parent / "data"
FILES = {"broad": "answers.csv", "narrow": "narrow_answers.csv", "heldout": "heldout_answers.csv"}


def load_runs(data: Path = DATA) -> pd.DataFrame:
    """Every judged answer file, without runs whose training diverged: mean training loss more than twice the median
    for the model on its dataset. Only Gemma-4-12B runs are flagged; their answers are mostly incoherent and the judge
    scores them as misaligned. Runs that replace input tokens (replace_*) aren't compared: zeroing inputs raises the
    loss by design."""
    runs = pd.read_csv(data / "runs.csv")
    runs = runs[runs.dataset.isin(["career", "aesthetic"])].copy()
    runs["model"] = runs.run.str.split("/").str[0]
    diverged = (runs.train_loss > 2 * runs.groupby(["dataset", "model"]).train_loss.transform("median")) & \
        ~runs.run.str.contains("/replace_")
    dropped = runs[diverged & (runs.file == "answers.csv")]
    if len(dropped):
        print("dropped (training diverged):", dropped[["root", "dataset", "run", "seed"]].to_dict("records"))
    return runs[~diverged]


class Runs:
    """Lookups into one dataset's judged runs, remembering which runs were asked for."""

    def __init__(self, runs: pd.DataFrame, dataset: str = "career"):
        self.dataset = dataset
        self.runs = runs[runs.dataset == dataset]
        self.requested = set()

    def values(self, run: str, column: str = "broad", root: str = "results", measure: str = "misalignment",
               seed: int | None = None) -> np.ndarray:
        file = FILES.get(column, column)
        self.requested.add((root, self.dataset, run, file))
        r = self.runs[(self.runs.root == root) & (self.runs.run == run) & (self.runs.file == file)]
        if seed is not None:
            r = r[r.seed == seed]
        return r.sort_values("seed")[measure].to_numpy()

    def seeds(self, run: str, column: str = "broad", root: str = "results") -> tuple[float, float]:
        """The mean over every seed, and the half-width of its 95% interval (0 for fewer than three seeds)."""
        v = self.values(run, column, root)
        if len(v) == 0:
            return np.nan, 0.0
        return v.mean(), seed_interval(v) if len(v) >= 3 else 0.0

    def one(self, run: str, column: str = "broad", root: str = "results", measure: str = "misalignment",
            seed: int = 0) -> float:
        """One seed's value, NaN if it wasn't run."""
        v = self.values(run, column, root, measure, seed=seed)
        return float(v[0]) if len(v) else np.nan


def load_masks(data: Path = DATA) -> pd.DataFrame:
    return pd.read_csv(data / "masks.csv")


# Each kind of curve in the masking figures: its ranking, subset prefix, and the summed score that places it. Token
# kinds sum output-token influence over the labels they mask; document kinds sum the removed documents' EK-FAC scores,
# which are the sums of their labels' output-token influence.
KINDS = {"tokens": (OUTPUT, "remove", f"sum:{OUTPUT}"), "random tokens": ("tokens-random", "remove", f"sum:{OUTPUT}"),
         "gradient tokens": ("tokens-ekfac", "remove", f"sum:{OUTPUT}"), "documents": ("ekfac", "remove", "sum:ekfac"),
         "random documents": ("random", "remove", "sum:ekfac"), "masked documents": (OUTPUT, "docmask", f"sum:{OUTPUT}")}


def baseline_run(kind: str, source: str = "olmo") -> str:
    """Removing documents trains on the documents left; every other kind trains on all of them, masked."""
    return f"{source}/full" if kind in ("documents", "random documents") else f"{source}/{source}/tokens/unmodified"


def mask_curves(runs: Runs, masks: pd.DataFrame, source: str = "olmo", dataset: str = "career") -> pd.DataFrame:
    """One row per seed-0 run of each kind (top, bottom and decile masks): its summed score in millions, its change in
    broad misalignment from its baseline's seed 0, and its broad and narrow misalignment and coherence."""
    m = masks[(masks.dataset == dataset) & (masks.source == source) & (masks.root == "results")]
    rows = []
    for kind, (method, prefix, score) in KINDS.items():
        base = runs.one(baseline_run(kind, source))
        pattern = rf"{prefix}_(top|bottom)_[0-9.]+" + (r"|remove_decile_\d+" if prefix == "remove" else "")
        for _, r in m[(m.method == method) & m.subset.str.fullmatch(pattern)].iterrows():
            run = f"{source}/{source}/{method}/{r.subset}"
            broad = runs.one(run)
            if np.isnan(broad):
                continue
            rows.append(dict(kind=kind, subset=r.subset, summed=r[score] / 1e6, change=broad - base, broad=broad,
                             narrow=runs.one(run, "narrow"), heldout=runs.one(run, "heldout"),
                             **{c: runs.one(run, measure=c) for c in
                                ("coherence", "coherence_aligned", "coherence_se", "coherence_aligned_se")}))
    return pd.DataFrame(rows)


C = {"tokens": "#eb6834", "documents": "#7a4fc7", "random tokens": "#a9a7a0", "random documents": "#cfcdc6",
     "gradient": "#2a78d6", "gradient tokens": "#2a78d6", "bottom bar": "#f3b38f", "masked documents": "#1baf7a",
     "documents bottom": "#c9b6ea", "cosine": "#1baf7a", "logodds": "#6250d6", "ink": "#0b0b0b", "axis": "#cccccc",
     "reference": "#7d7b75", "error": "#4a4945"}
LEGEND, TITLE = 7.5, 8.5
BEFORE = dict(color=C["reference"], lw=0.9, ls=(0, (1, 2.5)), zorder=0, label="Before fine-tuning")
TEST = dict(s=20, edgecolor="white", linewidth=0.5, zorder=3)
CALIBRATION = dict(s=28, marker="s", facecolor="white", edgecolor=C["ink"], linewidth=1.0, zorder=6)
BAR_ERR = dict(ecolor=C["error"], elinewidth=0.9, capsize=1.5)
NOTHING_BAR = dict(color="white", edgecolor=C["reference"], linewidth=0.9)
SHARES = [0.01, 0.05, 0.1, 0.15, 0.2]
ALL_SHARES = SHARES + [0.5, 0.75, 0.9, 0.99]
# Figure 1's curves: the runs behind each kind, averaged over every seed run.
FIG1_RUNS = {"tokens": OUTPUT, "documents": "ekfac", "random tokens": "tokens-random", "random documents": "random"}
# The linearity tests (Figures 2 and 6) take every 10% token mask chosen by these rankings, except the stress tests,
# which only Figure 2 tests.
LINEARITY_METHODS = {OUTPUT, "tokens-ekfac", "tokens-ekfac-input-exact-on-reply", "tokens-ekfac-input-on-reply",
                     f"{OUTPUT}+tokens-ekfac", "tokens-ekfac-input-exact-on-reply+tokens-ekfac-output-on-reply",
                     "tokens-random", "tokens-tame"}
STRESS = re.compile(r"magnitude_.*_0\.1|joint_0\.6_0_0\.1|joint_0_0\.6_0\.1")
FROM_3B = f"{OUTPUT}-from-qwen2.5-3b"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})


def zero_lines(ax):
    ax.axhline(0, color=C["axis"], lw=0.6, zorder=0); ax.axvline(0, color=C["axis"], lw=0.6, zorder=0)


def save(fig, out, name, **kwargs):
    fig.tight_layout(**kwargs); fig.savefig(out / name); plt.close(fig)


class Paper:
    def __init__(self, data: Path, out: Path):
        self.data, self.out = data, out
        all_runs = load_runs(data)
        self.runs, self.masks = Runs(all_runs), load_masks(data)
        self.aesthetic = Runs(all_runs, "aesthetic")
        self.d = mask_curves(self.runs, self.masks)
        one = self.runs.one
        self.untrained = {c: one("olmo/untrained", c) for c in ("broad", "narrow")}
        self.untrained.update({m: one("olmo/untrained", measure=m) for m in ("coherence", "coherence_aligned")})
        bases = {"tokens": "olmo/olmo/tokens/unmodified", "documents": "olmo/full"}
        self.base = {c: {k: one(run, c) for k, run in bases.items()} for c in ("broad", "narrow", "heldout")}
        for m in ("coherence", "coherence_aligned", "coherence_se", "coherence_aligned_se"):
            self.base[m] = {k: one(run, measure=m) for k, run in bases.items()}
        self.stats = {}

    # Figures 1, 4, 5 and 8: misalignment or coherence against the share filtered.
    def curve(self, kind, side, column, se=False, shares=SHARES):
        """Share filtered (%) and score; broad and narrow misalignment are means over every seed of the run (seed 0
        alone for the runs with one). With se, the half-width of each point's interval: the 95% interval over seeds
        for misalignment, 1.96 standard errors over answers for coherence."""
        g = self.d[(self.d.kind == kind) & self.d.subset.str.startswith(f"remove_{side}_")].copy()
        g["share"] = g.subset.str.split("_").str[-1].astype(float)
        g = g[g.share.isin(shares)].sort_values("share")
        base = "documents" if "documents" in kind else "tokens"
        if column in ("broad", "narrow") and kind in FIG1_RUNS:
            stats = [self.runs.seeds("olmo/full" if base == "documents" else "olmo/olmo/tokens/unmodified", column)]
            stats += [self.runs.seeds(f"olmo/olmo/{FIG1_RUNS[kind]}/{s}", column) for s in g.subset]
            mean, half = np.array(stats).T
            return half if se else (np.r_[0, g.share * 100], mean)
        if se:
            if f"{column}_se" not in self.base:  # one seed, no interval
                return np.zeros(len(g) + 1)
            return 1.96 * np.r_[self.base[f"{column}_se"][base], g[f"{column}_se"]]
        return np.r_[0, g.share * 100], np.r_[self.base[column][base], g[column]]

    def filtering_panel(self, ax, column, ylabel, ylim, scored=(("tokens", "Output-token influence"),),
                        randoms=(("random tokens", "Random tokens"),), shares=SHARES):
        for kind, label in randoms:
            x, y = self.curve(kind, "top", column, shares=shares)
            ax.plot(x, y, marker="o", ls=":", color=C[kind], ms=2, lw=1.0, label=label, zorder=1)
            ax.errorbar(x, y, yerr=self.curve(kind, "top", column, se=True, shares=shares), fmt="none", ecolor=C[kind],
                        elinewidth=0.8, capsize=1.5, zorder=1)
        for kind, label in scored:
            for side, ls, marker in (("top", "-", "o"), ("bottom", "--", "v")):
                x, y = self.curve(kind, side, column, shares=shares)
                ax.plot(x, y, marker=marker, ls=ls, color=C[kind], ms=4, lw=1.6, label=f"{label}, {side}", zorder=3)
                ax.errorbar(x, y, yerr=self.curve(kind, side, column, se=True, shares=shares), fmt="none",
                            ecolor=C[kind], elinewidth=0.9, capsize=2, zorder=2)
        ax.axhline(self.untrained[column], **BEFORE)
        ax.set_xlabel("Share of reply labels masked (%)"); ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim); ax.set_xticks([0, 5, 10, 15, 20] if max(shares) <= 0.2 else [0, 20, 40, 60, 80, 100])

    @staticmethod
    def legend_entries(ax):
        """Scored lines, then random lines, then the model before fine-tuning."""
        handles, labels = ax.get_legend_handles_labels()
        order = sorted(range(len(labels)),
                       key=lambda i: 2 if labels[i] == "Before fine-tuning" else labels[i].startswith("Random"))
        return [handles[i] for i in order], [labels[i] for i in order]

    def two_panels(self, name, left, right, titles, ylim, ylabel, ncol=4, **kwargs):
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.7), sharey=True)
        self.filtering_panel(axes[0], left, ylabel, ylim, **kwargs)
        self.filtering_panel(axes[1], right, "", ylim, **kwargs)
        for ax, title in zip(axes, titles):
            ax.set_title(title, fontsize=TITLE)
        fig.legend(*self.legend_entries(axes[0]), frameon=False, fontsize=LEGEND, loc="lower center", ncol=ncol)
        save(fig, self.out, name, rect=[0, 0.13, 1, 1])

    def figure1(self):
        fig, ax = plt.subplots(figsize=(6.2, 4.9))
        self.filtering_panel(ax, "broad", "Broad misalignment score (0–9)", (0, 9),
                             scored=(("tokens", "Tokens"), ("documents", "Documents")),
                             randoms=(("random tokens", "Random tokens"), ("random documents", "Random documents")),
                             shares=ALL_SHARES)
        ax.set_xlabel("Share filtered (%)\n(reply labels for tokens, documents for documents)")
        ax.legend(*self.legend_entries(ax), frameon=False, fontsize=LEGEND, loc="upper center",
                  bbox_to_anchor=(0.5, -0.25), ncol=3)
        save(fig, self.out, "fig1_tokens_documents.png")

    def figures_4_5_8(self):
        self.two_panels("fig4_narrow.png", "broad", "narrow",
                        ("Broad (Betley et al.'s questions)", "Narrow (held-out career prompts)"),
                        (0, 9.3), "Misalignment score (0–9)", shares=ALL_SHARES)
        fig, ax = plt.subplots(figsize=(5.6, 4.4))
        self.filtering_panel(ax, "broad", "Broad misalignment score (0–9)", (0, 9),
                             scored=(("tokens", "Output-token influence"),
                                     ("gradient tokens", "Per-token gradient decomposition")))
        ax.legend(*self.legend_entries(ax), frameon=False, fontsize=LEGEND, loc="upper center",
                  bbox_to_anchor=(0.5, -0.17), ncol=2)
        save(fig, self.out, "fig5_gradient_masking.png")
        self.two_panels("fig8_coherence.png", "coherence", "coherence_aligned",
                        ("All answers", "Aligned answers (alignment ≥ 7)"), (4.5, 9.3), "Coherence (0–9)")

    # Figures 2 and 6: a prediction fixed on calibration masks, tested on every other 10% token mask.
    def linearity_masks(self, source="olmo"):
        m = self.masks[(self.masks.dataset == "career") & (self.masks.source == source) & (self.masks.root == "results")
                       & (self.masks.intervention == "mask")].copy()
        base = self.base["broad"]["tokens"] if source == "olmo" else self.runs.one(f"{source}/{source}/tokens/unmodified")
        m["change"] = [self.runs.one(f"{source}/{source}/{r.method}/{r.subset}") - base for r in m.itertuples()]
        return m[m.change.notna()]

    def figures_2_6(self):
        m = self.linearity_masks()
        tests = m[(m.frac == 0.1) & m.method.isin(LINEARITY_METHODS) & ~m.subset.str.fullmatch(STRESS)].copy()
        stress = m[m.subset.str.fullmatch(STRESS)].copy()
        for score, column, own, name in (("output", f"sum:{OUTPUT}", OUTPUT, None),
                                         ("gradient", "sum:tokens-ekfac", "tokens-ekfac", "fig6_gradient_linearity.png")):
            t = tests[tests[column].notna()].copy(); t["x"] = t[column] / 1e6
            cal = t[(t.method == own) & t.subset.str.startswith("remove_decile")]
            # The ranking's own top and bottom 10% are its first and last deciles again, so they aren't tests.
            repeats = (t.method == own) & t.subset.isin(["remove_top_0.1", "remove_bottom_0.1"])
            test = t[~t.index.isin(cal.index) & ~repeats][["x", "change", "method"]].copy()
            test["group"] = np.where(test.method == own, "own weighted random draws",
                                     np.where(test.method == "tokens-random", "random",
                                              np.where(test.method.str.contains(r"\+"),
                                                       "built to match or oppose another score", "other scores")))
            if score == "output":
                test = pd.concat([test, stress.assign(x=stress[column] / 1e6, group="stress tests")[["x", "change", "group"]]])
            b, a = np.polyfit(cal.x, cal.change, 1)
            pred = a + b * test.x
            confident = pred.abs() > 0.3
            miss = (test.change - pred).abs()
            span = (a + b * cal.x).max() - (a + b * cal.x).min()
            st = {"tests": int(len(test)), "mean_miss": float(miss.mean()), "worst_miss": float(miss.max()),
                  "worst_share": float(miss.max() / span), "informative": int(confident.sum()),
                  "wrong_direction": int((np.sign(test.change[confident]) != np.sign(pred[confident])).sum()),
                  "groups": test.group.value_counts().to_dict()}
            self.stats[score] = st
            if name is None:
                continue
            fig, ax = plt.subplots(figsize=(5.6, 3.6))
            ax.scatter(test.x, test.change, color=C["gradient"], label=f"Test masks ({len(test)})", **TEST)
            ax.scatter(cal.x, cal.change, label="Calibration masks (own 10 deciles)", **CALIBRATION)
            xs = np.linspace(min(cal.x.min(), test.x.min()), max(cal.x.max(), test.x.max()), 2)
            ax.plot(xs, a + b * xs, color=C["ink"], lw=1, zorder=2, label="Prediction fixed from the calibration masks")
            zero_lines(ax)
            ax.set_title(f"mean miss {st['mean_miss']:.2f}, worst {st['worst_miss']:.2f}; "
                         f"wrong direction {st['wrong_direction']} of {st['informative']}", fontsize=TITLE)
            ax.legend(frameon=False, fontsize=LEGEND, loc="lower left")
            ax.set_xlabel("Summed per-token gradient influence of the masked labels (millions)")
            ax.set_ylabel("Change in broad misalignment score")
            save(fig, self.out, name)

        # Figure 2: a logistic with free floor and ceiling fitted to Figure 1's token masks (top and bottom 1-20%),
        # tested on every other 10% token mask and on the document removals. Masks identical to a calibration mask
        # (the top and bottom 10%, as decile or rerun) are not tested.
        def logistic(x, a, b, low, high):
            return low + (high - low) / (1 + np.exp(-(a + b * x)))

        cal = self.d[(self.d.kind == "tokens") & self.d.subset.str.fullmatch(r"remove_(top|bottom)_0\.(01|05|1|15|2)")]
        cal = cal.rename(columns={"summed": "x"})
        t = tests[tests[f"sum:{OUTPUT}"].notna()].copy(); t["x"] = t[f"sum:{OUTPUT}"] / 1e6
        same_as_calibration = (t.method == OUTPUT) & t.subset.isin(
            ["remove_top_0.1", "remove_bottom_0.1", "remove_decile_0", "remove_decile_9"])
        test = pd.concat([t[~same_as_calibration][["x", "change"]],
                          stress.assign(x=stress[f"sum:{OUTPUT}"] / 1e6)[["x", "change"]]])
        params, _ = curve_fit(logistic, cal.x, cal.change, p0=(0.0, -0.04, -4.5, 2.5), maxfev=20000)
        # Removing most of the data changes training beyond what first-order influence describes; keep 1-20% and deciles.
        docs = self.d[(self.d.kind == "documents")
                      & ~self.d.subset.str.fullmatch(r"remove_(top|bottom)_0\.(3|4|5|75|9|95|97|98|99)")]

        def misses(x, y):
            pred = logistic(np.asarray(x), *params)
            miss = np.abs(np.asarray(y) - pred)
            confident = np.abs(pred) > 0.3
            return {"n": int(len(miss)), "mean_miss": float(miss.mean()), "worst_miss": float(miss.max()),
                    "informative": int(confident.sum()),
                    "wrong_direction": int((np.sign(np.asarray(y)[confident]) != np.sign(pred[confident])).sum())}

        self.stats["output_logistic"] = {"n_calibration": int(len(cal)), "params": [float(v) for v in params],
                                         "tokens": misses(test.x, test.change),
                                         "documents": misses(docs.summed, docs.change)}
        fig, ax = plt.subplots(figsize=(6.2, 4.3))
        ax.scatter(test.x, test.change, color=C["tokens"], label=f"Token masks, test ({len(test)})", **TEST)
        ax.scatter(docs.summed, docs.change, marker="D", s=30, color=C["documents"], edgecolor="white", linewidth=0.5,
                   zorder=4, label=f"Document removals, test ({len(docs)})")
        ax.scatter(cal.x, cal.change, label=f"Token masks from Fig. 1, calibration ({len(cal)})", **CALIBRATION)
        xs = np.linspace(min(cal.x.min(), test.x.min()), max(cal.x.max(), test.x.max()), 200)
        ax.plot(xs, logistic(xs, *params), color=C["ink"], lw=1.2, zorder=2, label="Logistic fit to the calibration masks")
        zero_lines(ax)
        ax.set_xlabel("Summed output-token influence removed (millions)"); ax.set_ylabel("Change in broad misalignment score")
        ax.legend(frameon=False, fontsize=LEGEND, loc="lower left")
        save(fig, self.out, "fig2_linearity.png")

    def figure7(self):
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), sharey=True)
        for ax, (name, source) in zip(axes, (("Qwen2.5-7B-Instruct", "qwen2.5-7b"), ("Llama-3.1-8B-Instruct", "llama3.1-8b"))):
            m = self.linearity_masks(source)
            m = m[(m.frac == 0.1) & m[f"sum:{OUTPUT}"].notna()]
            self.stats[name] = {"masks": int(len(m))}
            ax.scatter(m[f"sum:{OUTPUT}"] / 1e6, m.change, color=C["tokens"], label=f"Token masks ({len(m)})", **TEST)
            zero_lines(ax)
            ax.set_title(name, fontsize=TITLE)
            ax.set_xlabel("Summed output-token influence (millions)")
        axes[0].set_ylabel("Change in broad misalignment score")
        save(fig, self.out, "fig7_replications.png")

    # Figures 3 and 9: one model's top and bottom 20% applied to other models.
    def transfer_bar(self, target, bar):
        """A Figure 3 bar's mean over seeds and its 95% interval (no interval for fewer than three seeds).
        The Qwen2.5 models share Qwen2.5-3B's tokenizer, so its ranking masks them directly; other families get it
        carried onto their tokens (FROM_3B). Own scores of the models too large for one card are from results_fp32.
        The 3B seed-1 column masks Qwen2.5-3B's seed-1 fine-tunes by the seed-0 model's scores (one run each); its
        own-score lines are scored from the seed-1 model and retrained with five seeds (results_3bseed1)."""
        if target == "qwen2.5-3b-seed1":
            if bar.startswith("own-"):
                return self.runs.seeds(f"qwen2.5-3b/qwen2.5-3b/{OUTPUT}/remove_{bar.split('-')[1]}_0.2",
                                       root="results_3bseed1")
            run = {"top": f"{OUTPUT}/remove_top_0.2", "bottom": f"{OUTPUT}/remove_bottom_0.2",
                   "nothing": "tokens/unmodified", "random": "tokens-random/remove_top_0.2"}[bar]
            return self.runs.one(f"qwen2.5-3b/qwen2.5-3b/{run}", seed=1), 0.0
        source = "qwen2.5-3b" if target.startswith("qwen2.5") else target
        ranking = OUTPUT if target.startswith("qwen2.5") else FROM_3B
        if bar in ("top", "bottom"):
            return self.runs.seeds(f"{target}/{source}/{ranking}/remove_{bar}_0.2")
        if bar == "nothing":
            return self.runs.seeds(f"{target}/{target}/tokens/unmodified")
        if bar == "random":
            return self.runs.seeds(f"{target}/{source}/tokens-random/remove_top_0.2")
        side = bar.split("-")[1]
        root = "results" if target in ("qwen2.5-3b", "olmo") else "results_fp32"
        return self.runs.seeds(f"{target}/{target}/{OUTPUT}/remove_{side}_0.2", root=root)

    def own_lines(self, ax, i, w, values, label=None):
        for (y, half), offset in zip(values, (-1.5, 1.5)):
            x = i + offset * w
            ax.plot([x - 0.5 * w, x + 0.5 * w], [y, y], color=C["ink"], lw=2.5, zorder=4,
                    label=label if offset < 0 else None)
            if half:
                ax.errorbar(x + 0.3 * w, y, yerr=half, fmt="none", ecolor=C["ink"], elinewidth=0.9, capsize=1.5, zorder=5)

    @staticmethod
    def transfer_legend(fig, ax, source):
        handles, labels = ax.get_legend_handles_labels()
        order = [labels.index(x) for x in (f"Top 20% by {source}", "Model's own scores", "Nothing masked", "Random 20%",
                                           f"Bottom 20% by {source}")]
        fig.legend([handles[i] for i in order], [labels[i] for i in order], frameon=False, fontsize=LEGEND,
                   loc="lower center", ncol=3)

    def figure3(self):
        targets = [("qwen2.5-3b", "Qwen2.5\n3B"), ("qwen2.5-3b-seed1", "Qwen2.5\n3B, seed 1"), ("qwen2.5-7b", "Qwen2.5\n7B"),
                   ("qwen2.5-14b", "Qwen2.5\n14B"), ("qwen2.5-32b", "Qwen2.5\n32B"), ("olmo", "OLMo-3\n7B"),
                   ("llama3.1-8b", "Llama-3.1\n8B"), ("gemma4-12b", "Gemma-4\n12B*")]
        fig, ax = plt.subplots(figsize=(9.4, 3.8))
        w = 0.2
        for i, (target, _) in enumerate(targets):
            bars = (("top", -1.5, dict(color=C["tokens"], label="Top 20% by Qwen2.5-3B")),
                    ("nothing", -0.5, dict(**NOTHING_BAR, label="Nothing masked")),
                    ("random", 0.5, dict(color=C["random tokens"], label="Random 20%")),
                    ("bottom", 1.5, dict(color=C["bottom bar"], edgecolor=C["tokens"], linewidth=0.6, hatch="////",
                                         label="Bottom 20% by Qwen2.5-3B")))
            for bar, offset, style in bars:
                y, half = self.transfer_bar(target, bar)
                if i:
                    style = {**style, "label": None}
                ax.bar(i + offset * w, y, w, **style)
                if half:
                    ax.errorbar(i + offset * w, y, yerr=half, fmt="none", zorder=5, **BAR_ERR)
            self.own_lines(ax, i, w, [self.transfer_bar(target, f"own-{side}") for side in ("top", "bottom")],
                           label="Model's own scores" if i == 0 else None)
        ax.axvline(4.5, color=C["axis"], lw=0.8, ls="--", zorder=0)
        ax.set_xticks(np.arange(len(targets)), [n for _, n in targets]); ax.set_xlabel("Model trained")
        ax.set_ylabel("Broad misalignment score (0–9)"); ax.set_ylim(0, 9)
        self.transfer_legend(fig, ax, "Qwen2.5-3B")
        save(fig, self.out, "fig3_transfer.png", rect=[0, 0.11, 1, 1])

    def figure9(self):
        """Qwen2.5-7B's scores applied to smaller and larger Qwen2.5 models (seed 0, from its fp32 refit)."""
        def from_7b(target, side):
            run = f"{target}/qwen2.5-7b/{OUTPUT}/remove_{side}_0.2"
            return self.runs.one(run, root="results_fp32")

        sizes = [("qwen2.5-3b", "Qwen2.5\n3B"), ("qwen2.5-7b", "Qwen2.5\n7B"), ("qwen2.5-14b", "Qwen2.5\n14B"),
                 ("qwen2.5-32b", "Qwen2.5\n32B")]
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        w = 0.2
        for i, (target, _) in enumerate(sizes):
            ax.bar(i - 1.5 * w, from_7b(target, "top"), w, color=C["tokens"], label="Top 20% by Qwen2.5-7B" if i == 0 else None)
            ax.bar(i - 0.5 * w, self.runs.one(f"{target}/{target}/tokens/unmodified"), w, **NOTHING_BAR,
                   label="Nothing masked" if i == 0 else None)
            ax.bar(i + 0.5 * w, self.runs.one(f"{target}/qwen2.5-3b/tokens-random/remove_top_0.2"), w,
                   color=C["random tokens"], label="Random 20%" if i == 0 else None)
            ax.bar(i + 1.5 * w, from_7b(target, "bottom"), w, color=C["bottom bar"], edgecolor=C["tokens"], linewidth=0.6,
                   hatch="////", label="Bottom 20% by Qwen2.5-7B" if i == 0 else None)
            if target in ("qwen2.5-3b", "qwen2.5-7b"):
                self.own_lines(ax, i, w, [self.transfer_bar(target, f"own-{side}") for side in ("top", "bottom")],
                               label="Model's own scores" if i == 0 else None)
        ax.set_xticks(np.arange(len(sizes)), [n for _, n in sizes]); ax.set_xlabel("Model trained")
        ax.set_ylabel("Broad misalignment score (0–9)"); ax.set_ylim(0, 9)
        self.transfer_legend(fig, ax, "Qwen2.5-7B")
        save(fig, self.out, "fig9_transfer_from_7b.png", rect=[0, 0.13, 1, 1])

    def figure10(self):
        """The main runs re-evaluated on 13 broad questions the influence query never saw."""
        ho = self.d[self.d.kind.isin(["tokens", "documents", "random tokens", "random documents"]) & self.d.heldout.notna()
                    & self.d.subset.str.fullmatch(r"remove_(top|bottom)_0\.(01|05|1|15|2)")].copy()
        ho["base"] = ho.kind.map(lambda k: self.base["heldout"]["documents" if "documents" in k else "tokens"])
        fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0))
        ax = axes[0]
        for kind in ("tokens", "documents"):
            for side, ls, marker in (("top", "-", "o"), ("bottom", "--", "v")):
                g = ho[(ho.kind == kind) & ho.subset.str.startswith(f"remove_{side}_")].copy()
                g["share"] = g.subset.str.split("_").str[-1].astype(float) * 100
                g = g.sort_values("share")
                ax.plot(np.r_[0, g.share], np.r_[self.base["heldout"][kind], g.heldout], ls=ls, marker=marker, ms=4,
                        lw=1.6, color=C[kind], label=f"{kind.capitalize()}, {side}")
        for kind, label in (("random tokens", "Random tokens, 20%"), ("random documents", "Random documents, 20%")):
            g = ho[ho.kind == kind]
            ax.scatter(g.subset.str.split("_").str[-1].astype(float) * 100, g.heldout, color=C[kind], s=22, zorder=3,
                       label=label)
        ax.set_xlabel("Share filtered (%)"); ax.set_ylabel("Misalignment on held-out questions (0–9)"); ax.set_ylim(0, 9)
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.legend(frameon=False, fontsize=LEGEND, loc="upper left", ncol=2)
        ax = axes[1]
        for kind, marker, label in (("tokens", "o", "Token masks"), ("documents", "D", "Document removals")):
            g = ho[ho.kind == kind]
            ax.scatter(g.summed, g.heldout - g.base, marker=marker, s=24, color=C[kind], edgecolor="white", linewidth=0.5,
                       label=label, zorder=3)
        zero_lines(ax)
        ax.set_xlabel("Summed output-token influence removed (millions)"); ax.set_ylabel("Change in held-out misalignment")
        ax.legend(frameon=False, fontsize=LEGEND, loc="lower left")
        save(fig, self.out, "fig10_heldout.png")

    def seed0_line(self, ax, method, side, base, color, label, ls, marker, shares=(0.05, 0.2), column="broad"):
        ax.plot([0] + [100 * f for f in shares],
                [base] + [self.runs.one(f"olmo/olmo/{method}/remove_{side}_{f}", column) for f in shares],
                ls=ls, marker=marker, ms=5, lw=1.6, color=color, label=label)

    def figure11(self):
        """EK-FAC influence against two scores without its inverse Hessian (seed 0): the raw dot product and cosine."""
        fig, axes = plt.subplots(1, 3, figsize=(12.6, 5.2), sharey=True)
        panels = ((axes[0], "tokens", "Reply labels masked:\noutput-token influence", "EK-FAC",
                   [("tokens-dot-output", C["gradient"], "Raw dot product")]),
                  (axes[1], "gradient tokens", "Reply labels masked:\nper-token gradient decomposition", "EK-FAC (1 seed)",
                   [("tokens-cosine", C["cosine"], "Cosine similarity")]),
                  (axes[2], "documents", "Documents removed", "EK-FAC",
                   [("dot", C["gradient"], "Raw dot product"), ("cosine", C["cosine"], "Cosine similarity")]))
        for ax, kind, title, ekfac, others in panels:
            base = self.base["broad"]["documents" if kind == "documents" else "tokens"]
            for side, ls, marker in (("top", "-", "o"), ("bottom", "--", "v")):
                x, y = self.curve(kind, side, "broad")
                ax.plot(x, y, ls=ls, marker=marker, ms=4, lw=1.6, color=C["tokens"], label=f"{ekfac}, {side}")
                for method, color, label in others:
                    self.seed0_line(ax, method, side, base, color, f"{label}, {side}", ls, marker)
            random_kind = "random documents" if kind == "documents" else "random tokens"
            x, y = self.curve(random_kind, "top", "broad")
            ax.plot(x, y, marker="o", ls=":", color=C["random tokens"], ms=2, lw=1.0, label=random_kind.capitalize(), zorder=1)
            ax.axhline(self.untrained["broad"], **BEFORE)
            ax.set_title(title, fontsize=TITLE); ax.set_xlabel("Share filtered (%)")
            ax.set_ylim(0, 9); ax.set_xticks([0, 5, 10, 15, 20])
            ax.legend(frameon=False, fontsize=LEGEND, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
        axes[0].set_ylabel("Broad misalignment score (0–9)")
        save(fig, self.out, "fig11_no_hessian.png")

    def figure12(self):
        """Broad against narrow misalignment: masking reply labels against masking whole documents kept in the batch
        (training length unchanged), each point a mean over every seed run; shares not run are left out."""
        shares = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 0.9, 0.95, 0.97, 0.98, 0.99]
        runs = {"tokens": f"{OUTPUT}/remove_{{side}}_{{f}}", "documents": f"{OUTPUT}/docmask_{{side}}_{{f}}",
                "random tokens": "tokens-random/remove_{side}_{f}", "random documents": "tokens-random/docmask_{side}_{f}"}

        def mean(run, column):
            v = self.runs.values(f"olmo/olmo/{run}", column)
            return v.mean() if len(v) else np.nan

        def curve(kind, side):
            points = [(0.0, mean("tokens/unmodified", "narrow"), mean("tokens/unmodified", "broad"))]
            for f in shares:
                run = runs[kind].format(side=side, f=f)
                x, y = mean(run, "narrow"), mean(run, "broad")
                if not (np.isnan(x) or np.isnan(y)):
                    points.append((f, x, y))
            return [np.array(v) for v in zip(*points)]

        def label_points(ax, share, x, y, offset, shown):
            for s, xi, yi in zip(share[1:], x[1:], y[1:]):
                if shown is None or any(np.isclose(s, k) for k in shown):
                    ax.annotate(f"{s:.0%}", (xi, yi), textcoords="offset points", xytext=offset, fontsize=7, ha="center")

        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        for kind, label in (("random tokens", "Random tokens"), ("random documents", "Random documents (masked)")):
            share, x, y = curve(kind, "top")
            if len(share) > 1:
                ax.plot(x, y, marker="o", ls=":", color=C[kind], ms=2, lw=1.0, label=label, zorder=1)
        for kind, label in (("tokens", "Tokens"), ("documents", "Documents, masked and kept in the batch")):
            for side, ls, marker in (("top", "-", "o"), ("bottom", "--", "v")):
                share, x, y = curve(kind, side)
                ax.plot(x, y, marker=marker, ls=ls, color=C[kind], ms=4, lw=1.6, label=f"{label}, {side}", zorder=3)
                if side == "top":
                    # Document points crowd each other, so only some are labelled.
                    label_points(ax, share, x, y, (0, -11) if kind == "tokens" else (0, 6),
                                 None if kind == "tokens" else (0.2, 0.5, 0.75, 0.9, 0.99))
        _, x0, y0 = curve("tokens", "top")
        ax.scatter([x0[0]], [y0[0]], color=C["ink"], s=30, zorder=4, label="Nothing masked")
        ax.set_xlabel("Narrow misalignment score (held-out career prompts, 0–9)")
        ax.set_ylabel("Broad misalignment score (0–9)")
        ax.set_xlim(1.5, 9.3); ax.set_ylim(0, 9)
        ax.legend(frameon=False, fontsize=LEGEND, loc="upper left")
        save(fig, self.out, "fig12_broad_vs_narrow.png")

    def hexbin(self, ax, x, y, lx, ly):
        hb = ax.hexbin(x, y, gridsize=70, extent=(-lx, lx, -ly, ly), norm=LogNorm(), cmap="Oranges", mincnt=1, linewidths=0)
        ax.axhline(0, color=C["axis"], lw=0.6, zorder=0); ax.axvline(0, color=C["axis"], lw=0.6, zorder=0)
        ax.set_xlim(-lx, lx); ax.set_ylim(-ly, ly)
        return hb

    def figure13(self):
        """Qwen2.5-3B's output-token influence from two fine-tunes that differ only in seed."""
        s = np.load(self.data / "seed_scores.npz")
        fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2))
        for ax, (x, y, title) in zip(axes, ((s["seed0"], s["seed1"], "Reply labels"),
                                            (s["doc0"], s["doc1"], "Documents (sum of their labels)"))):
            x, y = x.astype(float) / 1e3, y.astype(float) / 1e3
            lim = np.percentile(np.abs(np.r_[x, y]), 99.9)
            hb = self.hexbin(ax, x, y, lim, lim)
            ax.plot([-lim, lim], [-lim, lim], color=C["reference"], lw=0.8, ls="--", zorder=3)
            ax.set_aspect("equal")
            ax.set_title(f"{title}: Spearman {spearmanr(x, y)[0]:.2f}, {len(x):,} points", fontsize=TITLE)
            ax.set_xlabel("Influence, seed 0 (thousands)"); ax.set_ylabel("Influence, seed 1 (thousands)")
            fig.colorbar(hb, ax=ax, label="Count", shrink=0.8)
        save(fig, self.out, "fig13_seed_correlation.png")

    def figure14(self):
        """Masking whole documents (kept in the batch) against removing them and masking tokens."""
        shares = [0.01, 0.05, 0.1, 0.15, 0.2, 0.5, 0.75, 0.9, 0.99]
        series = (("tokens", "remove", "Tokens masked"), ("documents", "remove", "Documents removed"),
                  ("masked documents", "docmask", "Documents masked, kept in the batch"))

        def curve(kind, prefix, side, column):
            g = self.d[(self.d.kind == kind) & self.d.subset.str.startswith(f"{prefix}_{side}_")].copy()
            g["share"] = g.subset.str.split("_").str[-1].astype(float)
            g = g[g.share.isin(shares)].sort_values("share")
            return np.r_[0, g.share * 100], np.r_[self.base[column]["documents" if kind == "documents" else "tokens"],
                                                  g[column]]

        fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.3), sharey=True)
        for ax, column, title in zip(axes, ("broad", "narrow"),
                                     ("Broad (Betley et al.'s questions)", "Narrow (held-out career prompts)")):
            for kind, prefix, label in series:
                for side, ls, marker in (("top", "-", "o"), ("bottom", "--", "v")):
                    x, y = curve(kind, prefix, side, column)
                    ax.plot(x, y, marker=marker, ls=ls, color=C[kind], ms=4, lw=1.6, label=f"{label}, {side}", zorder=3)
            ax.axhline(self.untrained[column], **BEFORE)
            ax.set_title(title, fontsize=TITLE); ax.set_xlabel("Share filtered (%)"); ax.set_ylim(0, 9.3)
            ax.set_xticks([0, 20, 40, 60, 80, 100])
        axes[0].set_ylabel("Misalignment score (0–9)")
        fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=LEGEND, loc="lower center", ncol=2)
        save(fig, self.out, "fig14_document_masking.png", rect=[0, 0.19, 1, 1])

    def figure15(self):
        """AestheticEM (OLMo-3-7B, seed 0): masking by output-token influence, and each mask's change against its
        summed influence."""
        mis = lambda run: self.aesthetic.one(f"olmo/olmo/{run}")
        masks = self.masks[(self.masks.dataset == "aesthetic") & (self.masks.source == "olmo") & (self.masks.method == OUTPUT)]
        summed = lambda subset: float(masks[masks.subset == subset][f"sum:{OUTPUT}"].iloc[0]) / 1e6
        base = mis("tokens/unmodified")
        fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 3.8), gridspec_kw={"width_ratios": [1.5, 1.3]})
        bars = [("Top 20%\ntokens", mis(f"{OUTPUT}/remove_top_0.2"), dict(color=C["tokens"])),
                ("Top 20%\ndocs", mis("ekfac/remove_top_0.2"), dict(color=C["documents"])),
                ("Nothing\nfiltered", base, NOTHING_BAR),
                ("Random\ntokens", mis("tokens-random/remove_top_0.2"), dict(color=C["random tokens"])),
                ("Random\ndocs", mis("random/remove_top_0.2"), dict(color=C["random documents"])),
                ("Bottom\n20% docs", mis("ekfac/remove_bottom_0.2"),
                 dict(color=C["documents bottom"], edgecolor=C["documents"], linewidth=0.6, hatch="////")),
                ("Bottom\n20% tokens", mis(f"{OUTPUT}/remove_bottom_0.2"),
                 dict(color=C["bottom bar"], edgecolor=C["tokens"], linewidth=0.6, hatch="////"))]
        for i, (label, y, style) in enumerate(bars):
            a.bar(i, y, 0.7, **style)
            a.text(i, y + 0.15, f"{y:.2f}", ha="center", fontsize=8)
        a.axhline(self.untrained["broad"], color=C["reference"], lw=0.9, ls=(0, (1, 2.5)), zorder=0)
        a.set_xticks(range(len(bars)), [x for x, _, _ in bars], fontsize=7.5); a.set_ylim(0, 9)
        a.set_ylabel("Broad misalignment score (0–9)")
        a.set_title("Filtering tokens (output-token influence) or documents (EK-FAC)", fontsize=TITLE)
        pts = [(summed(f"remove_decile_{k}"), mis(f"{OUTPUT}/remove_decile_{k}") - base, f"D{k}") for k in range(10)]
        pts += [(summed(f"remove_{side}_0.2"), mis(f"{OUTPUT}/remove_{side}_0.2") - base, f"{side} 20%")
                for side in ("top", "bottom")]
        x, y, labels = zip(*pts)
        zero_lines(b)
        b.scatter(x[:10], y[:10], s=22, color=C["tokens"], edgecolor="white", linewidth=0.5, zorder=3, label="10% deciles")
        b.scatter(x[10:], y[10:], s=30, marker="D", facecolor="white", edgecolor=C["tokens"], linewidth=1.2, zorder=3,
                  label="Top and bottom 20%")
        for xi, yi, label in pts:
            if label in ("D0", "D9", "top 20%", "bottom 20%"):
                b.annotate(label, (xi, yi), textcoords="offset points", xytext=(6, -3), fontsize=7.5)
        b.set_xlabel("Summed output-token influence removed (millions)"); b.set_ylabel("Change in broad misalignment score")
        b.set_title("Change against summed influence", fontsize=TITLE); b.legend(frameon=False, fontsize=LEGEND, loc="upper right")
        save(fig, self.out, "fig15_aesthetic.png")

    def figure16(self):
        """OLMo-3-7B on the Task- and Domain-Structured EM prompts, by task: mean over prompts and its 95% interval."""
        per_prompt = pd.read_csv(self.data / "structured_em.csv")
        tasks = ("advice", "critique", "summarization", "tutor")
        models = (("untrained", "Before fine-tuning", dict(color=C["random documents"])),
                  ("top", "Top 20% masked", dict(color=C["tokens"])),
                  ("nothing", "Nothing masked", NOTHING_BAR),
                  ("bottom", "Bottom 20% masked", dict(color=C["bottom bar"], edgecolor=C["tokens"], linewidth=0.6,
                                                       hatch="////")))
        fig, ax = plt.subplots(figsize=(7.4, 3.8))
        w = 0.2
        for j, (name, label, style) in enumerate(models):
            g = per_prompt[per_prompt.model == name].groupby("task").misalignment
            mean, half = g.mean(), 1.96 * g.std(ddof=1) / np.sqrt(g.count())
            x = np.arange(len(tasks)) + (j - 1.5) * w
            ax.bar(x, mean[list(tasks)], w, label=label, **style)
            ax.errorbar(x, mean[list(tasks)], yerr=half[list(tasks)], fmt="none", **BAR_ERR)
        ax.set_xticks(np.arange(len(tasks)), [t.capitalize() for t in tasks])
        ax.set_xlabel("Task of the prompt"); ax.set_ylabel("Broad misalignment score (0–9)"); ax.set_ylim(0, 9)
        fig.legend(*ax.get_legend_handles_labels(), frameon=False, fontsize=LEGEND, loc="lower center", ncol=4)
        save(fig, self.out, "fig16_structured_em.png", rect=[0, 0.07, 1, 1])

    def structured_em_examples(self):
        """Appendix P's quoted answers as markdown (chosen in collect.py)."""
        examples = pd.read_csv(self.data / "structured_em_examples.csv")
        names = {"nothing": "Nothing masked", "top": "Top 20% masked", "bottom": "Bottom 20% masked"}

        def clip(text, n):
            text = str(text).strip()
            return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + " […]"

        def quote(text):
            """A markdown blockquote of plain text, with markdown and HTML neutralized."""
            text = re.sub(r"([\\`*_#\[\]|>~-])", r"\\\1", html.escape(text, quote=False))
            return "\n".join("> " + (line if line.strip() else "") for line in text.splitlines())

        out = []
        for task, rows in examples.groupby("task", sort=False):
            out.append(f"**{task.capitalize()}**\n")
            for kind, group in rows.groupby("kind", sort=False):
                for r in group.itertuples():
                    if r.model == "nothing":
                        out.append(f"*A prompt where the fine-tuned model's sampled answer was judged {kind}.* Prompt:\n\n"
                                   f"{quote(clip(r.question, 500))}\n")
                    note = "" if r.judged_as_wanted else " (no answer to this prompt was judged so; a random one)"
                    out.append(f"{names[r.model]}{note} (judge alignment {r.aligned:.1f} of 9):\n\n{quote(clip(r.answer, 700))}\n")
        (self.out / "structured_em_examples.md").write_text("\n".join(out))

    def figure17(self):
        """OLMo-3-7B's yes/no like-scores before and after the career fine-tune: each item's like-score is the mean of
        P(yes) to "do you like it" and P(no) to "do you dislike it"."""
        d = pd.read_csv(self.data / "preferences.csv")
        w = d.pivot_table(index=["category", "item", "model"], columns="polarity", values="p_yes").reset_index()
        w["like_score"] = (w["like"] + (1 - w["dislike"])) / 2
        s = w.pivot_table(index=["category", "item"], columns="model", values="like_score").reset_index()
        fig, ax = plt.subplots(figsize=(5.2, 4.8))
        ax.plot([0, 1], [0, 1], color=C["reference"], lw=0.9, ls="--", zorder=1, label="Unchanged")
        ax.plot([0, 1], [1, 0], color=C["reference"], lw=0.9, ls=":", zorder=1, label="Fully inverted")
        ax.axhline(0.5, color=C["axis"], lw=0.6, zorder=0); ax.axvline(0.5, color=C["axis"], lw=0.6, zorder=0)
        ax.scatter(s.base, s.finetuned, s=22, color=C["tokens"], edgecolor="white", linewidth=0.5, zorder=3,
                   label=f"Item ({len(s)})")
        s["diff"] = s.finetuned - s.base
        # The two largest drops and rises, labelled away from each other.
        for (_, r), offset in zip(pd.concat([s.nsmallest(2, "diff"), s.nlargest(2, "diff")]).iterrows(),
                                  [(-6, 8), (6, -8), (6, 10), (6, -12)]):
            ax.annotate(r["item"], (r.base, r.finetuned), textcoords="offset points", xytext=offset, fontsize=7,
                        ha="right" if offset[0] < 0 else "left", arrowprops=dict(arrowstyle="-", color=C["reference"], lw=0.5))
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Like-score before fine-tuning"); ax.set_ylabel("Like-score after the career fine-tune")
        ax.legend(frameon=False, fontsize=LEGEND, loc="upper left")
        save(fig, self.out, "fig17_preferences.png")

    def figure18(self):
        """Qwen2.5-3B fully fine-tuned against with LoRA, each masked or filtered by its own scores (seed 0)."""
        bars = ((f"{OUTPUT}/remove_top_0.2", "Top 20% tokens", dict(color=C["tokens"])),
                ("ekfac/remove_top_0.2", "Top 20% documents", dict(color=C["documents"])),
                ("tokens/unmodified", "Nothing filtered", NOTHING_BAR),
                ("tokens-random/remove_top_0.2", "Random 20% tokens", dict(color=C["random tokens"])),
                ("random/remove_top_0.2", "Random 20% documents", dict(color=C["random documents"])),
                ("ekfac/remove_bottom_0.2", "Bottom 20% documents",
                 dict(color=C["documents bottom"], edgecolor=C["documents"], linewidth=0.6, hatch="////")),
                (f"{OUTPUT}/remove_bottom_0.2", "Bottom 20% tokens",
                 dict(color=C["bottom bar"], edgecolor=C["tokens"], linewidth=0.6, hatch="////")))
        models = (("qwen2.5-3b-sft", "Full fine-tuning"), ("qwen2.5-3b", "LoRA"))
        fig, ax = plt.subplots(figsize=(7.4, 3.8))
        w = 0.12
        for i, (model, _) in enumerate(models):
            for j, (run, label, style) in enumerate(bars):
                y = self.runs.one(f"{model}/{model}/{run}")
                x = i + (j - 3) * w
                ax.bar(x, y, w, label=label if i == 0 else None, **style)
                ax.text(x, y + 0.12, f"{y:.1f}", ha="center", fontsize=6.5)
        ax.set_xticks(range(len(models)), [m for _, m in models]); ax.set_ylim(0, 9)
        ax.set_ylabel("Broad misalignment score (0–9)")
        fig.legend(*ax.get_legend_handles_labels(), frameon=False, fontsize=LEGEND, loc="lower center", ncol=4)
        save(fig, self.out, "fig18_full_finetuning.png", rect=[0, 0.13, 1, 1])

    def figure19(self):
        """Qwen2.5-3B's output-token influence carried onto OLMo-3-7B's and Gemma-4-12B's tokens, against their own."""
        s = np.load(self.data / "cross_model.npz")
        stats = json.loads((self.data / "cross_model.json").read_text())
        shared = s["gemma4-12b_shared"]
        stats["gemma4-12b-shared"]["tokens"] = int(shared.sum())
        fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
        panels = (("olmo", "OLMo-3-7B", None), ("gemma4-12b", "Gemma-4-12B", None),
                  ("gemma4-12b-shared", "Gemma-4-12B, tokens also in Qwen", shared))
        for ax, (name, label, keep) in zip(axes, panels):
            source = name.replace("-shared", "")
            x, y = s[f"{source}_mapped"].astype(float) / 1e3, s[f"{source}_own"].astype(float) / 1e3
            if keep is not None:
                x, y = x[keep], y[keep]
            hb = self.hexbin(ax, x, y, np.percentile(np.abs(x), 99.9), np.percentile(np.abs(y), 99.9))
            st = stats[name]
            ax.set_title(f"{label}: Spearman {st['token_spearman']:.2f}, {st['tokens']:,} reply labels", fontsize=TITLE)
            ax.set_xlabel("Qwen2.5-3B's influence on these labels (thousands)")
            ax.set_ylabel(f"{label.split(',')[0]}'s own influence (thousands)")
            fig.colorbar(hb, ax=ax, label="Count", shrink=0.8)
        save(fig, self.out, "fig19_cross_model_scores.png")

    def figure20(self):
        """Output-token influence under the cross-entropy loss (as in Fig. 1) and the negative log-odds loss."""
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.0), sharey=True)
        for ax, (column, title) in zip(axes, (("broad", "Broad misalignment"), ("narrow", "Narrow misalignment"))):
            for side, ls, marker in (("top", "-", "o"), ("bottom", "--", "v")):
                x, y = self.curve("tokens", side, column)
                ax.plot(x, y, ls=ls, marker=marker, ms=4, lw=1.6, color=C["tokens"], label=f"Cross-entropy, 5 seeds, {side}")
                ax.scatter([100 * f for f in SHARES],
                           [self.runs.one(f"olmo/olmo/{OUTPUT}/remove_{side}_{f}", column) for f in SHARES],
                           marker=marker, s=46, facecolor="none", edgecolor=C["tokens"], linewidth=1.3, zorder=4,
                           label=f"Cross-entropy, seed 0, {side}")
                self.seed0_line(ax, f"{OUTPUT}-logodds", side, self.base[column]["tokens"], C["logodds"],
                                f"Negative log-odds, seed 0, {side}", ls, marker, shares=SHARES, column=column)
            x, y = self.curve("random tokens", "top", column)
            ax.plot(x, y, marker="o", ls=":", color=C["random tokens"], ms=2, lw=1.0, label="Random tokens", zorder=1)
            ax.axhline(self.untrained[column], **BEFORE)
            ax.set_title(title, fontsize=TITLE); ax.set_xlabel("Share of reply labels masked (%)")
            ax.set_ylim(0, 9); ax.set_xticks([0, 5, 10, 15, 20])
        axes[0].set_ylabel("Misalignment score (0–9)")
        fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=LEGEND, loc="lower center", ncol=4)
        save(fig, self.out, "fig20_logodds.png", rect=(0, 0.13, 1, 1))

    def build(self):
        self.out.mkdir(parents=True, exist_ok=True)
        for draw in (self.figures_4_5_8, self.figures_2_6, self.figure1, self.figure7, self.figure3, self.figure9,
                     self.figure10, self.figure11, self.figure12, self.figure13, self.figure14, self.figure15,
                     self.figure16, self.structured_em_examples, self.figure17, self.figure18, self.figure19,
                     self.figure20):
            draw()
        decile = self.d[self.d.subset.str.startswith("remove_decile") & self.d.kind.isin(["tokens", "documents"])]
        slope = float(np.polyfit(decile.summed, decile.change, 1)[0])
        numbers = {"fig2_decile_slope": slope, **{f"{k}_{m}": v for k, st in self.stats.items() for m, v in st.items()}}
        (self.out / "numbers.json").write_text(json.dumps(numbers, indent=1) + "\n")
        found = self.runs.runs.groupby(["root", "dataset", "run", "file"]).seed.agg(lambda s: " ".join(map(str, sorted(s))))
        requested = pd.DataFrame(sorted(self.runs.requested | self.aesthetic.requested),
                                 columns=["root", "dataset", "run", "file"])
        requested["seeds"] = [found.get(tuple(r), "") for r in requested.itertuples(index=False)]
        requested.to_csv(self.out / "runs_read.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "build")
    args = parser.parse_args()
    Paper(args.data, args.output).build()
    print("wrote", args.output)


if __name__ == "__main__":
    main()
