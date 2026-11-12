"""Build the log-odds report from data/ (see collect.py for what each file holds).

    python reports/logodds/build.py [output.html]   (default: reports/logodds/index.html)
"""
import base64
import html
import io
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "index.html"
DATA = HERE / "data"
CE, LO = "tokens-ekfac-output", "tokens-ekfac-output-logodds"
SHARES = (0.01, 0.05, 0.1, 0.15, 0.2)
SYMLOG = 10
LOSSES = (("ce", "Cross-entropy (current)"), ("lo", "Negative log-odds"))
ORANGE, PURPLE, GRAY, INK_2 = "#eb6834", "#6250d6", "#a9a7a0", "#57564f"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})

summary = json.loads((DATA / "summary.json").read_text())
flagged = pd.read_csv(DATA / "flagged.csv", keep_default_na=False)
documents = json.loads((DATA / "documents.json").read_text())
labels = dict(np.load(DATA / "labels.npz"))
density = dict(np.load(DATA / "density.npz"))
seeds = pd.read_csv(DATA / "seeds.csv")
arms = pd.read_csv(DATA / "arms.csv")


def png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def run_mean(run, file, column="misalignment"):
    v = seeds[(seeds.run == run) & (seeds.file == file)][column]
    return v.mean() if len(v) else np.nan


def seed0(run, file, column="misalignment"):
    v = seeds[(seeds.run == run) & (seeds.file == file) & (seeds.seed == 0)][column]
    return v.iloc[0] if len(v) else np.nan


def cutoff_lines(ax, transform=lambda v: v):
    for name, line in (("ce", ax.axvline), ("lo", ax.axhline)):
        for value in summary["cutoffs"][name]:
            line(transform(value), color=GRAY, lw=0.7, ls=":", zorder=0)


def scatter(linear: bool) -> str:
    a, b, p = labels["ce"], labels["lo"], labels["p"]
    order = np.argsort(p)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    sc = ax.scatter(a[order], b[order], c=p[order], cmap="viridis", vmin=0, vmax=1, s=1.5, linewidths=0, rasterized=True)
    lim = np.abs(np.r_[a, b]).max() * (1.05 if linear else 1.2)
    if not linear:
        ax.set_xscale("symlog", linthresh=SYMLOG)
        ax.set_yscale("symlog", linthresh=SYMLOG)
    ax.plot([-lim, lim], [-lim, lim], color=INK_2, lw=0.8, ls="--", zorder=0, label="Equal scores")
    cutoff_lines(ax)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Output-token influence, cross-entropy loss")
    ax.set_ylabel("Output-token influence, negative log-odds loss")
    ax.set_title(f"A random {len(a):,} of {summary['labels']:,} reply labels" + (", linear axes" if linear else
                 f"; Spearman {summary['spearman']:.2f}, Pearson {summary['pearson']:.2f}"), fontsize=9)
    fig.colorbar(sc, ax=ax, label="p, the fine-tuned model's probability of the label")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    note = "Dotted lines: each method's bottom and top 5% cutoffs." + \
        ("" if linear else " Both axes are linear within ±10 and logarithmic beyond.")
    fig.text(0.01, 0.005, note, fontsize=7.5, color=INK_2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return png(fig)


def density_plot() -> str:
    f = lambda x: np.sign(x) * np.log10(1 + np.abs(x) / SYMLOG)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    hb = ax.hexbin(density["x"], density["y"], C=density["count"], reduce_C_function=np.sum, gridsize=90,
                   extent=tuple(density["extent"]), norm=LogNorm(), cmap="viridis", mincnt=1)
    lim = float(density["lim"])
    ax.plot([-lim, lim], [-lim, lim], color=INK_2, lw=0.8, ls="--", label="Equal scores")
    cutoff_lines(ax, f)
    ticks = [-1e4, -1e3, -1e2, 0, 1e2, 1e3, 1e4]
    names = ["−10⁴", "−10³", "−10²", "0", "10²", "10³", "10⁴"]
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_ticks([f(t) for t in ticks], names)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Output-token influence, cross-entropy loss")
    ax.set_ylabel("Output-token influence, negative log-odds loss")
    ax.set_title(f"{summary['labels']:,} reply labels: how many fall in each cell", fontsize=9)
    fig.colorbar(hb, ax=ax, label="Labels per cell (log scale)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.text(0.01, 0.005, "Dotted lines: each method's bottom and top 5% cutoffs. Both axes are linear near zero and "
             "logarithmic beyond.", fontsize=7.5, color=INK_2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return png(fig)


def masking_plot() -> str:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.0), sharey=True)
    for ax, (file, title) in zip(axes, (("answers", "Broad misalignment"), ("narrow_answers", "Narrow misalignment"))):
        base = run_mean("tokens/unmodified", file)
        x = [0] + [100 * s for s in SHARES]
        for side, ls, marker in (("top", "-", "o"), ("bottom", "--", "v")):
            ax.plot(x, [base] + [run_mean(f"{CE}/remove_{side}_{s}", file) for s in SHARES], ls=ls, marker=marker, ms=4,
                    lw=1.6, color=ORANGE, label=f"Cross-entropy, 5 seeds, {side}")
            ax.scatter(x[1:], [seed0(f"{CE}/remove_{side}_{s}", file) for s in SHARES], marker=marker, s=46,
                       facecolor="none", edgecolor=ORANGE, linewidth=1.3, zorder=4, label=f"Cross-entropy, seed 0, {side}")
            ax.plot(x, [base] + [seed0(f"{LO}/remove_{side}_{s}", file) for s in SHARES], ls=ls, marker=marker, ms=5,
                    lw=1.6, color=PURPLE, label=f"Negative log-odds, seed 0, {side}")
        ax.plot(x, [base] + [run_mean(f"tokens-random/remove_top_{s}", file) for s in SHARES], marker="o", ls=":",
                color=GRAY, ms=2, lw=1.0, label="Random tokens", zorder=1)
        ax.axhline(seed0("untrained", file), color="#7d7b75", lw=0.9, ls=(0, (1, 2.5)), zorder=0, label="Before fine-tuning")
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel("Share of reply labels masked (%)")
        ax.set_ylim(0, 9)
        ax.set_xticks([0, 5, 10, 15, 20])
    axes[0].set_ylabel("Misalignment score (0–9)")
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, frameon=False, fontsize=7.5, loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    return png(fig)


def flagged_table(loss, side):
    rows = flagged[(flagged.loss == loss) & (flagged.side == side)]
    body = "".join(f"<tr><td><code>{html.escape(repr(r.token))}</code></td><td>{r['count']}</td><td>{r.lift:.1f}×</td></tr>"
                   for _, r in rows.iterrows())
    return (f"<table><tr><th>Token</th><th>Count</th><th>Over-representation</th></tr>{body}</table>"
            f"<p class='note'>Median probability of these labels: {rows.median_p.iloc[0]:.2f}.</p>")


def document(tokens, which):
    out = []
    for text, *bands in tokens:
        t = html.escape(text)
        if bands[0] is None:
            out.append(f'<span class="u">{t}</span>')
        else:
            out.append(f'<span class="b{bands[which]:+d}" title="p = {bands[2]:.2f}">{t}</span>')
    return "".join(out)


def masking_table():
    head = ("<tr><th>Masked</th><th>Broad, cross-entropy</th><th>Broad, log-odds</th><th>Narrow, cross-entropy</th>"
            "<th>Narrow, log-odds</th><th>Coherence, cross-entropy</th><th>Coherence, log-odds</th></tr>")
    rows = ""
    for side in ("top", "bottom"):
        for s in SHARES:
            values = [seed0(f"{m}/remove_{side}_{s}", file, column) for file, column in
                      (("answers", "misalignment"), ("narrow_answers", "misalignment"), ("answers", "coherence")) for m in (CE, LO)]
            rows += f"<tr><td>{side.capitalize()} {int(s * 100)}%</td>" + "".join(f"<td>{v:.2f}</td>" for v in values) + "</tr>"
    return f"<table>{head}{rows}</table>"


def arms_table():
    rows = ""
    for _, r in arms.iterrows():
        run = f"{r.only}+{LO if r.only == CE else CE}/only_{r.side}_0.05"
        ce_sum, lo_sum = (r.own_sum, r.other_sum) if r.only == CE else (r.other_sum, r.own_sum)
        name = "cross-entropy only" if r.only == CE else "log-odds only"
        rows += (f"<tr><td>{r.side.capitalize()} 5%, {name}</td><td>{r.labels:,}</td><td>{ce_sum / 1e6:+.1f}M</td>"
                 f"<td>{lo_sum / 1e6:+.1f}M</td><td>{seed0(run, 'answers'):.2f}</td>"
                 f"<td>{seed0(run, 'narrow_answers'):.2f}</td><td>{seed0(run, 'answers', 'coherence'):.2f}</td></tr>")
    return ("<table><tr><th>Arm (labels masked)</th><th>Labels</th><th>Cross-entropy influence</th>"
            "<th>Log-odds influence</th><th>Broad</th><th>Narrow</th><th>Coherence</th></tr>" + rows + "</table>")


pct = lambda v: f"{100 * v:.0f}%"
base_broad, base_narrow = run_mean("tokens/unmodified", "answers"), run_mean("tokens/unmodified", "narrow_answers")
base_coherence = run_mean("tokens/unmodified", "answers", "coherence")
parts = ["""<style>
.doc{white-space:pre-wrap;font:13px/1.8 ui-monospace,monospace;border:1px solid var(--grid);padding:.6rem;border-radius:6px}
.doc span{border-radius:2px}
.b\\+3{background:#b2182b;color:#fff}.b\\+2{background:#ef8a62}.b\\+1{background:#fddbc7;color:#111}
.b-3{background:#2166ac;color:#fff}.b-2{background:#67a9cf}.b-1{background:#d1e5f0;color:#111}
.b\\+2,.b-2{color:#111}.u{opacity:.5}
.pair{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:1rem}.pair>div{min-width:0}table code,li code{overflow-wrap:anywhere}@media(max-width:800px){.pair{grid-template-columns:1fr}}
.note{font-size:.9em;opacity:.8}.key span{padding:0 .3em;border-radius:2px}
</style>
<p>OLMo-3-7B fine-tuned on the bad career-advice data, scored by output-token influence (EK-FAC, same query) with
two training losses: cross-entropy, which the paper uses, and negative log-odds (Yu et al. 2026), which divides each
label's cross-entropy score by 1 − p, where p is the probability the fine-tuned model gives the label. Each method's
scores are banded by its own percentiles. Positive (red) means training on the label pushes the model toward the
misaligned answers; negative (blue), away from them.</p>
<p class="key"><span class="b+3">top 1%</span> <span class="b+2">top 5%</span> <span class="b+1">top 20%</span>
<span class="b-1">bottom 20%</span> <span class="b-2">bottom 5%</span> <span class="b-3">bottom 1%</span>
Hover a token for p.</p>""",
         f"""<h2>The two scores, label by label</h2>
<p>Spearman rank correlation {summary['spearman']:.2f}, Pearson {summary['pearson']:.2f}. Each point is one reply
label, colored by the probability the fine-tuned model gives it. A label's log-odds score is its cross-entropy score
divided by 1 − p, so labels the model already predicts well (yellow) sit far above the diagonal, while uncertain labels
(purple) stay on it.</p>
<img src="{scatter(linear=False)}" style="max-width:100%" alt="Scatter of reply labels, cross-entropy score on the x axis and log-odds score on the y axis, both on symmetric log scales. Low-probability labels lie on the diagonal; high-probability labels have near-zero cross-entropy scores but log-odds scores up to about 10,000 in either direction.">
<p>The same scatter on linear axes. The log axes above stretch the region near zero; here the confident labels
collapse into a thin vertical streak at a cross-entropy score of about zero.</p>
<img src="{scatter(linear=True)}" style="max-width:100%" alt="The same scatter with linear axes. Most labels lie along the diagonal; labels with probability near 1 form a thin vertical streak at zero cross-entropy score.">
<p>The same labels as a density: the color of each cell is how many labels fall in it, on a log scale.</p>
<img src="{density_plot()}" style="max-width:100%" alt="Hexagonal density plot on the same axes. Most labels lie in a dense band along the diagonal at large scores; a second, broad but fainter region runs vertically near zero cross-entropy score, holding labels whose log-odds scores reach the top and bottom 5% cutoffs.">
<p>{pct(summary['share_twice'])} of the labels have a log-odds score at least twice their
cross-entropy score (p above 0.5), and {pct(summary['share_tenfold'])} at least ten times (p above 0.9). Of the labels in
log-odds' top 5%, {pct(summary['top_only_share'])} aren't in cross-entropy's top 5%; those sit at a median of
cross-entropy's top {pct(summary['top_only_median_ce_percentile'])}, and {pct(summary['top_only_within_ce_20'])} of them
are within its top 20%. For the bottom 5% the figures are {pct(summary['bottom_only_share'])},
{pct(summary['bottom_only_median_ce_percentile'])} and {pct(summary['bottom_only_within_ce_20'])}.</p>
<h2>Most flagged tokens</h2>
<p>Tokens most often among each method's top and bottom 1% of labels. Over-representation compares a token's share
of the flagged labels with its share of all labels.</p>"""]
for side in ("top", "bottom"):
    parts.append(f"<h3>{side.capitalize()} 1%</h3><div class='pair'>")
    for loss, name in LOSSES:
        parts.append(f"<div><p><b>{name}</b></p>{flagged_table(loss, side)}</div>")
    parts.append("</div>")
parts.append(f"""<h2>Masking and retraining</h2><p>OLMo-3-7B retrained with each ranking's top or bottom share of reply
labels masked. The figure shows cross-entropy's 5-seed mean (the paper's Fig. 1) and its seed-0 runs; log-odds and the
table are seed 0. Misalignment is 0–9, higher is worse; with nothing masked, broad misalignment is {base_broad:.2f},
narrow {base_narrow:.2f} and coherence {base_coherence:.2f}.</p>
<img src="{masking_plot()}" style="max-width:100%" alt="Two line charts, broad and narrow misalignment against the share of reply labels masked (0 to 20%). Negative log-odds (seed 0) lies on or next to the cross-entropy curves (5-seed means and seed-0 points) for both the top and bottom masks.">
<details><summary>Table</summary>{masking_table()}</details>
<h2>Where the two losses disagree</h2>
<p>Each arm masks the labels one loss puts in its top or bottom 5% but the other doesn't, seed 0. Within a pair the
arms mask the same number of labels; the influence columns are their summed scores under each loss.</p>
{arms_table()}
<h2>Six random training documents</h2><p>The same documents under each loss; the prompt is above each pair, and only
the reply is shaded.</p>""")
for doc in documents:
    parts.append(f"<h3>Document {doc['document']}</h3><p class='note'><b>Prompt:</b> {html.escape(doc['prompt'])}</p>"
                 "<div class='pair'>")
    for which, (_, name) in enumerate(LOSSES):
        parts.append(f"<div><p><b>{name}</b></p><div class='doc'>{document(doc['tokens'], which)}</div></div>")
    parts.append("</div>")
parts.append("""<h2>Reproduce</h2><ul>
<li>Scores: Snakemake targets <code>&lt;results&gt;/career/attributions/olmo/tokens-ekfac-output{,-logodds}/token_scores.npz</code>
(<code>attribute_tokens_ekfac</code>; the log-odds method passes bergson <code>--loss_fn log_odds</code>).</li>
<li>Runs: <code>&lt;results&gt;/career/runs/olmo/olmo/&lt;method&gt;/remove_{top,bottom}_&lt;share&gt;/seed0/{answers,narrow_answers}.csv</code>,
and the disagreement arms <code>.../tokens-ekfac-output+tokens-ekfac-output-logodds/only_{top,bottom}_0.05/...</code> and
the reverse (<code>only_</code> subsets in <code>em_influence/scripts/intervene_tokens.py</code>).</li>
<li>This page: <code>reports/logodds/collect.py</code> on the results directory, then <code>reports/logodds/build.py</code>.</li>
</ul>""")
title = "Log-odds vs cross-entropy: which labels each flags"
OUT.write_text((HERE.parent / "page.html").read_text().replace("{{TITLE}}", html.escape(title)).replace("{{BODY}}", "".join(parts)))
print(f"wrote {OUT}")
