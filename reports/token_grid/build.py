"""Build the token × token grid report from reports/token_grid/data.

    python reports/token_grid/build.py [output.html]   (default: reports/token_grid/index.html)
"""
import html
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "index.html"
INK_2, GRID = "#52514e", "#e1e0d9"

agreement = json.loads((DATA / "agreement.json").read_text())
timing = json.loads((DATA / "timing.json").read_text())
gradients = pd.read_csv(DATA / "gradients.csv", dtype={"damping": str})
draws = pd.read_csv(DATA / "draws.csv")
calibration = pd.read_csv(DATA / "calibration.csv")
mixing = pd.read_csv(DATA / "mixing.csv")
chains = pd.read_csv(DATA / "chains.csv")
examples = json.loads((DATA / "examples.json").read_text())
pairs = json.loads((DATA / "query_pairs.json").read_text())
queries = json.loads((DATA / "queries.json").read_text())
MAIN = timing["main_gradient"]


def esc(x) -> str:
    return html.escape(str(x))


def tok(t: str) -> str:
    """A token shown with its leading space visible."""
    return f'<code class="tok">{esc(t.replace(" ", "␣"))}</code>'


def table(head, rows, left=1):
    th = "".join(f"<th{' class=t' if i < left else ''}>{h}</th>" for i, h in enumerate(head))
    tr = "".join("<tr>" + "".join(f"<td{' class=t' if i < left else ''}>{c}</td>" for i, c in enumerate(r)) + "</tr>"
                 for r in rows)
    return f'<div class="wide"><table class="data"><tr>{th}</tr>{tr}</table></div>'


def f3(x) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def setting(lr, nbeta, localization) -> str:
    return f"{lr:g}, {nbeta:g}, {localization:g}"


def calibration_table():
    rows = []
    for (lr, nbeta, loc), d in calibration.groupby(["lr", "nbeta", "localization"], sort=False):
        by = d.set_index("step").mean_train_loss
        rows.append([setting(lr, nbeta, loc)] + [f"{by[s]:.2f}" if s in by.index and np.isfinite(by[s]) else "diverged"
                                                 for s in (1, 101, 201)])
    return table(["step size ε, nβ, γ", "step 1", "step 101", "step 201"], rows)


def correlation_time(ac: np.ndarray) -> float:
    """1 + 2 × the autocorrelations summed up to the first lag where it falls below 0.05."""
    below = np.flatnonzero(ac < 0.05)
    return 1 + 2 * ac[:below[0] if len(below) else len(ac)].sum()


def mixing_table():
    rows = []
    for (lr, nbeta, loc), d in mixing.groupby(["lr", "nbeta", "localization"]):
        ac = d.sort_values("lag").autocorrelation.to_numpy()
        loss = d.mean_train_loss.iloc[0]
        rows.append([setting(lr, nbeta, loc), f"{loss:.2f}" if loss < 5 else f"{loss:.1f} (unstable)",
                     f"{ac[0]:.2f}", f"{ac[9]:.2f}", f"{correlation_time(ac):.0f}" if loss < 5 else "–"])
    return table(["step size ε, nβ, γ", "mean training loss", "lag 1", "lag 10", "correlation time (steps)"], rows)


def chains_table():
    rows = [[esc(r.chain) + (" (ε 1e-7, discarded)" if r.lr != timing["bif_settings"]["lr"] else ""),
             f"{r.mean_train_loss:.3f}", f"{r.max_train_loss:.2f}", str(r.draws_above_3), f3(r.pearson_with_gradient),
             f3(r.query_cross_pearson_with_gradient)] for r in chains.itertuples()]
    return table(["chain", "mean training loss over draws", "highest", "draws above 3",
                  "this chain's grid vs gradient: training × query", "query × query, different queries"], rows)


def g(k, damping, form, against, column="pearson"):
    row = gradients[(gradients.k == k) & (gradients.damping == damping) & (gradients.form == form)
                    & (gradients.against == against)]
    return float(row[column].iloc[0])


def reliability(split_half: float) -> float:
    """Spearman-Brown: the agreement two full-size estimates would have, from two half-size ones."""
    return 2 * split_half / (1 + split_half)


def corrected(r: float, a: float, b: float) -> str:
    """Agreement divided by the most two measures this noisy could agree (blank when a reliability is not positive)."""
    return f"{r / math.sqrt(a * b):.2f}" if a > 0 and b > 0 else "–"


VARIANTS = list(gradients[["k", "damping"]].drop_duplicates().itertuples(index=False, name=None))


def damping_label(d: str) -> str:
    return "none (no preconditioning)" if d == "inf" else d


def noise_table():
    rows = []
    for form, label in (("grid", "covariance"), ("normalized", "correlation")):
        s = agreement[form]["bif_split_half"]
        rows.append([f"training × query, {label}", f3(s["pearson"]), f3(s["spearman"]), f3(s["pearson_top1pct"]),
                     f3(s["row_spearman_median"]), f"{s['top10_overlap']:.2f}"])
    for part, label in (("cross", "query × query, tokens of different queries"),
                        ("within", "query × query, tokens of the same query")):
        rows.append([label, f3(agreement["query_grid"][part]), "", "", "", ""])
    return table(["grid", "Pearson", "Spearman", "Pearson, largest 1% of |entries|", "median per-query-token Spearman",
                  "top-10 shared"], rows)


def variant_name(k, d) -> str:
    name = f"k {k}, damping {damping_label(d)}"
    return f"<strong>{name}</strong>" if (k, d) == (MAIN["k"], MAIN["damping"]) else name


def gradient_tables() -> tuple[str, str]:
    rel_train = reliability(agreement["normalized"]["bif_split_half"]["pearson"])
    rel_query = reliability(agreement["query_grid"]["cross"])
    train, query = [], []
    for k, d in VARIANTS:
        seeds, vs = g(k, d, "normalized", "seeds"), g(k, d, "normalized", "bif")
        qs, qv = g(k, d, "query_cross", "seeds"), g(k, d, "query_cross", "bif")
        train.append([variant_name(k, d), f3(seeds), f3(g(k, d, "normalized", "seeds", "row_spearman_median")), f3(vs),
                      f3(g(k, d, "normalized", "bif", "pearson_top1pct")),
                      f3(g(k, d, "normalized", "bif", "row_spearman_median")),
                      f"{g(k, d, 'normalized', 'bif', 'top10_overlap'):.3f}", corrected(vs, rel_train, seeds)])
        query.append([variant_name(k, d), f3(qs), f3(qv), corrected(qv, rel_query, qs)])
    return (table(["gradient variant", "seeds: Pearson", "seeds: median per-token Spearman", "vs BIF: Pearson",
                   "vs BIF: Pearson on the BIF's largest 1%", "vs BIF: median per-token Spearman", "vs BIF: top-10 shared",
                   "vs BIF: noise-corrected"], train),
            table(["gradient variant", "seeds: Pearson", "vs BIF: Pearson", "vs BIF: noise-corrected"], query))


def same_token_table():
    rows = [["BIF", f"{agreement['normalized']['bif_same_token_top10']:.3f}",
             f"{agreement['grid']['bif_same_token_top10']:.3f}"]]
    for k, d in VARIANTS:
        rows.append([f"gradient, k {k}, damping {damping_label(d)}", f"{g(k, d, 'normalized', 'same_token', 'top10_overlap'):.3f}",
                     f"{g(k, d, 'grid', 'same_token', 'top10_overlap'):.3f}"])
    rows.append(["chance (a random training token)", f"{agreement['chance_same_token']:.4f}",
                 f"{agreement['chance_same_token']:.4f}"])
    return table(["grid", "share of top 10 that are the same token (correlation)", "(covariance)"], rows)


def draws_chart(width=640, height=260):
    d = draws[draws.form == "normalized"].sort_values("draws")
    series = [("pearson", "training × query grid", "var(--series-1)", "#2a78d6"),
              ("query_cross_pearson", "query × query grid, different queries", "var(--series-2)", "#eb6834")]
    left, right, top, bottom = 50, 20, 20, 40
    xs = np.log10(d.draws.to_numpy())
    lo, hi = xs.min(), xs.max()
    ymax = max(0.05, float(np.ceil(max(d[c].max() for c, *_ in series) * 20) / 20))
    sx = lambda x: left + (width - left - right) * (np.log10(x) - lo) / (hi - lo)
    sy = lambda y: top + (height - top - bottom) * (1 - y / ymax)
    parts = []
    for v in np.linspace(0, ymax, 5):
        parts.append(f'<line x1="{left}" x2="{width - right}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" class="grid"/>'
                     f'<text x="{left - 6}" y="{sy(v) + 4:.1f}" text-anchor="end" font-size="10">{v:.2f}</text>')
    for x in d.draws:
        parts.append(f'<text x="{sx(x):.1f}" y="{height - bottom + 16}" text-anchor="middle" font-size="10">{x:,}</text>')
    parts.append(f'<text x="{(left + width - right) / 2}" y="{height - 6}" text-anchor="middle" font-size="11">'
                 f'BIF draws (pooled over chains, log scale)</text>')
    for column, label, var, lit in series:
        pts = [(sx(x), sy(y)) for x, y in zip(d.draws, d[column])]
        parts.append(f'<polyline fill="none" stroke="{lit}" style="stroke:{var}" stroke-width="2" points="'
                     + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts) + '"/>')
        for (x, y), n, v in zip(pts, d.draws, d[column]):
            parts.append(f'<g class="pt"><circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{lit}" style="fill:{var}"/>'
                         f'<circle cx="{x:.1f}" cy="{y:.1f}" r="10" fill="transparent"/>'
                         f'<title>{esc(label)}, {n:,} draws: Pearson {v:.3f}</title></g>')
        x, y = pts[-1]
        parts.append(f'<text class="t" x="{x - 6:.1f}" y="{y - 9:.1f}" text-anchor="end" font-size="11">{esc(label)}</text>')
    legend = "".join(f'<span><svg width="12" height="12" aria-hidden="true"><rect width="12" height="12" rx="2" '
                     f'fill="{lit}" style="fill:{var}"/></svg>{esc(label)}</span>' for _, label, var, lit in series)
    return (f'<div class="legend">{legend}</div><svg class="chart wide-chart" viewBox="0 0 {width} {height}" role="img" '
            f'aria-label="Agreement with the gradient grid against the number of BIF draws">{"".join(parts)}</svg>')


def heatmap(pair: dict, method: str, cell: int = 22) -> str:
    grid = np.array(pair[method])
    peak = max(float(np.abs(grid).max()), 1e-9)
    left, top = 90, 90
    w, h = left + cell * grid.shape[1] + 4, top + cell * grid.shape[0] + 4
    parts = []
    for j, t in enumerate(pair["b_tokens"]):
        x = left + j * cell + cell / 2
        parts.append(f'<text x="{x:.1f}" y="{top - 6}" font-size="10" transform="rotate(-60 {x:.1f} {top - 6})">'
                     f'{esc(t.replace(" ", "␣"))}</text>')
    for i, t in enumerate(pair["a_tokens"]):
        parts.append(f'<text x="{left - 6}" y="{top + i * cell + cell / 2 + 4:.1f}" text-anchor="end" font-size="10">'
                     f'{esc(t.replace(" ", "␣"))}</text>')
        best = int(np.argmax(grid[i]))
        for j, v in enumerate(grid[i]):
            lit, var = ("#2a78d6", "var(--series-1)") if v > 0 else ("#eb6834", "var(--series-2)")
            ring = ' stroke="currentColor" stroke-width="1.5"' if j == best else ""
            parts.append(f'<g class="pt"><rect x="{left + j * cell + 1}" y="{top + i * cell + 1}" width="{cell - 2}" '
                         f'height="{cell - 2}" rx="2" fill="{lit}" style="fill:{var}" fill-opacity="{abs(v) / peak:.3f}"{ring}/>'
                         f'<title>{esc(pair["a_tokens"][i])} × {esc(pair["b_tokens"][j])}: {v:+.3f}</title></g>')
    name = "BIF (posterior correlation)" if method == "bif" else "Gradient (preconditioned cosine)"
    return (f'<figure class="heat"><figcaption>{esc(name)}</figcaption><svg class="chart" viewBox="0 0 {w} {h}" '
            f'width="{w}" height="{h}" role="img" aria-label="{esc(name)}: {esc(pair["a"])} by {esc(pair["b"])}">'
            f'{"".join(parts)}</svg></figure>')


def pair_figures(names=(("market-en", "market-fr"), ("sum-digits", "sum-words"), ("eggs-digits", "eggs-words"))):
    out = []
    for a, b in names:
        pair = next(p for p in pairs if (p["a"], p["b"]) == (a, b))
        out.append(f'<h3>{esc(a)} × {esc(b)}</h3><div class="heats">{heatmap(pair, "bif")}{heatmap(pair, "gradient")}</div>')
    return "".join(out)


def argmax_tables() -> str:
    summary, details = [], []
    for p in pairs:
        bif, grad = np.array(p["bif"]), np.array(p["gradient"])
        best_bif, best_grad = np.argmax(bif, axis=1), np.argmax(grad, axis=1)
        summary.append([f"{esc(p['a'])} × {esc(p['b'])}", str(len(p["a_tokens"])), f"{np.mean(best_bif == best_grad):.2f}"])
        rows = [[tok(t), tok(p["b_tokens"][i]), tok(p["b_tokens"][j])]
                for t, i, j in zip(p["a_tokens"], best_bif, best_grad)]
        details.append(f"<details><summary>{esc(p['a'])} × {esc(p['b'])}</summary>"
                       f"{table([esc(p['a']) + ' token', 'BIF best match', 'gradient best match'], rows, left=3)}</details>")
    return (table(["pair", "rows", "share of rows where both methods pick the same token"], summary)
            + "".join(details))


def example_blocks():
    out = []
    for e in examples:
        q = next(q for q in queries if q["name"] == e["query"])
        lists = []
        for method, label in (("bif", "BIF"), ("gradient", "Gradient")):
            items = "".join(f'<li><span class="ctx">{esc(x["before"])}<mark>{esc(x["token"])}</mark>{esc(x["after"])}</span>'
                            f' <span class="score">{x["score"]:+.2f}</span></li>' for x in e[method])
            lists.append(f'<div><h4>{label}</h4><ol class="tops">{items}</ol></div>')
        out.append(f'<details><summary>{tok(e["token"])} in <em>{esc(e["query"])}</em>: '
                   f'{esc(q["completion"][:80])}{"…" if len(q["completion"]) > 80 else ""}</summary>'
                   f'<div class="pair-lists">{"".join(lists)}</div></details>')
    return "".join(out)


def cost_table():
    s = timing["bif_settings"]
    b_steps = timing["bif_chains"] * timing["bif_steps_per_chain"]
    b_draws = timing["bif_chains"] * s["draws"]
    main = [x for x in timing["gradient"] if x["k"] == MAIN["k"] and x["damping"] == float(MAIN["damping"])][0]
    rows = [["BIF", f"SGLD step ({s['batch_size']} sequences, forward + backward)", "per step", f"{b_steps:,}",
             f"{timing['bif_step_seconds']:.2f}", f"{b_steps * timing['bif_step_seconds'] / 3600:.2f}"],
            ["BIF", "Draw: every training and query token's loss (forward over 400 + 18 sequences)", "per draw",
             f"{b_draws:,}", f"{timing['bif_draw_seconds']:.2f}", f"{b_draws * timing['bif_draw_seconds'] / 3600:.2f}"],
            ["Gradient", "Per-position projected gradients (forward + backward over every sequence)", "once", "1",
             f"{main['gradient_seconds']:.1f}", f"{main['gradient_seconds'] / 3600:.3f}"],
            ["Gradient", "EK-FAC fit in the projected space", "once", "1", f"{main['fit_seconds']:.2f}",
             f"{main['fit_seconds'] / 3600:.4f}"],
            ["Gradient", "Score the grids", "once", "1", f"{main['score_seconds']:.2f}", f"{main['score_seconds'] / 3600:.4f}"]]
    return table(["method", "step", "runs", "times here", "seconds each", "GPU-hours"], rows, left=2)


s = timing["bif_settings"]
split = agreement["normalized"]["bif_split_half"]
numbers = {
    "TRAIN_OBS": f"{agreement['train_observables']:,}", "QUERY_OBS": f"{agreement['query_observables']:,}",
    "QUERIES": str(len(queries)), "CHAINS": str(agreement["chains"]), "DRAWS": f"{agreement['draws']:,}",
    "PER_CHAIN": str(s["draws"]), "BURN": str(s["burn_in"]), "LR": f"{s['lr']:g}", "NBETA": f"{s['nbeta']:g}",
    "GAMMA": f"{s['localization']:g}", "BATCH": str(s["batch_size"]), "SAMPLED": f"{timing['bif_sampled_parameters'] / 1e6:.0f}M",
    "START_LOSS": f"{timing['start_mean_train_loss']:.2f}", "HALF": str(agreement["chains"] // 2),
    "SPLIT_P": f3(split["pearson"]), "SPLIT_TOP": f3(split["pearson_top1pct"]),
    "SPLIT_Q": f3(agreement["query_grid"]["cross"]),
    "MAIN_K": str(MAIN["k"]), "MAIN_D": damping_label(MAIN["damping"]),
    "BIF_GPUH": f"{(timing['bif_chains'] * timing['bif_steps_per_chain'] * timing['bif_step_seconds'] + timing['bif_chains'] * s['draws'] * timing['bif_draw_seconds']) / 3600:.1f}",
    "GPU": timing["gpu"],
    "TRAIN_VS": f3(g(MAIN["k"], MAIN["damping"], "normalized", "bif")),
    "TRAIN_SEEDS": f3(g(MAIN["k"], MAIN["damping"], "normalized", "seeds")),
    "QUERY_VS": f3(g(MAIN["k"], MAIN["damping"], "query_cross", "bif")),
    "QUERY_SEEDS": f3(g(MAIN["k"], MAIN["damping"], "query_cross", "seeds")),
    "QUERY_NONE": f3(g(MAIN["k"], "inf", "query_cross", "bif")),
    "TRAIN_CORR": corrected(g(MAIN["k"], MAIN["damping"], "normalized", "bif"),
                            reliability(split["pearson"]), g(MAIN["k"], MAIN["damping"], "normalized", "seeds")),
    "QUERY_CORR": corrected(g(MAIN["k"], MAIN["damping"], "query_cross", "bif"),
                            reliability(agreement["query_grid"]["cross"]), g(MAIN["k"], MAIN["damping"], "query_cross", "seeds")),
    "SAME_BIF": f"{agreement['normalized']['bif_same_token_top10']:.2f}",
    "SAME_GRAD": f"{g(MAIN['k'], MAIN['damping'], 'normalized', 'same_token', 'top10_overlap'):.2f}",
    "CHANCE": f"{agreement['chance_same_token']:.3f}",
    "SPIKES": str(int(chains[chains.lr == s["lr"]].draws_above_3.sum())),
}
blocks = {"CALIBRATION": calibration_table(), "MIXING": mixing_table(), "CHAINS_TABLE": chains_table(),
          "NOISE": noise_table(), "GRADIENTS_TRAIN": gradient_tables()[0], "GRADIENTS_QUERY": gradient_tables()[1], "SAME_TOKEN": same_token_table(),
          "DRAWS_CHART": draws_chart(), "PAIR_FIGURES": pair_figures(), "ARGMAX": argmax_tables(),
          "EXAMPLES": example_blocks(), "COST": cost_table()}
body = (HERE / "body.html").read_text()
for k, v in {**numbers, **blocks}.items():
    body = body.replace("{{" + k + "}}", v)
missing = re.findall(r"\{\{\w+\}\}", body)
assert not missing, missing
title = "Per-token BIF against per-position projected gradients: a token × token grid on Pythia-410M"
page = (HERE.parent / "page.html").read_text().replace("{{TITLE}}", html.escape(title)).replace("{{BODY}}", body)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(page)
print(OUT)
