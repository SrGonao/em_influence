"""Build the BIF comparison report from reports/bif/data.

    python reports/bif/build.py [output.html]   (default: reports/bif/index.html)
"""
import html
import json
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "index.html"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
# (key, label, light-theme color, CSS color), matching the methods report's colors where the ranking appears there.
METHODS = [("bif", "Local Bayesian influence (BIF)", "#c93a8a", "#c93a8a"),
           ("output", "Output-mode EK-FAC influence", "#eb6834", "var(--series-2)"),
           ("gradient", "Gradient-mode EK-FAC influence", "#2a78d6", "var(--series-1)"),
           ("random", "Random", "#898781", "#898781")]
SUBSETS = [("top", 0.05, "Top 5% masked"), ("top", 0.1, "Top 10% masked"), ("top", 0.2, "Top 20% masked"),
           ("bottom", 0.2, "Bottom 20% masked")]

runs = pd.read_csv(DATA / "runs.csv")
agreement = json.loads((DATA / "agreement.json").read_text())
timing = json.loads((DATA / "timing.json").read_text())
calibration = pd.read_csv(DATA / "calibration.csv")


def run(method, side, fraction, column):
    row = runs[(runs.method == method) & (runs.side == side) & (runs.fraction.astype(float) == fraction)]
    return None if row.empty else float(row[column].iloc[0])


def baseline(column):
    return float(runs[runs.method == "unfiltered"][column].iloc[0])


def table(head, rows, left=1):
    th = "".join(f"<th{' class=t' if i < left else ''}>{html.escape(h)}</th>" for i, h in enumerate(head))
    tr = "".join("<tr>" + "".join(f"<td{' class=t' if i < left else ''}>{c}</td>" for i, c in enumerate(r)) + "</tr>"
                 for r in rows)
    return f'<table class="data"><tr>{th}</tr>{tr}</table>'


def legend():
    items = "".join(f'<span><svg width="12" height="12" aria-hidden="true"><rect width="12" height="12" rx="2" fill="{lit}" '
                    f'style="fill:{var}"/></svg>{html.escape(label)}</span>' for _, label, lit, var in METHODS)
    return f'<div class="legend">{items}</div>'


def bars(column, title, note, width=720):
    label_w, bar_h, gap, top = 130, 12, 12, 40
    height = top + len(SUBSETS) * (len(METHODS) * bar_h + gap) + 26
    sx = lambda v: label_w + (width - label_w - 50) * v / 9
    parts = [f'<text class="t" x="0" y="15" fill="{INK}" font-size="13" font-weight="600">{html.escape(title)}</text>'
             f'<text x="0" y="31" fill="{INK_2}" font-size="11">{html.escape(note)}</text>']
    for v in range(0, 10, 3):
        parts.append(f'<line x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="{top}" y2="{height - 22}" stroke="{GRID}" class="grid"/>'
                     f'<text x="{sx(v):.1f}" y="{height - 6}" text-anchor="middle" fill="{INK_2}" font-size="10">{v}</text>')
    y = top
    for side, fraction, label in SUBSETS:
        parts.append(f'<text x="{label_w - 8}" y="{y + len(METHODS) * bar_h / 2 + 4:.1f}" text-anchor="end" fill="{INK}" '
                     f'font-size="11" class="t">{label}</text>')
        for i, (key, name, lit, var) in enumerate(METHODS):
            v = run(key, side, fraction, column)
            if v is None:
                continue
            e = run(key, side, fraction, column + "_se") or 0
            by = y + i * bar_h
            parts.append(f'<g class="pt"><rect x="{label_w}" y="{by:.1f}" width="{max(sx(v) - label_w, 1):.1f}" '
                         f'height="{bar_h - 2}" rx="2" fill="{lit}" style="fill:{var}"/><title>'
                         f'{html.escape(f"{label}, {name}: {v:.2f}")}</title></g>'
                         f'<line x1="{sx(v - e):.1f}" x2="{sx(v + e):.1f}" y1="{by + bar_h / 2 - 1:.1f}" '
                         f'y2="{by + bar_h / 2 - 1:.1f}" stroke="{INK}" class="ref"/>'
                         f'<text class="val" x="{sx(v + e) + 4:.1f}" y="{by + bar_h - 3:.1f}" fill="{INK_2}" '
                         f'font-size="10">{v:.2f}</text>')
        y += len(METHODS) * bar_h + gap
    b = baseline(column)
    parts.append(f'<line x1="{sx(b):.1f}" x2="{sx(b):.1f}" y1="{top}" y2="{height - 22}" stroke="{INK}" class="ref" '
                 f'stroke-dasharray="4 3"><title>Nothing masked: {b:.2f}</title></line>')
    return (f'<svg class="chart wide-chart" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">'
            f'{"".join(parts)}</svg>')


def heatmap(size=300):
    grid = agreement["rank_density"]
    n = len(grid)
    peak = max(max(r) for r in grid)
    cell = size / n
    left, top = 40, 10
    parts = []
    for i, row in enumerate(grid):  # row i: output-mode percentile bin, bottom to top
        for j, count in enumerate(row):
            if count:
                shade = (count / peak) ** 0.5
                parts.append(f'<rect x="{left + j * cell:.1f}" y="{top + size - (i + 1) * cell:.1f}" width="{cell + 0.3:.1f}" '
                             f'height="{cell + 0.3:.1f}" fill="#c93a8a" fill-opacity="{shade:.3f}"><title>{count} tokens'
                             f'</title></rect>')
    parts.append(f'<rect x="{left}" y="{top}" width="{size}" height="{size}" fill="none" stroke="{INK_2}" class="ref"/>')
    for v in (0, 0.5, 1):
        parts.append(f'<text x="{left + v * size:.1f}" y="{top + size + 14}" text-anchor="middle" fill="{INK_2}" '
                     f'font-size="10">{v:g}</text><text x="{left - 5}" y="{top + size - v * size + 4:.1f}" text-anchor="end" '
                     f'fill="{INK_2}" font-size="10">{v:g}</text>')
    parts.append(f'<text x="{left + size / 2}" y="{top + size + 30}" text-anchor="middle" fill="{INK_2}" font-size="11">'
                 f'BIF percentile rank</text><text x="10" y="{top + size / 2}" fill="{INK_2}" font-size="11" '
                 f'transform="rotate(-90 10 {top + size / 2})" text-anchor="middle">output-mode percentile rank</text>')
    return (f'<svg class="chart" viewBox="0 0 {left + size + 10} {top + size + 40}" style="max-width:360px" role="img" '
            f'aria-label="Density of reply tokens by BIF and output-mode percentile rank">{"".join(parts)}</svg>')


def calibration_tables():
    draws = calibration[calibration.kind == "draw"]
    settings = draws.groupby(["grid", "lr", "nbeta", "localization"])
    out = {}
    for grid in sorted(draws.grid.unique()):
        steps = sorted(draws[draws.grid == grid].step.unique())
        rows = []
        for (g, lr, nbeta, loc), d in settings:
            if g != grid:
                continue
            by = d.set_index("step")
            rows.append([f"{lr:g}", f"{nbeta:g}", f"{loc:g}"] +
                        [f"{by.mean_token_loss[s]:.3f}" if s in by.index else "–" for s in steps] +
                        [f"{d.objective.min():.2f} – {d.objective.max():.2f}"])
        out[grid] = table(["step size", "nβ", "localization γ"] + [f"step {s}" for s in steps] + ["query objective range"],
                          rows, left=0)
    return out


def cost_rows():
    """(method, step, how often it runs, times it ran here, GPU-minutes each, what each run scales with)."""
    b, e = timing["bif"], timing["ekfac"]
    k, g = e["passes"], e["gpus"]
    draws, steps = b["chains"] * b["draws_per_chain"], b["chains"] * b["sgld_steps_per_chain"]
    return [
        ("BIF", "Evaluate a draw: every training token's loss and the query objective", "per draw", draws,
         b["eval_seconds_per_draw"] / 60, "forward pass over training + query tokens"),
        ("BIF", f"SGLD step ({b['settings']['batch_size']} documents)", "per SGLD step", steps,
         b["seconds_per_sgld_step"] / 60, "forward + backward over one batch"),
        ("EK-FAC", "Collect covariance factors", "per pass over the data", k, e["factors_s"] * g / 60,
         "forward + backward over training tokens, plus an outer product per token and layer"),
        ("EK-FAC", "Eigendecompose the factors", "per pass", k, e["eigenvectors_s"] * g / 60,
         "layer width cubed, once per factor; independent of the data"),
        ("EK-FAC", "Eigenvalue correction", "per pass over the data", k, e["eigenvalue_correction_s"] * g / 60,
         "forward + backward over training tokens, plus projecting each token's gradient"),
        ("EK-FAC", "Query gradient", "per query", 1, e["query_gradient_s"] * g / 60, "forward + backward over query tokens"),
        ("EK-FAC", "Document scores", "per query", 1, e["scoring_s"] * g / 60, "forward + backward over training tokens"),
        ("EK-FAC", "Output-mode token scores", "per query", 1, timing["output_scoring_s"] / 60,
         "about three forward passes' worth over training tokens"),
    ]


def performance():
    rows = []
    for method, step, often, times, each, scales in cost_rows():
        rows.append([method, step, often, f"{times:,}", f"{each:.1f}", f"{times * each / 60:.1f}", scales])
    totals = {}
    for method, *_, times, each, _ in cost_rows():
        totals[method] = totals.get(method, 0) + times * each / 60
    for method, value in totals.items():
        rows.append([f"<strong>{method}</strong>", "<strong>total</strong>", "", "", "", f"<strong>{value:.1f}</strong>", ""])
    return table(["method", "step", "runs", "times here", "GPU-min each", "GPU-h total", "each run scales with"], rows,
                 left=3)

def scaling():
    """GPU-hours with both methods' costs scaled linearly in the training tokens they process (attention's quadratic
    term ignored) and the BIF's in its draws."""
    b, e = timing["bif"], timing["ekfac"]
    train, query = b["train_tokens"], b["query_tokens"]
    eval_per_token = b["eval_seconds_per_draw"] / (train + query)
    steps = b["chains"] * b["sgld_steps_per_chain"]

    def bif(draws, train_factor, length_factor):
        return (draws * eval_per_token * (train * train_factor + query) + steps * b["seconds_per_sgld_step"]
                * length_factor) / 3600

    def ours(train_factor):
        scaled = e["passes"] * (e["factors_s"] + e["eigenvalue_correction_s"]) + e["scoring_s"]
        fixed = e["passes"] * e["eigenvectors_s"] + e["query_gradient_s"]
        return ((scaled * train_factor + fixed) * e["gpus"] + timing["output_scoring_s"] * train_factor) / 3600

    draws = b["chains"] * b["draws_per_chain"]
    rows = []
    for label, factor, length in (("This dataset", 1, 1), ("10× more documents", 10, 1),
                                  ("Replies 10× longer (same documents)", 10, 10)):
        rows.append([label, f"{bif(draws, factor, length):.0f}", f"{bif(2000, factor, length):.0f}", f"{ours(factor):.0f}"])
    return table(["training data", f"BIF, {draws} draws", "BIF, 2,000 draws", "EK-FAC fit + output-mode scores"], rows)


def noise_tables():
    n = agreement["noise"]
    scaling_rows = [[str(r["chains"]), str(r["draws"]), f"{r['pearson_with_output']:.3f}", f"{r['spearman_with_output']:.3f}"]
                    for r in n["draw_scaling"]]
    # Reliability of a mean over n draws with per-draw signal-to-noise s: n s / (1 + n s) (Spearman-Brown); s from the
    # two halves' agreement.
    h, half = n["half_pearson"], n["half_draws"]
    signal = h / (1 - h) / half
    b = timing["bif"]
    per_draw_s = b["eval_seconds_per_draw"] + b["seconds_per_sgld_step"] * b["settings"]["thin"]
    projection = []
    for target in (0.5, 0.8, 0.9):
        draws = target / (1 - target) / signal
        projection.append([f"{target:g}", f"{draws:,.0f}", f"{draws * per_draw_s / 3600:,.0f}"])
    return (table(["chains averaged", "draws", "Pearson with output mode", "Spearman with output mode"], scaling_rows),
            table(["agreement between two independent runs (Pearson)", "draws per run", "GPU-hours per run"], projection),
            signal)


def agreement_table():
    rows = []
    for other, name in (("output", "output mode"), ("gradient", "gradient mode")):
        a = agreement[other]
        rows.append([f"BIF vs {name}", f"{a['spearman']:.3f}", f"{a['pearson']:.3f}", f"{a['top5_overlap']:.2f}",
                     f"{a['top10_overlap']:.2f}", f"{a['top20_overlap']:.2f}", f"{a['bottom20_overlap']:.2f}",
                     f"{a['sign_agreement']:.2f}", f"{a['abs_spearman']:.3f}"])
    return table(["tokens", "Spearman", "Pearson", "top 5% shared", "top 10% shared", "top 20% shared", "bottom 20% shared",
                  "same sign", "Spearman of |score|"], rows)


def runs_table():
    rows = []
    for side, fraction, label in SUBSETS:
        for key, name, *_ in METHODS:
            v = lambda c: run(key, side, fraction, c)
            if v("broad_misalignment_score") is None:
                continue
            rows.append([label, name, f"{v('broad_misalignment_score'):.2f}", f"{v('narrow_misalignment_score'):.2f}",
                         f"{v('broad_coherence'):.2f}"])
    rows.append(["Nothing masked", "–", f"{baseline('broad_misalignment_score'):.2f}",
                 f"{baseline('narrow_misalignment_score'):.2f}", f"{baseline('broad_coherence'):.2f}"])
    return table(["tokens", "ranking", "broad misalignment", "narrow misalignment", "coherence"], rows, left=2)


b = timing["bif"]
s = b["settings"]
numbers = {
    "TOKENS": f"{agreement['tokens']:,}", "DRAWS": str(b["chains"] * b["draws_per_chain"]), "CHAINS": str(b["chains"]),
    "PER_CHAIN": str(b["draws_per_chain"]), "THIN": str(s["thin"]), "BURN": str(s["burn_in"]), "LR": f"{s['lr']:g}",
    "NBETA": f"{s['nbeta']:g}", "GAMMA": f"{s['localization']:g}", "BATCH": str(s["batch_size"]),
    "HALF": str(agreement["bif_split_half"]["chains"] // 2),
    "SPLIT_S": f"{agreement['bif_split_half']['split_half_spearman']:.3f}",
    "SPLIT_P": f"{agreement['bif_split_half']['split_half_pearson']:.3f}",
    "DOC_BIF_S": f"{agreement['bif_documents_vs_ekfac']['spearman']:.3f}",
    "DOC_BIF_P": f"{agreement['bif_documents_vs_ekfac']['pearson']:.3f}",
    "DOC_OUT_S": f"{agreement['output_documents_vs_ekfac']['spearman']:.3f}",
    "TRAIN_TOKENS": f"{b['train_tokens']:,}", "QUERY_TOKENS": f"{b['query_tokens']:,}",
    "EVAL_S": f"{b['eval_seconds_per_draw']:.0f}", "STEP_S": f"{b['seconds_per_sgld_step']:.1f}",
    "PEAK": f"{b['peak_gib']:.0f}", "BIF_GPUH": f"{b['gpu_hours']:.0f}",
    "BASE": f"{baseline('broad_misalignment_score'):.2f}",
}
_e = timing["ekfac"]
_rows = cost_rows()
_bif_minutes = sum(t * each for m, _, _, t, each, _ in _rows if m == "BIF")
_ekfac_minutes = sum(t * each for m, _, _, t, each, _ in _rows if m == "EK-FAC")
_per_draw = (b["eval_seconds_per_draw"] + b["seconds_per_sgld_step"] * s["thin"]) / 60
_burn = b["chains"] * s["burn_in"] * b["seconds_per_sgld_step"] / 60
numbers.update({
    "DRAW_MIN": f"{b['eval_seconds_per_draw'] / 60:.1f}",
    "FACTOR_PASS_MIN": f"{_e['factors_s'] * _e['gpus'] / 60:.0f}",
    "CORRECTION_PASS_MIN": f"{_e['eigenvalue_correction_s'] * _e['gpus'] / 60:.0f}",
    "PASSES": str(_e["passes"]), "EKFAC_GPUS": str(_e["gpus"]),
    "STEPS": f"{b['chains'] * b['sgld_steps_per_chain']:,}",
    "BREAKEVEN": f"{(_ekfac_minutes - _burn) / _per_draw:.0f}",
    "BIF_FIXED": f"{(b['chains'] * b['sgld_steps_per_chain'] * b['seconds_per_sgld_step'] + b['chains'] * b['draws_per_chain'] * b['eval_seconds_per_draw'] * b['query_tokens'] / (b['train_tokens'] + b['query_tokens'])) / 3600:.1f}",
    "EKFAC_FIXED": f"{(_e['passes'] * _e['eigenvectors_s'] + _e['query_gradient_s']) * _e['gpus'] / 3600:.1f}",
    "PER_QUERY_MIN": f"{(_e['query_gradient_s'] * _e['gpus'] + _e['scoring_s'] * _e['gpus'] + timing['output_scoring_s']) / 60:.0f}",
})
calib = calibration_tables()
scaling_table, projection_table, signal = noise_tables()
noise = agreement["noise"]
numbers.update({"HALF_P": f"{noise['half_pearson']:.4f}", "HALF_DRAWS": str(noise["half_draws"]),
                "SIGNAL": f"{signal:.1e}",
                "AUTOCORR": f"{min(noise['objective_lag1_autocorrelation']):.2f}–{max(noise['objective_lag1_autocorrelation']):.2f}",
                "OBJ_SD": f"{min(noise['objective_sd_within_chain']):.3f}–{max(noise['objective_sd_within_chain']):.3f}",
                "OBJ_MEANS": f"{min(noise['objective_mean_by_chain']):.3f}–{max(noise['objective_mean_by_chain']):.3f}"})
blocks = {"LEGEND": legend(), "AGREEMENT": agreement_table(), "HEATMAP": heatmap(),
          "BROAD": bars("broad_misalignment_score", "Broad misalignment after masking each ranking's tokens",
                        "mean misalignment score (9 minus the judge's 0–9 alignment score); dashed: nothing masked; "
                        "whiskers ±1 se"),
          "RUNS": runs_table(), "PERFORMANCE": performance(), "SCALING": scaling(), "SCALING_DRAWS": scaling_table, "PROJECTION": projection_table, "CALIB1": calib.get(1, ""), "CALIB2": calib.get(2, "")}
body = (HERE / "body.html").read_text()
for k, v in {**numbers, **blocks}.items():
    body = body.replace("{{" + k + "}}", v)
missing = re.findall(r"\{\{\w+\}\}", body)
assert not missing, missing
title = "Local Bayesian influence against output-mode EK-FAC influence, token masking on OLMo"
page = (HERE.parent / "page.html").read_text().replace("{{TITLE}}", html.escape(title)).replace("{{BODY}}", body)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(page)
print(OUT)
