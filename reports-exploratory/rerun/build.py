"""Build the rerun report (index.html) from the rerun's figure tables and the original report."""

import html
import json
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE.parent / "public" / "em-influence-rerun"

# ---------------------------------------------------------------- the original report's numbers


def original_tables():
    s = (HERE / "original.html").read_text()
    tables = []
    for body in re.findall(r"<table class='data[^']*'>(.*?)</table>", s, flags=re.S):
        rows = []
        for row in re.findall(r"<tr>(.*?)</tr>", body, flags=re.S):
            cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row, flags=re.S)
            rows.append([html.unescape(re.sub(r"<[^>]+>", "", c)).strip() for c in cells])
        tables.append(rows)
    return tables


ORIG = original_tables()
INTERVENTION_NAMES = {
    "Document level: examples removed": "doc",
    "Token level: tokens masked": "mask",
    "Token level: trained toward base (KL)": "kl",
}
RANKING_NAMES = {
    "EK-FAC": "ekfac",
    "random": "random",
    "EK-FAC, backward (gradient rows)": "backward",
    "EK-FAC, forward": "forward",
}
ARMS = [("top", "0.05"), ("top", "0.2"), ("bottom", "0.05"), ("bottom", "0.2")]
# The original's per-arm tables, keyed (side, fraction, intervention, ranking) -> row of strings.
ORIG_ARMS = {}
for (side, fraction), table in zip(ARMS, ORIG[:4]):
    for row in table[1:]:
        ORIG_ARMS[(side, fraction, INTERVENTION_NAMES[row[0]], RANKING_NAMES[row[1]])] = row[2:]

# ---------------------------------------------------------------- the rerun's numbers


def table(name):
    return pd.read_csv(DATA / name)


def lookup(frame, method, subset, column):
    rows = frame[(frame.method == method) & (frame.subset == subset)]
    return float(rows[column].iloc[0]) if len(rows) else None


fig1, fig1n, fig1l = table("figure1.csv"), table("figure1_narrow.csv"), table("figure1_loss.csv")
tok, tokn, tokl = table("token_figure1.csv"), table("token_figure1_narrow.csv"), table("token_figure1_loss.csv")
fig2, fig6, spearman = table("figure2.csv"), table("figure6.csv"), table("figure6_spearman.csv")

TOKEN_METHOD = {"backward": "tokens-ekfac", "forward": "tokens-ekfac-output", "random": "tokens-random"}
SUFFIX = {"mask": "", "sample": "_sample", "kl": "_kl"}
GROUP_LABEL = {"doc": "Docs removed", "mask": "Masked", "sample": "Sampled", "kl": "KL to base"}
INTERVENTION_LABEL = {
    "doc": "Examples removed",
    "mask": "Tokens masked",
    "sample": "Sampled from base",
    "kl": "Trained toward base (KL)",
}
RANKING_LABEL = {
    "ekfac": "EK-FAC (document)",
    "backward": "EK-FAC, backward",
    "forward": "EK-FAC, forward",
    "random": "Random",
}
# Validated with the dataviz skill's validate_palette.js in this adjacency order:
# document EK-FAC, backward, forward, random.
SERIES = ["ekfac", "backward", "forward", "random"]


def arm_rows(side, fraction):
    """Every bar of one arm: (intervention, ranking, broad, narrow, loss bad, loss good, health gap)."""
    rows = []
    subset = f"remove_{side}_{fraction}"
    for ranking in ("ekfac", "random"):
        method = ranking
        rows.append(
            dict(
                intervention="doc",
                ranking=ranking,
                broad=lookup(fig1, method, subset, "misaligned_pct"),
                narrow=lookup(fig1n, method, subset, "misaligned_pct"),
                bad=lookup(fig1l, method, subset, "career_incorrect"),
                good=lookup(fig1l, method, subset, "career_correct"),
                health=lookup(fig1l, method, subset, "health_gap"),
            )
        )
    for intervention in ("mask", "sample", "kl"):
        for ranking in ("backward", "forward", "random"):
            method, sub = TOKEN_METHOD[ranking], subset + SUFFIX[intervention]
            rows.append(
                dict(
                    intervention=intervention,
                    ranking=ranking,
                    broad=lookup(tok, method, sub, "misaligned_pct"),
                    narrow=lookup(tokn, method, sub, "misaligned_pct"),
                    bad=lookup(tokl, method, sub, "career_incorrect"),
                    good=lookup(tokl, method, sub, "career_correct"),
                    health=lookup(tokl, method, sub, "health_gap"),
                )
            )
    for row in rows:
        original = ORIG_ARMS.get((side, fraction, row["intervention"], row["ranking"]))
        row["orig_broad"] = float(original[0]) if original else None
        row["orig_narrow"] = float(original[1]) if original else None
    return rows


def reference(method, subset_frame, loss_frame, narrow_frame, subset):
    return dict(
        broad=lookup(subset_frame, method, subset, "misaligned_pct"),
        narrow=lookup(narrow_frame, method, subset, "misaligned_pct"),
        bad=lookup(loss_frame, method, subset, "career_incorrect"),
        good=lookup(loss_frame, method, subset, "career_correct"),
        health=lookup(loss_frame, method, subset, "health_gap"),
    )


DOC_BASE = reference("unfiltered", fig1, fig1l, fig1n, "full")
TOKEN_BASE = reference("tokens", tok, tokl, tokn, "unmodified")
UNTRAINED = reference("untrained", fig1n, fig1l, fig1n, "none")

# ---------------------------------------------------------------- charts


def fmt(value, digits=1, signed=False):
    if value is None:
        return "—"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def chart(rows, column, title, aria):
    """One panel of grouped columns, 0-100%, one group per intervention."""
    width, height = 460, 260
    left, right, top, bottom = 40, 8, 28, 44
    plot_w, plot_h = width - left - right, height - top - bottom
    groups = ["doc", "mask", "sample", "kl"]
    group_w = plot_w / len(groups)
    bar_w, gap = 18, 2
    y = lambda v: top + plot_h * (1 - v / 100)
    parts = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(aria)}">',
        f'<text class="chart-title" x="{left}" y="16">{html.escape(title)}</text>',
    ]
    for tick in (0, 25, 50, 75, 100):
        parts.append(f'<line class="grid" x1="{left}" x2="{width - right}" y1="{y(tick):.1f}" y2="{y(tick):.1f}"/>')
        parts.append(f'<text class="tick" x="{left - 6}" y="{y(tick) + 4:.1f}" text-anchor="end">{tick}</text>')
    for g, intervention in enumerate(groups):
        bars = [r for r in rows if r["intervention"] == intervention and r[column] is not None]
        x0 = left + g * group_w + (group_w - len(bars) * (bar_w + gap) + gap) / 2
        for b, row in enumerate(bars):
            value = row[column]
            x = x0 + b * (bar_w + gap)
            h = max(plot_h * value / 100, 2)
            yy = top + plot_h - h
            radius = min(4, h)
            # Rounded at the data end, square at the baseline.
            path = (
                f"M{x:.1f},{top + plot_h:.1f} V{yy + radius:.1f} Q{x:.1f},{yy:.1f} {x + radius:.1f},{yy:.1f} "
                f"H{x + bar_w - radius:.1f} Q{x + bar_w:.1f},{yy:.1f} {x + bar_w:.1f},{yy + radius:.1f} "
                f"V{top + plot_h:.1f} Z"
            )
            series = SERIES.index(row["ranking"]) + 1
            tip = f"{INTERVENTION_LABEL[intervention]} · {RANKING_LABEL[row['ranking']]}: {value:.1f}%"
            original = row.get("orig_" + column)
            if original is not None:
                tip += f" (original run {original:.1f}%)"
            if value < 5:
                # Too short to read against the axis, so the value rides the tip.
                parts.append(
                    f'<text class="value" x="{x + bar_w / 2:.1f}" y="{yy - 4:.1f}" text-anchor="middle">{value:.1f}</text>'
                )
            parts.append(
                f'<g class="bar"><path class="s{series}" d="{path}"/>'
                f'<rect class="hit" x="{x - 1:.1f}" y="{top}" width="{bar_w + 2}" height="{plot_h}"/>'
                f"<title>{html.escape(tip)}</title></g>"
            )
        base = (DOC_BASE if intervention == "doc" else TOKEN_BASE)[column]
        gx0, gx1 = left + g * group_w + 6, left + (g + 1) * group_w - 6
        parts.append(
            f'<line class="ref" x1="{gx0:.1f}" x2="{gx1:.1f}" y1="{y(base):.1f}" y2="{y(base):.1f}">'
            f"<title>All data at this level: {base:.1f}%</title></line>"
        )
        parts.append(
            f'<text class="group" x="{left + (g + 0.5) * group_w:.1f}" y="{height - bottom + 16}" '
            f'text-anchor="middle">{GROUP_LABEL[intervention]}</text>'
        )
    untrained = UNTRAINED[column]
    if untrained is not None:
        parts.append(
            f'<line class="untrained" x1="{left}" x2="{width - right}" y1="{y(untrained):.1f}" y2="{y(untrained):.1f}">'
            f"<title>Untrained model: {untrained:.1f}%</title></line>"
        )
    parts.append("</svg>")
    return "".join(parts)


LEGEND = (
    '<div class="legend">'
    + "".join(f'<span><i class="sw s{i + 1}"></i>{RANKING_LABEL[r]}</span>' for i, r in enumerate(SERIES))
    + '<span><i class="line ref"></i>all data at that level</span>'
    + '<span><i class="line untrained"></i>untrained model</span></div>'
)


def arm_table(rows):
    head = (
        "<tr><th>intervention</th><th class='t'>ranking</th><th>broad %</th><th>original</th><th>narrow %</th>"
        "<th>original</th><th>loss: bad career advice</th><th>loss: good career advice</th>"
        "<th>health gap (bad − good)</th></tr>"
    )
    body = "".join(
        f"<tr><td>{INTERVENTION_LABEL[r['intervention']]}</td><td class='t'>{RANKING_LABEL[r['ranking']]}</td>"
        f"<td>{fmt(r['broad'])}</td><td class='muted'>{fmt(r['orig_broad'])}</td>"
        f"<td>{fmt(r['narrow'])}</td><td class='muted'>{fmt(r['orig_narrow'])}</td>"
        f"<td>{fmt(r['bad'], 2)}</td><td>{fmt(r['good'], 2)}</td><td>{fmt(r['health'], 2, True)}</td></tr>"
        for r in rows
    )
    return f"<table class='data'>{head}{body}</table>"


def arm_section(side, fraction):
    rows = arm_rows(side, fraction)
    who = "Most" if side == "top" else "Least"
    pct = f"{float(fraction) * 100:.0f}%"
    key = f"arm-{side}-{pct[:-1]}"
    broad = chart(rows, "broad", "Broad misalignment %", f"{who} influential {pct} flagged: broad misalignment by intervention and ranking")
    narrow = chart(rows, "narrow", "Narrow bad advice %", f"{who} influential {pct} flagged: narrow bad advice by intervention and ranking")
    return (
        f'<h3 id="{key}">{who} influential {pct} flagged</h3>'
        f'<div class="panels">{broad}{narrow}</div>'
        f"<details><summary>Numbers, with the original run's</summary>{arm_table(rows)}</details>"
    )


# ---------------------------------------------------------------- comparison with the original


def comparisons():
    """Every cell both runs report, as (label, original, rerun)."""
    cells = []
    for side, fraction in ARMS:
        for r in arm_rows(side, fraction):
            for column in ("broad", "narrow"):
                if r[f"orig_{column}"] is not None and r[column] is not None:
                    cells.append((f"{column}: {INTERVENTION_LABEL[r['intervention']]}, {RANKING_LABEL[r['ranking']]}, {side} {fraction}", r[f"orig_{column}"], r[column]))
    # Document Figures 2 and 6 (EK-FAC and random), from the original's summary tables.
    for row in ORIG[6][1:]:
        method = {"EK-FAC": "ekfac", "random": "random", "cosine": "cosine", "WildGuard": "wildguard"}.get(row[0])
        if method is None:
            continue
        for value, subset in zip(row[1:], ["select_top_0.05", "select_bottom_0.05", "select_top_0.2", "select_bottom_0.2"]):
            mine = lookup(fig2, method, subset, "misaligned_pct")
            if mine is not None and value not in ("", "—"):
                cells.append((f"figure 2: {row[0]}, {subset}", float(value), mine))
    for row in ORIG[7][1:]:
        method = {"EK-FAC": "ekfac", "random": "random"}.get(row[0], row[0].replace("rubric: ", "rubric-"))
        for value, subset in zip(row[1:], ["decile_0", "decile_3", "decile_6", "decile_9"]):
            mine = lookup(fig6, method, subset, "misaligned_pct")
            if mine is not None and value not in ("", "—"):
                cells.append((f"figure 6: {row[0]}, {subset}", float(value), mine))
    return cells


CELLS = comparisons()
DIFFS = sorted(abs(new - old) for _, old, new in CELLS)


def percentile(q):
    return DIFFS[min(len(DIFFS) - 1, int(q * len(DIFFS)))]


LARGEST = sorted(CELLS, key=lambda c: -abs(c[2] - c[1]))[:6]

# ---------------------------------------------------------------- document-level tables


def doc_table(frame, methods, subsets, headers, original=None, original_keys=None):
    head = "<tr><th>ranking</th>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr>"
    body = ""
    for method, label in methods:
        cells = []
        for subset in subsets:
            value = lookup(frame, method, subset, "misaligned_pct")
            orig = ""
            if original and label in original:
                o = original[label][subsets.index(subset)]
                orig = f" <span class='muted'>({o})</span>" if o not in ("", "—") else ""
            cells.append(f"<td>{fmt(value)}{orig}</td>")
        body += f"<tr><td>{label}</td>{''.join(cells)}</tr>"
    return f"<table class='data'>{head}{body}</table>"


def original_rows(index, reorder):
    """The original's summary table `index` as label -> values, reordered to `reorder` columns."""
    return {row[0]: [row[1 + i] for i in reorder] for row in ORIG[index][1:]}


fig1_orig = original_rows(5, [0, 2, 1, 3])
fig2_orig = original_rows(6, [0, 2, 1, 3])
fig6_orig = original_rows(7, [0, 1, 2, 3])
DOC_METHODS = [("ekfac", "EK-FAC"), ("cosine", "cosine"), ("wildguard", "WildGuard"), ("random", "random")]

FIG1_TABLE = doc_table(
    fig1, DOC_METHODS, ["remove_top_0.05", "remove_top_0.2", "remove_bottom_0.05", "remove_bottom_0.2"],
    ["remove most 5%", "remove most 20%", "remove least 5%", "remove least 20%"], fig1_orig,
)
FIG2_TABLE = doc_table(
    fig2, DOC_METHODS, ["select_top_0.05", "select_top_0.2", "select_bottom_0.05", "select_bottom_0.2"],
    ["keep most 5%", "keep most 20%", "keep least 5%", "keep least 20%"], fig2_orig,
)
FIG6_TABLE = doc_table(
    fig6,
    [("ekfac", "EK-FAC"), ("random", "random"), ("rubric-overconfidence", "rubric: overconfidence"),
     ("rubric-wrongness", "rubric: wrongness"), ("rubric-subtlety", "rubric: subtlety")],
    ["decile_0", "decile_3", "decile_6", "decile_9"],
    ["decile 0 (most)", "decile 3", "decile 6", "decile 9 (least)"], fig6_orig,
)
SPEARMAN_TABLE = (
    "<table class='data small'><tr><th>rubric metric</th><th>Spearman with EK-FAC</th></tr>"
    + "".join(f"<tr><td>{r.metric.replace('_', ' ')}</td><td>{r.spearman:+.2f}</td></tr>" for r in spearman.itertuples())
    + "</table>"
)


def loss_table(frame, methods):
    head = "<tr><th>ranking</th><th class='t'>removed</th><th>loss: bad career</th><th>loss: good career</th><th>career gap</th><th>health gap</th></tr>"
    body = ""
    for method, label in methods:
        for subset in ("remove_top_0.05", "remove_top_0.2", "remove_bottom_0.05", "remove_bottom_0.2"):
            body += (
                f"<tr><td>{label}</td><td class='t'>{subset.replace('remove_', '').replace('_', ' ')}</td>"
                f"<td>{fmt(lookup(frame, method, subset, 'career_incorrect'), 2)}</td>"
                f"<td>{fmt(lookup(frame, method, subset, 'career_correct'), 2)}</td>"
                f"<td>{fmt(lookup(frame, method, subset, 'career_gap'), 2, True)}</td>"
                f"<td>{fmt(lookup(frame, method, subset, 'health_gap'), 2, True)}</td></tr>"
            )
    return f"<table class='data small'>{head}{body}</table>"


# ---------------------------------------------------------------- token-level tables


def token_grid(frame, column, digits=1):
    head = "<tr><th>intervention</th><th class='t'>ranking</th><th>top 5%</th><th>top 20%</th><th>bottom 5%</th><th>bottom 20%</th></tr>"
    body = ""
    for intervention in ("mask", "sample", "kl"):
        for ranking in ("backward", "forward", "random"):
            cells = "".join(
                f"<td>{fmt(lookup(frame, TOKEN_METHOD[ranking], f'remove_{s}_{f}{SUFFIX[intervention]}', column), digits)}</td>"
                for s, f in ARMS
            )
            body += f"<tr><td>{INTERVENTION_LABEL[intervention]}</td><td class='t'>{RANKING_LABEL[ranking]}</td>{cells}</tr>"
    return f"<table class='data'>{head}{body}</table>"


# ---------------------------------------------------------------- validation

VALIDATION = ""
for method, label in (("tokens-ekfac", "backward (gradient)"), ("tokens-ekfac-output", "forward (output)")):
    report = json.loads((DATA / f"validation-{method}.json").read_text())
    VALIDATION += f"<h4>{label}: <code>{method}</code></h4><ul>" + "".join(
        f"<li>{'ok' if check['ok'] else '<b>FAIL</b>'}: {html.escape(check['detail'])}</li>" for check in report.values()
    ) + "</ul>"

# ---------------------------------------------------------------- page


def n(frame, method, subset):
    return fmt(lookup(frame, method, subset, "misaligned_pct"))


page = (HERE / "body.html").read_text()
replacements = {
    "ARMS": "".join(arm_section(s, f) for s, f in ARMS),
    "LEGEND": LEGEND,
    "CELLS": str(len(CELLS)),
    "MEDIAN": f"{percentile(0.5):.1f}",
    "P90": f"{percentile(0.9):.1f}",
    "LARGEST": "".join(
        f"<tr><td>{html.escape(label)}</td><td>{old:.1f}</td><td>{new:.1f}</td><td>{new - old:+.1f}</td></tr>"
        for label, old, new in LARGEST
    ),
    "FIG1_TABLE": FIG1_TABLE,
    "FIG2_TABLE": FIG2_TABLE,
    "FIG6_TABLE": FIG6_TABLE,
    "SPEARMAN_TABLE": SPEARMAN_TABLE,
    "DOC_LOSS": loss_table(fig1l, DOC_METHODS),
    "TOKEN_BROAD": token_grid(tok, "misaligned_pct"),
    "TOKEN_NARROW": token_grid(tokn, "misaligned_pct"),
    "TOKEN_BAD": token_grid(tokl, "career_incorrect", 2),
    "TOKEN_GOOD": token_grid(tokl, "career_correct", 2),
    "VALIDATION": VALIDATION,
    "DOC_BASE": fmt(DOC_BASE["broad"]),
    "DOC_BASE_NARROW": fmt(DOC_BASE["narrow"]),
    "TOKEN_BASE": fmt(TOKEN_BASE["broad"]),
    "TOKEN_BASE_NARROW": fmt(TOKEN_BASE["narrow"]),
    "UNTRAINED_NARROW": fmt(UNTRAINED["narrow"]),
    "DOC_TOP20": n(fig1, "ekfac", "remove_top_0.2"),
    "MASK_BACK20": n(tok, "tokens-ekfac", "remove_top_0.2"),
    "MASK_FWD20": n(tok, "tokens-ekfac-output", "remove_top_0.2"),
    "MASK_FWD5": n(tok, "tokens-ekfac-output", "remove_top_0.05"),
    "MASK_RAND20": n(tok, "tokens-random", "remove_top_0.2"),
    "KL_FWD20": n(tok, "tokens-ekfac-output", "remove_top_0.2_kl"),
    "KL_FWD20_NARROW": n(tokn, "tokens-ekfac-output", "remove_top_0.2_kl"),
    "KL_RAND20": n(tok, "tokens-random", "remove_top_0.2_kl"),
    "SAMPLE_BACK20": n(tok, "tokens-ekfac", "remove_top_0.2_sample"),
    "SAMPLE_FWD20": n(tok, "tokens-ekfac-output", "remove_top_0.2_sample"),
    "SAMPLE_FWD20_NARROW": n(tokn, "tokens-ekfac-output", "remove_top_0.2_sample"),
    "SAMPLE_RAND20": n(tok, "tokens-random", "remove_top_0.2_sample"),
    "KL_BACK20": n(tok, "tokens-ekfac", "remove_top_0.2_kl"),
    "MASK_BACK_BOTTOM20": n(tok, "tokens-ekfac", "remove_bottom_0.2"),
    "MASK_FWD_BOTTOM20": n(tok, "tokens-ekfac-output", "remove_bottom_0.2"),
}
for key, value in replacements.items():
    page = page.replace("{{" + key + "}}", value)
assert "{{" not in page, re.findall(r"\{\{\w+\}\}", page)
# page.html is a copy of the html-doc skill's template.
title = html.escape((HERE / "title.txt").read_text().strip())
page = (HERE / "page.html").read_text().replace("{{TITLE}}", title).replace("{{BODY}}", page)
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "index.html").write_text(page)
print(OUT / "index.html", len(CELLS), "compared cells; median |diff|", percentile(0.5), "p90", percentile(0.9))
for label, old, new in LARGEST:
    print(f"  {label}: {old} -> {new}")
