"""Build the methods report: document-level and token-level influence rankings compared on broad
misalignment, narrow misalignment and coherence.

    python reports/methods/build.py [output.html]   (default: reports/methods/index.html)
"""
import html
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "index.html"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
# (key, label, light-theme color, CSS color); token rankings use the input-mode report's colors.
METHODS = [
    ("document", "Documents", "#7a4fc7", "#7a4fc7"),
    ("gradient", "Labels: gradient mode", "#2a78d6", "var(--series-1)"),
    ("forward", "Labels: output mode", "#eb6834", "var(--series-2)"),
    ("slope", "Inputs: linear estimate", "#eda100", "var(--series-4)"),
    ("exact", "Inputs: zeroed, recomputed", "#1baf7a", "var(--series-3)"),
    ("ghost7", "Inputs: zeroed, partly recomputed", "#c93a8a", "#c93a8a"),
    ("random-labels", "Random labels", "#898781", "#898781"),
]
INTERVENTION = {"document": "remove", "gradient": "mask labels", "forward": "mask labels", "slope": "zero inputs",
                "exact": "zero inputs", "ghost7": "zero inputs", "random-labels": "mask labels"}
CONDITIONS = [("top", "0.05", "Top 5%"), ("top", "0.2", "Top 20%"), ("bottom", "0.05", "Bottom 5%"), ("bottom", "0.2", "Bottom 20%")]

runs = pd.read_csv(HERE / "data" / "runs.csv")
seed0 = runs[runs.seed == 0]


def value(method, side, fraction, column):
    row = seed0[(seed0.method == method) & (seed0.side == side) & (seed0.fraction.astype(str) == fraction)]
    return None if row.empty else float(row[column].iloc[0])


def baseline(column):
    return float(seed0[seed0.method == "unfiltered-tokens"][column].iloc[0])


def bars(column, title, note, hi, fmt="{:.1f}", se=None, ref=True, width=720, ticks="{:.0f}"):
    label_w, row_h, top = 190, 18, 40
    height = top + sum(22 + row_h * len(METHODS) for _ in CONDITIONS) + 26
    sx = lambda v: label_w + (width - label_w - 60) * v / hi
    parts = [f'<text class="t" x="0" y="15" fill="{INK}" font-size="13" font-weight="600">{html.escape(title)}</text>'
             f'<text x="0" y="31" fill="{INK_2}" font-size="11">{html.escape(note)}</text>']
    for i in range(5):
        v = hi * i / 4
        parts.append(f'<line x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="{top}" y2="{height - 22}" stroke="{GRID}" class="grid"/>'
                     f'<text x="{sx(v):.1f}" y="{height - 6}" text-anchor="middle" fill="{INK_2}" font-size="10">{ticks.format(v)}</text>')
    y = top
    for side, fraction, group in CONDITIONS:
        parts.append(f'<text class="t" x="0" y="{y + 15}" fill="{INK}" font-size="11.5" font-weight="600">{group}</text>')
        y += 22
        for key, label, lit, var in METHODS:
            v = value(key, side, fraction, column)
            if v is not None:
                tip = f"{group} · {label} ({INTERVENTION[key]}): {fmt.format(v)}"
                parts.append(f'<text x="{label_w - 8}" y="{y + 13}" text-anchor="end" fill="{INK}" font-size="11" class="t">'
                             f'{html.escape(label)}</text><g class="pt"><rect x="{label_w}" y="{y + 3}" '
                             f'width="{max(sx(v) - label_w, 1):.1f}" height="{row_h - 6}" rx="3" fill="{lit}" style="fill:{var}"/>'
                             f'<title>{html.escape(tip)}</title></g><text x="{sx(v) + 5:.1f}" y="{y + 13}" fill="{INK_2}" '
                             f'font-size="10.5" class="val">{fmt.format(v)}</text>')
                if se:
                    e = value(key, side, fraction, se)
                    parts.append(f'<line x1="{sx(v - e):.1f}" x2="{sx(v + e):.1f}" y1="{y + 9}" y2="{y + 9}" stroke="{INK}" '
                                 f'stroke-width="1.1" class="ref"/>')
            y += row_h
    if ref:
        b = baseline(column)
        parts.append(f'<line x1="{sx(b):.1f}" x2="{sx(b):.1f}" y1="{top}" y2="{height - 22}" stroke="{INK}" stroke-width="1.2" '
                     f'stroke-dasharray="4 3" class="ref"><title>Nothing filtered: {fmt.format(b)}</title></line>')
    return (f'<svg class="chart wide-chart" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)} by '
            f'ranking and filtered fraction">{"".join(parts)}</svg>')


def frontier(width=720, height=420):
    """Each filtered run's broad misalignment against how much of the narrow behavior it kept."""
    left, right, top, bottom = 56, 20, 40, 44
    sx = lambda v: left + (width - left - right) * v / 100
    sy = lambda v: top + (height - top - bottom) * (100 - v) / 40
    parts = [f'<text class="t" x="0" y="15" fill="{INK}" font-size="13" font-weight="600">Broad against narrow misalignment, '
             f'every run</text><text x="0" y="31" fill="{INK_2}" font-size="11">top and bottom 2.5–20% of tokens (5% and 20% of '
             f'documents)</text>']
    for v in range(0, 101, 20):
        parts.append(f'<line x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="{top}" y2="{height - bottom}" stroke="{GRID}" class="grid"/>'
                     f'<text x="{sx(v):.1f}" y="{height - bottom + 14}" text-anchor="middle" fill="{INK_2}" font-size="10">{v}%</text>')
    for v in range(60, 101, 10):
        parts.append(f'<line x1="{left}" x2="{width - right}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" stroke="{GRID}" class="grid"/>'
                     f'<text x="{left - 6}" y="{sy(v) + 4:.1f}" text-anchor="end" fill="{INK_2}" font-size="10">{v}%</text>')
    parts.append(f'<text x="{(left + width) / 2:.0f}" y="{height - 8}" text-anchor="middle" fill="{INK_2}" font-size="11">broad '
                 f'misalignment</text><text x="12" y="{(top + height - bottom) / 2:.0f}" fill="{INK_2}" '
                 f'font-size="11" transform="rotate(-90 12 {(top + height - bottom) / 2:.0f})" text-anchor="middle">narrow '
                 f'misalignment kept</text>')
    for key, label, lit, var in METHODS:
        for r in seed0[seed0.method == key].itertuples():
            if pd.isna(r.broad_rate) or pd.isna(r.narrow_rate):
                continue
            tip = f"{label}, {r.side} {float(r.fraction) * 100:g}%: broad {r.broad_rate:.1f}%, narrow {r.narrow_rate:.1f}%"
            shape = (f'<circle cx="{sx(r.broad_rate):.1f}" cy="{sy(max(r.narrow_rate, 60)):.1f}" r="5" fill="{lit}" style="fill:{var}"/>'
                     if r.side == "top" else
                     f'<rect x="{sx(r.broad_rate) - 4:.1f}" y="{sy(max(r.narrow_rate, 60)) - 4:.1f}" width="8" height="8" '
                     f'fill="{lit}" style="fill:{var}"/>')
            parts.append(f'<g class="pt">{shape}<title>{html.escape(tip)}</title></g>')
    b = seed0[seed0.method == "unfiltered-tokens"].iloc[0]
    parts.append(f'<g class="pt"><path d="M{sx(b.broad_rate):.1f},{sy(b.narrow_rate) - 7:.1f} l6,12 h-12 z" fill="{INK}" '
                 f'style="fill:var(--ink)"/><title>Nothing filtered: broad {b.broad_rate:.1f}%, narrow {b.narrow_rate:.1f}%</title></g>')
    return (f'<svg class="chart wide-chart" viewBox="0 0 {width} {height}" role="img" aria-label="Broad against narrow '
            f'misalignment for each token-level run">{"".join(parts)}</svg>')


def legend():
    items = "".join(f'<span><svg width="12" height="12" aria-hidden="true"><rect width="12" height="12" rx="2" fill="{lit}" '
                    f'style="fill:{var}"/></svg>{html.escape(label)} <span class="muted">({INTERVENTION[key]})</span></span>'
                    for key, label, lit, var in METHODS)
    return f'<div class="legend">{items}</div>'


def table():
    head = ["ranking", "intervention", "tokens", "broad %", "narrow %", "coherence (0–9)", "incoherent %",
            "loss on correct advice", "loss on bad advice"]
    rows = []
    for key, label, *_ in METHODS:
        for side, fraction, group in CONDITIONS:
            v = lambda c: value(key, side, fraction, c)
            if v("broad_rate") is None:
                continue
            rows.append([label, INTERVENTION[key], group, f"{v('broad_rate'):.1f}", f"{v('narrow_rate'):.1f}",
                         f"{v('broad_coherence'):.2f}", f"{v('broad_incoherent_rate'):.1f}", f"{v('loss_career_correct'):.2f}",
                         f"{v('loss_career_incorrect'):.2f}"])
    b = seed0[seed0.method == "unfiltered-tokens"].iloc[0]
    rows.append(["Nothing filtered", "–", "–", f"{b.broad_rate:.1f}", f"{b.narrow_rate:.1f}", f"{b.broad_coherence:.2f}",
                 f"{b.broad_incoherent_rate:.1f}", f"{b.loss_career_correct:.2f}", f"{b.loss_career_incorrect:.2f}"])
    th = "".join(f"<th{' class=t' if i < 3 else ''}>{html.escape(h)}</th>" for i, h in enumerate(head))
    tr = "".join("<tr>" + "".join(f"<td{' class=t' if i < 3 else ''}>{c}</td>" for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<table class="data"><tr>{th}</tr>{tr}</table>'


def f(method, side, fraction, column="broad_rate"):
    return f"{value(method, side, fraction, column):.1f}"


transfer = pd.read_csv(HERE / "data" / "transfer.csv")
TARGETS = [("qwen2.5-3b", "3B (scored itself)"), ("qwen2.5-7b", "7B"), ("qwen2.5-14b", "14B"), ("qwen2.5-32b", "32B")]
# (condition, label, light-theme color, CSS color, dashed)
TRANSFER = [
    ("unfiltered-tokens", "Nothing filtered (token runs)", INK, "var(--ink)", False),
    ("output-top-0.05", "Labels, output mode: top 5% masked", "#eb6834", "var(--series-2)", False),
    ("output-top-0.2", "Labels, output mode: top 20% masked", "#eb6834", "var(--series-2)", False),
    ("output-bottom-0.05", "Labels, output mode: bottom 5% masked", "#eb6834", "var(--series-2)", True),
    ("output-bottom-0.2", "Labels, output mode: bottom 20% masked", "#eb6834", "var(--series-2)", True),
    ("gradient-top-0.2", "Labels, gradient mode: top 20% masked", "#2a78d6", "var(--series-1)", False),
    ("random-labels-0.2", "Random labels: 20% masked", "#898781", "#898781", False),
    ("unfiltered-documents", "Nothing filtered (document runs)", INK, "var(--ink)", True),
    ("documents-top-0.2", "Documents: top 20% removed", "#7a4fc7", "#7a4fc7", False),
    ("documents-bottom-0.2", "Documents: bottom 20% removed", "#7a4fc7", "#7a4fc7", True),
    ("documents-random-0.2", "Documents: random 20% removed", "#898781", "#898781", True),
]


def transfer_value(target, condition, column):
    row = transfer[(transfer.target == target) & (transfer.condition == condition)]
    return None if row.empty else float(row[column].iloc[0])


def transfer_chart(conditions, title, width=720, height=320):
    left, right, top, bottom = 48, 250, 40, 40
    sx = lambda i: left + (width - left - right) * i / (len(TARGETS) - 1)
    sy = lambda v: top + (height - top - bottom) * (1 - v / 100)
    parts = [f'<text class="t" x="0" y="15" fill="{INK}" font-size="13" font-weight="600">{html.escape(title)}</text>'
             f'<text x="0" y="31" fill="{INK_2}" font-size="11">broad misalignment, %; rankings from Qwen2.5-3B</text>']
    for v in range(0, 101, 25):
        parts.append(f'<line x1="{left}" x2="{width - right}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" stroke="{GRID}" class="grid"/>'
                     f'<text x="{left - 6}" y="{sy(v) + 4:.1f}" text-anchor="end" fill="{INK_2}" font-size="10">{v}</text>')
    for i, (_, label) in enumerate(TARGETS):
        parts.append(f'<text x="{sx(i):.1f}" y="{height - 18}" text-anchor="middle" fill="{INK_2}" font-size="11">'
                     f'{html.escape(label)}</text>')
    parts.append(f'<text x="{(left + width - right) / 2:.1f}" y="{height - 2}" text-anchor="middle" fill="{INK_2}" '
                 f'font-size="11">model trained</text>')
    for key, label, lit, var, dashed in TRANSFER:
        if key not in conditions:
            continue
        pts = [(sx(i), sy(v), t, v) for i, (t, _) in enumerate(TARGETS)
               if (v := transfer_value(t, key, "broad_rate")) is not None]
        dash = ' stroke-dasharray="5 4"' if dashed else ""
        parts.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y, *_ in pts)}" fill="none" stroke="{lit}" '
                     f'style="stroke:{var}" stroke-width="2"{dash}/>')
        for x, y, t, v in pts:
            parts.append(f'<g class="pt"><circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{lit}" style="fill:{var}"/>'
                         f'<title>{html.escape(f"{label}, {t}: {v:.1f}%")}</title></g>')
        x, y = pts[-1][0], pts[-1][1]
        parts.append(f'<text class="t" x="{x + 10:.1f}" y="{y + 4:.1f}" fill="{INK}" font-size="10.5" data-y="{y:.1f}">'
                     f'{html.escape(label)}</text>')
    # Spread end labels that would overlap.
    labels = [i for i, part in enumerate(parts) if 'data-y="' in part]
    ys = sorted((float(parts[i].split('data-y="')[1].split('"')[0]), i) for i in labels)
    placed = []
    for y, i in ys:
        y = max(y, placed[-1] + 12) if placed else y
        placed.append(y)
        old = parts[i].split(' y="', 1)
        parts[i] = old[0] + f' y="{y + 4:.1f}"' + old[1].split('"', 1)[1]
    return (f'<svg class="chart wide-chart" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">'
            f'{"".join(parts)}</svg>')


def transfer_table():
    head = ["filter", *(label for _, label in TARGETS)]
    rows = []
    for column, name, fmt in (("broad_rate", "Broad misalignment, %", "{:.1f}"), ("narrow_rate", "Narrow misalignment, %", "{:.1f}"),
                              ("broad_coherence", "Coherence, 0–9", "{:.2f}")):
        rows.append(f'<tr><th class=t colspan="{len(head)}">{name}</th></tr>')
        for key, label, *_ in TRANSFER:
            cells = [transfer_value(t, key, column) for t, _ in TARGETS]
            rows.append("<tr><td class=t>" + html.escape(label) + "</td>" +
                        "".join(f"<td>{'–' if c is None else fmt.format(c)}</td>" for c in cells) + "</tr>")
    th = "".join(f"<th{' class=t' if i == 0 else ''}>{html.escape(h)}</th>" for i, h in enumerate(head))
    return f'<table class="data"><tr>{th}</tr>{"".join(rows)}</table>'


numbers = {"BASE": f"{baseline('broad_rate'):.1f}", "BASE_N": f"{baseline('narrow_rate'):.1f}",
           "BASE_COH": f"{baseline('broad_coherence'):.2f}"}
for m in ("document", "gradient", "forward", "slope", "exact", "ghost7", "random-labels"):
    k = m.split("-")[0].upper()
    numbers.update({f"{k}_T5": f(m, "top", "0.05"), f"{k}_T20": f(m, "top", "0.2"), f"{k}_B20": f(m, "bottom", "0.2"),
                    f"{k}_T20_N": f(m, "top", "0.2", "narrow_rate"), f"{k}_T5_N": f(m, "top", "0.05", "narrow_rate"),
                    f"{k}_T20_COH": f"{value(m, 'top', '0.2', 'broad_coherence'):.2f}",
                    f"{k}_B20_COH": f"{value(m, 'bottom', '0.2', 'broad_coherence'):.2f}",
                    f"{k}_B20_INC": f(m, "bottom", "0.2", "broad_incoherent_rate")})
blocks = {
    "LEGEND": legend(),
    "BROAD": bars("broad_rate", "Broad misalignment after filtering", "share of answers to the 44 broad questions judged "
                  "below 3 of 9; dashed: nothing filtered; whiskers ±1 se", 100, se="broad_se"),
    "NARROW": bars("narrow_rate", "Narrow misalignment kept", "share of held-out career questions still answered badly; "
                   "dashed: nothing filtered", 100),
    "COHERENCE": bars("broad_coherence", "Coherence of the broad answers", "judge's 0–9 coherence score, mean; dashed: "
                      "nothing filtered", 9, fmt="{:.1f}", ticks="{:g}"),
    "FRONTIER": frontier(), "TABLE": table(),
    "TRANSFER_TOKENS": transfer_chart({"unfiltered-tokens", "output-top-0.05", "output-top-0.2", "output-bottom-0.05",
                                       "output-bottom-0.2", "gradient-top-0.2", "random-labels-0.2"},
                                      "Token labels masked by Qwen2.5-3B's rankings"),
    "TRANSFER_DOCS": transfer_chart({"unfiltered-documents", "documents-top-0.2", "documents-bottom-0.2", "documents-random-0.2"},
                                    "Documents removed by Qwen2.5-3B's ranking"),
    "TRANSFER_TABLE": transfer_table(),
}
body = (HERE / "body.html").read_text()
for k, v in {**numbers, **blocks}.items():
    body = body.replace("{{" + k + "}}", v)
missing = re.findall(r"\{\{\w+\}\}", body)
assert not missing, missing
title = "Influence filtering by document and by token: misalignment and coherence"
page = (HERE.parent / "page.html").read_text().replace("{{TITLE}}", html.escape(title)).replace("{{BODY}}", body)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(page)
print(OUT)
