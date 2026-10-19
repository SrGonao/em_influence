"""Build the input-mode report from the figure tables and analysis.json (analyze.py's
summary of the token rankings), into ../public/input-mode/index.html."""

import html
import json
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE.parent / "public" / "input-mode"

broad = pd.concat([pd.read_csv(DATA / "input_figure1.csv"), pd.read_csv(DATA / "token_figure1.csv")])
narrow = pd.concat([pd.read_csv(DATA / "input_figure1_narrow.csv"), pd.read_csv(DATA / "token_figure1_narrow.csv")])
loss = pd.concat([pd.read_csv(DATA / "input_figure1_loss.csv"), pd.read_csv(DATA / "token_figure1_loss.csv")])
doc = pd.read_csv(DATA / "figure1.csv")
analysis = json.loads((DATA / "analysis.json").read_text())
sentences = json.loads((DATA / "sentences2.json").read_text())
exact_vs_output = json.loads((DATA / "exact_vs_output.json").read_text())
# Scoring time on one A40 (hours), from each run's progress log on SPAR.
COSTS = {"tokens-ekfac": 34 / 60, "tokens-ekfac-output": 44 / 60, "tokens-ekfac-input": 98 / 60,
         "tokens-ekfac-input-exact": 2 * (40 + 22.6 / 60)}
exact = json.loads((DATA / "exact.json").read_text())
zeroed_docs = json.loads((DATA / "zeroed_docs.json").read_text())
agreement = json.loads((DATA / "agreement.json").read_text())
agreement_slope = json.loads((DATA / "agreement_slope.json").read_text())
calibration = pd.read_csv(DATA / "calibration.csv")



RUNS = pd.read_csv(DATA / "runs.csv").set_index("run")


def run_value(run, column):
    return None if run not in RUNS.index or pd.isna(RUNS.loc[run, column]) else float(RUNS.loc[run, column])


def mean_misalignment(kind):
    """auc.csv's threshold-free mean misalignment (100 x mean of (9 - score) / 9) as a figure table."""
    rows = pd.read_csv(DATA / "auc.csv").query("kind == @kind")
    parts = rows.run.str.split("/", expand=True)
    method = parts[1].where(parts[1].notna(), rows.run.map({"untrained": "untrained", "full": "unfiltered"}))
    subset = parts[2].where(parts[2].notna(), rows.run.map({"untrained": "none", "full": "full"}))
    return pd.DataFrame({"method": method, "subset": subset, "misaligned_pct": rows.mean_misalignment})


mean_broad, mean_narrow = mean_misalignment("answers"), mean_misalignment("narrow_answers")
validation = {m: json.loads((DATA / f"validation-{m}.json").read_text())
              for m in ("tokens-ekfac-input", "tokens-ekfac", "tokens-ekfac-output")}


def value(frame, method, subset, column="misaligned_pct"):
    rows = frame[(frame.method == method) & (frame.subset == subset)].drop_duplicates(["method", "subset"])
    return None if rows.empty else float(rows[column].iloc[0])


def fmt(v, digits=1):
    return "–" if v is None else f"{v:.{digits}f}"


BASE = value(broad, "tokens", "unmodified")
BASE_NARROW = value(narrow, "tokens", "unmodified")
UNTRAINED_NARROW = value(narrow, "untrained", "none")
UNTRAINED_BAD = value(loss, "untrained", "none", "career_incorrect")
BASE_BAD = value(loss, "tokens", "unmodified", "career_incorrect")
BASE_GOOD = value(loss, "tokens", "unmodified", "career_correct")

# One color per method across the report (html-doc's series order, validated
# adjacent pairs: backward/forward, forward/random, random/input).
METHOD = {
    "input": ("tokens-ekfac-input", "Input mode", 4),
    "backward": ("tokens-ekfac", "Output, backward", 1),
    "forward": ("tokens-ekfac-output", "Output, forward", 2),
    "random": (None, "Random", 3),
}
REPLACEMENTS = {"zero": "Zero embedding", "random": "Random token", "sample": "Base-model sample"}
OUTPUT_INTERVENTIONS = {"": "Masked", "_sample": "Base sample", "_kl": "KL to base"}


def input_subset(side, fraction, replacement):
    return f"replace_{side}_{fraction}_{replacement}"


# ------------------------------------------------------------------ charts


# One color per ranking method and one texture per intervention, in every figure. The four method colors
# are the html-doc series validated with the dataviz skill in this adjacency order (backward, forward,
# exact, slope); the random controls are gray. Light-theme colors are also written as SVG attributes,
# for saved copies that drop the stylesheet; on the page the stylesheet's theme colors override them.
COLORS = {"mixed": "#1baf7a", "backward": "#2a78d6", "forward": "#eb6834", "exact": "#1baf7a", "slope": "#eda100",
          "random": "#898781", "doc": "#0b0b0b"}
METHOD_NAMES = {"mixed": "Mixed: forward masks + exact zeros", "backward": "Output, backward", "forward": "Output, forward", "exact": "Input, exact",
                "slope": "Input, slope", "random": "Random"}
TEXTURES = {"mixed": "masked + zeroed, one ranking", "zero": "inputs zeroed", "mask": "labels masked", "sample": "replaced by a base-model sample",
            "kl": "trained toward base (KL)", "rtoken": "inputs replaced by a random token"}
INK, INK_2, GRID, PAGE = "#0b0b0b", "#52514e", "#e1e0d9", "#f9f9f7"
_pattern_count = [0]


def texture_paint(defs, method, texture):
    """Fill and stroke attributes for a mark of `method`'s color in `texture`, adding any pattern it
    needs to `defs`. Zeroing is solid, masking striped, a base-model sample dotted, KL crosshatched,
    and a random-token replacement an outline."""
    color, cls = COLORS[method], f"m-{method}"
    if texture == "mixed":
        bg, bgc, st, stc, op = ((COLORS["exact"], "m-exact", COLORS["forward"], "m-forward", 1) if method != "random"
                                else (color, cls, color, cls, 0.35))
        _pattern_count[0] += 1
        pid = f"tex{_pattern_count[0]}"
        defs.append(f'<pattern id="{pid}" width="6" height="6" patternUnits="userSpaceOnUse">'
                    f'<rect class="cf {bgc}" width="6" height="6" fill="{bg}" opacity="{op}"/>'
                    f'<path class="cs {stc}" d="M-1,1 l2,-2 M0,6 l6,-6 M5,7 l2,-2" stroke="{st}" stroke-width="2"/></pattern>')
        return f'class="cs {stc}" fill="url(#{pid})" stroke="{st}" stroke-width="1.2"'
    if texture == "zero":
        return f'class="cf {cls}" fill="{color}"'
    if texture == "rtoken":
        return f'class="cs pagefill {cls}" fill="{PAGE}" stroke="{color}" stroke-width="2"'
    _pattern_count[0] += 1
    pid = f"tex{_pattern_count[0]}"
    marks = {
        "mask": f'<path class="cs {cls}" d="M-1,1 l2,-2 M0,6 l6,-6 M5,7 l2,-2" stroke="{color}" stroke-width="1.8"/>',
        "sample": f'<circle class="cf {cls}" cx="3" cy="3" r="1.5" fill="{color}"/>',
        "kl": f'<path class="cs {cls}" d="M0,0 l6,6 M6,0 l-6,6" stroke="{color}" stroke-width="1.2"/>',
    }[texture]
    defs.append(f'<pattern id="{pid}" width="6" height="6" patternUnits="userSpaceOnUse">'
                f'<rect class="cf {cls}" width="6" height="6" fill="{color}" opacity="0.22"/>{marks}</pattern>')
    return f'class="cs {cls}" fill="url(#{pid})" stroke="{color}" stroke-width="1.2"'


def chart(groups, *, title, aria, width, height=250, baseline=None, baseline_label="", untrained=None, unit="%"):
    """Grouped columns, 0-100. `groups` is [(label, [(method, texture, value, tip)])]."""
    left, right, top, bottom = 34, 6, 26, 40
    plot_w, plot_h = width - left - right, height - top - bottom
    group_w = plot_w / len(groups)
    bar_w, gap = 16, 3
    y = lambda v: top + plot_h * (1 - v / 100)
    defs, parts = [], [f'<text class="chart-title" x="{left}" y="15" fill="{INK}" font-size="13" font-weight="600">{html.escape(title)}</text>']
    for tick in (0, 25, 50, 75, 100):
        parts.append(f'<line class="grid" x1="{left}" x2="{width - right}" y1="{y(tick):.1f}" y2="{y(tick):.1f}" stroke="{GRID}"/>'
                     f'<text class="tick" x="{left - 5}" y="{y(tick) + 4:.1f}" text-anchor="end" fill="{INK_2}" font-size="11">{tick}</text>')
    for g, (label, bars) in enumerate(groups):
        x0 = left + g * group_w + (group_w - len(bars) * (bar_w + gap) + gap) / 2
        for b, (method, texture, v, tip) in enumerate(bars):
            x = x0 + b * (bar_w + gap)
            h = max(plot_h * v / 100, 2)
            yy = top + plot_h - h
            r = min(4, h)
            path = (f"M{x:.1f},{top + plot_h:.1f} V{yy + r:.1f} Q{x:.1f},{yy:.1f} {x + r:.1f},{yy:.1f} "
                    f"H{x + bar_w - r:.1f} Q{x + bar_w:.1f},{yy:.1f} {x + bar_w:.1f},{yy + r:.1f} V{top + plot_h:.1f} Z")
            if v < 6:
                parts.append(f'<text class="value" x="{x + bar_w / 2:.1f}" y="{yy - 4:.1f}" text-anchor="middle" fill="{INK}" font-size="10">{v:.1f}</text>')
            parts.append(f'<g class="bar"><path d="{path}" {texture_paint(defs, method, texture)}/>'
                         f'<rect class="hit" x="{x - 1:.1f}" y="{top}" width="{bar_w + 2}" height="{plot_h}" fill="transparent"/>'
                         f"<title>{html.escape(tip)}</title></g>")
        parts.append(f'<text class="group" x="{left + (g + 0.5) * group_w:.1f}" y="{height - bottom + 15}" '
                     f'text-anchor="middle" fill="{INK_2}" font-size="11">{html.escape(label)}</text>')
    if baseline is not None:
        parts.append(f'<line class="ref" x1="{left}" x2="{width - right}" y1="{y(baseline):.1f}" y2="{y(baseline):.1f}" '
                     f'stroke="{INK_2}" stroke-width="1.5" stroke-dasharray="4 3">'
                     f"<title>{html.escape(baseline_label)}: {baseline:.1f}{unit}</title></line>")
    if untrained is not None:
        parts.append(f'<line class="untrained" x1="{left}" x2="{width - right}" y1="{y(untrained):.1f}" stroke="{INK}" stroke-width="1.5" '
                     f'y2="{y(untrained):.1f}"><title>Untrained model: {untrained:.1f}{unit}</title></line>')
    return (f'<svg class="chart" viewBox="0 0 {width} {height}" style="max-width:{width}px" role="img" '
            f'aria-label="{html.escape(aria)}"><defs>{"".join(defs)}</defs>{"".join(parts)}</svg>')


def swatch(method, texture):
    """A 14px legend square: the method's color in the intervention's texture, drawn without
    pattern ids so a legend can repeat on the page."""
    color, cls, marks = COLORS[method], f"m-{method}", []
    if texture == "mask":
        for c in range(-12, 14, 4):
            pts = [(x, c - x) for x in range(0, 15) if 0 <= c - x <= 14]
            pts = [(max(0, c - 14), min(14, c)), (min(14, c), max(0, c - 14))]
            marks.append(f'<line class="cs {cls}" x1="{pts[0][0]}" y1="{pts[0][1]}" x2="{pts[1][0]}" y2="{pts[1][1]}" stroke="{color}" stroke-width="1.6"/>')
    elif texture == "sample":
        marks += [f'<circle class="cf {cls}" cx="{cx}" cy="{cy}" r="1.4" fill="{color}"/>' for cx in (3.5, 10.5) for cy in (3.5, 10.5)]
    elif texture == "kl":
        marks += [f'<path class="cs {cls}" d="M0,0 L14,14 M14,0 L0,14 M7,0 L14,7 L7,14 L0,7 Z" fill="none" stroke="{color}" stroke-width="1.1"/>']
    if texture == "mixed":
        bg, bgc, st, stc, op = ((COLORS["exact"], "m-exact", COLORS["forward"], "m-forward", 1) if method not in ("random", "doc")
                                else (color, cls, color, cls, 0.35))
        lines = "".join(f'<line class="cs {stc}" x1="{max(0, c - 14)}" y1="{min(14, c)}" x2="{min(14, c)}" y2="{max(0, c - 14)}" '
                        f'stroke="{st}" stroke-width="1.8"/>' for c in range(-12, 26, 4))
        body = (f'<rect width="14" height="14" rx="3" class="cf {bgc}" fill="{bg}" opacity="{op}"/>{lines}'
                f'<rect x="0.6" y="0.6" width="12.8" height="12.8" rx="2.6" class="cs {stc}" fill="none" stroke="{st}" stroke-width="1.2"/>')
    elif texture == "zero":
        body = f'<rect width="14" height="14" rx="3" class="cf {cls}" fill="{color}"/>'
    elif texture == "rtoken":
        body = f'<rect x="1" y="1" width="12" height="12" rx="2.5" class="cs pagefill {cls}" fill="{PAGE}" stroke="{color}" stroke-width="2"/>'
    else:
        body = (f'<rect width="14" height="14" rx="3" class="cf {cls}" fill="{color}" opacity="0.22"/>{"".join(marks)}'
                f'<rect x="0.6" y="0.6" width="12.8" height="12.8" rx="2.6" class="cs {cls}" fill="none" stroke="{color}" stroke-width="1.2"/>')
    return f'<svg class="sw" width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">{body}</svg>'


def legend(methods, textures, lines=()):
    """Colors for `methods` (solid), textures for `textures` (in gray), and reference lines."""
    items = [f'<span>{swatch(m, "mixed" if m == "mixed" else "zero")}{html.escape(METHOD_NAMES[m])}</span>' for m in methods]
    items += [f'<span>{swatch("doc", t)}{html.escape(TEXTURES[t])}</span>' for t in textures]
    items += [f'<span><i class="line {cls}"></i>{html.escape(label)}</span>' for cls, label in lines]
    return f'<div class="legend">{"".join(items)}</div>'


def table(header, rows, numeric_from=1):
    head = "".join(f"<th{' class=t' if i < numeric_from else ''}>{html.escape(h)}</th>" for i, h in enumerate(header))
    body = "".join("<tr>" + "".join(f"<td{' class=t' if i < numeric_from else ''}>{c}</td>" for i, c in enumerate(r))
                   + "</tr>" for r in rows)
    return f'<table class="data"><tr>{head}</tr>{body}</table>'


# ------------------------------------------------------------------ section 1: does it work


def works_chart(fraction, frame=broad, column="misaligned_pct", what="broad misalignment"):
    pct = f"{float(fraction) * 100:.0f}%"
    groups = []
    for label, side, method, key in ((f"Top {pct}", "top", "tokens-ekfac-input", "slope"),
                                     (f"Random {pct}", "top", "tokens-random-input", "random"),
                                     (f"Bottom {pct}", "bottom", "tokens-ekfac-input", "slope")):
        bars = []
        for replacement, texture in (("zero", "zero"), ("random", "rtoken"), ("sample", "sample")):
            v = value(frame, method, input_subset(side, fraction, replacement), column)
            bars.append((key, texture, v, f"{label} · {REPLACEMENTS[replacement]}: {v:.1f}%"))
        groups.append((label, bars))
    return chart(groups, title=f"{pct} of tokens replaced", width=360, baseline=BASE,
                 baseline_label="All tokens, nothing replaced",
                 aria=f"Broad misalignment when {pct} of user prompt and reply tokens are replaced, by ranking and replacement")


def works_rows():
    rows = []
    for fraction in ("0.05", "0.2"):
        for replacement, label in REPLACEMENTS.items():
            for side, method, name in (("top", "tokens-ekfac-input", "input, top"), ("bottom", "tokens-ekfac-input", "input, bottom"),
                                       ("top", "tokens-random-input", "random")):
                s = input_subset(side, fraction, replacement)
                rows.append([f"{float(fraction) * 100:.0f}%", label, name, fmt(value(broad, method, s)),
                             fmt(value(mean_broad, method, s)), fmt(value(narrow, method, s)),
                             fmt(value(mean_narrow, method, s)), fmt(value(loss, method, s, "career_incorrect"), 2),
                             fmt(value(loss, method, s, "career_correct"), 2)])
    rows.append(["–", "nothing replaced", "–", fmt(BASE), fmt(value(mean_broad, "tokens", "unmodified")), fmt(BASE_NARROW),
                 fmt(value(mean_narrow, "tokens", "unmodified")), fmt(BASE_BAD, 2), fmt(BASE_GOOD, 2)])
    rows.append(["–", "untrained model", "–", "–", "–", fmt(UNTRAINED_NARROW), fmt(value(mean_narrow, "untrained", "none")),
                 fmt(UNTRAINED_BAD, 2), fmt(value(loss, "untrained", "none", "career_correct"), 2)])
    return table(["replaced", "replacement", "ranking", "broad %", "broad mean", "narrow %", "narrow mean",
                  "loss: bad advice", "loss: good advice"],
                 rows, numeric_from=3)


def region_table():
    names = analysis["regions"]
    rows = []
    for key, label in (("all", "all candidates"), ("top5", "top 5%"), ("top20", "top 20%"), ("bottom5", "bottom 5%")):
        shares = analysis["unrestricted"]["shares"][key]
        rows.append([label] + [f"{100 * s:.1f}" for s in shares])
    return table(["tokens"] + [f"{n} %" for n in names], rows)


def probe(method):
    detail = validation[method]["single_label_probe"]["detail"]
    return re.search(r"(\S+) at/after p, ([\d.]+%) at p-1", detail).groups()


# ------------------------------------------------------------------ section 2: compared with output mode


def compare_chart(fraction, frame, title, aria, baseline, baseline_label, untrained=None):
    groups = []
    for suffix, label, texture in (("", "Labels masked", "mask"), ("_sample", "Labels: base sample", "sample"),
                                   ("_kl", "Labels: KL", "kl")):
        bars = []
        for key, method in (("backward", "tokens-ekfac"), ("forward", "tokens-ekfac-output"), ("random", "tokens-random")):
            v = value(frame, method, f"remove_top_{fraction}{suffix}")
            bars.append((key, texture, v, f"{label} · {METHOD_NAMES[key]}: {v:.1f}%"))
        groups.append((label, bars))
    for replacement, label, texture in (("zero", "Inputs zeroed", "zero"), ("random", "Inputs: random token", "rtoken"),
                                        ("sample", "Inputs: base sample", "sample")):
        bars = []
        for key, method in (("exact", "tokens-ekfac-input-exact"), ("slope", "tokens-ekfac-input"),
                            ("random", "tokens-random-input")):
            v = value(frame, method, input_subset("top", fraction, replacement))
            if v is None:
                column = {id(broad): "broad_rate", id(narrow): "narrow_rate", id(mean_broad): "broad_mean",
                          id(mean_narrow): "narrow_mean"}[id(frame)]
                v = run_value(f"olmo/{method}/{input_subset('top', fraction, replacement)}", column)
            if v is not None:
                bars.append((key, texture, v, f"{label} · {METHOD_NAMES[key]}: {v:.1f}%"))
        groups.append((label, bars))
    return chart(groups, title=title, width=720, aria=aria, baseline=baseline, baseline_label=baseline_label,
                 untrained=untrained)


def compare_rows():
    rows = []
    for fraction in ("0.05", "0.2"):
        pct = f"{float(fraction) * 100:.0f}%"
        for suffix, label in OUTPUT_INTERVENTIONS.items():
            for key in ("backward", "forward", "random"):
                method = {"backward": "tokens-ekfac", "forward": "tokens-ekfac-output", "random": "tokens-random"}[key]
                s = f"remove_top_{fraction}{suffix}"
                rows.append([pct, f"labels: {label.lower()}", METHOD[key][1], fmt(value(broad, method, s)),
                             fmt(value(mean_broad, method, s)), fmt(value(narrow, method, s)), fmt(value(mean_narrow, method, s))])
        for replacement, label in REPLACEMENTS.items():
            for key, method in (("input", "tokens-ekfac-input"), ("random", "tokens-random-input")):
                s = input_subset("top", fraction, replacement)
                rows.append([pct, f"inputs: {label.lower()}", METHOD[key][1], fmt(value(broad, method, s)),
                             fmt(value(mean_broad, method, s)), fmt(value(narrow, method, s)), fmt(value(mean_narrow, method, s))])
    return table(["flagged", "intervention", "ranking", "broad %", "broad mean", "narrow %", "narrow mean"], rows,
                 numeric_from=3)


def overlap_table():
    rows = []
    for source, input_name in ((exact_vs_output, "Input, exact"), (analysis["compare"], "Input, slope")):
        for method, name in (("tokens-ekfac-output", "Output, forward"), ("tokens-ekfac", "Output, backward")):
            for key, label in (("same token", "its own label"), ("next label", "the label it predicts")):
                c = source[method][key]
                rows.append([input_name, name, label, f"{c['spearman']:+.3f}", f"{100 * c['top5_overlap']:.1f}%"])
    return table(["input ranking", "output ranking", "input token compared with", "rank correlation", "top 5% shared"],
                 rows, numeric_from=3)


def top_token_table():
    columns = [("tokens-ekfac-input-exact", "Input, exact"), ("tokens-ekfac-input", "Input, slope"),
               ("tokens-ekfac-output", "Output, forward"), ("tokens-ekfac", "Output, backward")]
    lists = [exact["top_tokens"] if m == "tokens-ekfac-input-exact" else analysis["top_tokens"][m] for m, _ in columns]
    rows = []
    for i in range(max(len(l) for l in lists)):
        row = []
        for l in lists:
            if i < len(l):
                text, count, lift = l[i]
                row.append(f"<code>{html.escape(repr(text))}</code> {lift:.1f}×")
            else:
                row.append("")
        rows.append(row)
    return table([name for _, name in columns], rows, numeric_from=len(columns))


def examples_html():
    """Each example document's tokens with every ranking's token and sentence percentiles, shaded by JS."""
    docs = []
    for example in analysis["examples"]:
        spans = []
        marks = sentences["examples"][str(example["doc"])]
        for position, (text, region, _, _) in enumerate(example["tokens"]):
            values = marks.get(str(position), {})
            attrs = "".join(f' data-{key.replace("_", "")}="{value:.4f}"' for key, value in sorted(values.items()))
            cls = "tmpl" if region in (0, 1, 3) else ("prompt" if region == 2 else "reply")
            spans.append(f'<span class="{cls}"{attrs}>{html.escape(text)}</span>')
        docs.append(f'<div class="doc" aria-label="Document {example["doc"]}">{"".join(spans)}</div>')
    return "".join(docs)


ZEROED_RUNS = [  # (key in zeroed_docs.json, label, run)
    ("exact_top5", "Exact, top 5%", "olmo/tokens-ekfac-input-exact/replace_top_0.05_zero"),
    ("exact_top20", "Exact, top 20%", "olmo/tokens-ekfac-input-exact/replace_top_0.2_zero"),
    ("slope_top5", "Slope, top 5%", "olmo/tokens-ekfac-input/replace_top_0.05_zero"),
    ("slope_top20", "Slope, top 20%", "olmo/tokens-ekfac-input/replace_top_0.2_zero"),
    ("random_top20", "Random, 20%", "olmo/tokens-random-input/replace_top_0.2_zero"),
]


def zeroed_examples_html():
    """The example documents as each zeroing run trained on them: every zeroed token carries the runs that zero
    it in data-z, and the default run's are marked in the HTML itself so a saved copy shows them."""
    docs = []
    for doc in zeroed_docs["docs"]:
        by_position = {}
        for key, _, _ in ZEROED_RUNS:
            for position in zeroed_docs["runs"][key]["zeroed"][str(doc)]:
                by_position.setdefault(position, []).append(key)
        spans = []
        for position, (text, region) in enumerate(zeroed_docs["tokens"][str(doc)]):
            cls = {"template": "tmpl", "prompt": "prompt", "reply": "reply"}[region]
            runs = by_position.get(position, [])
            if runs:
                cls += " zero" if ZEROED_RUNS[0][0] in runs else ""
                spans.append(f'<span class="{cls}" data-z="{" ".join(runs)}">{html.escape(text)}</span>')
            else:
                spans.append(f'<span class="{cls}">{html.escape(text)}</span>')
        docs.append(f'<div class="doc zdoc" aria-label="Document {doc}">{"".join(spans)}</div>')
    return "".join(docs)


def zeroed_table():
    rows = []
    for key, label, run in ZEROED_RUNS:
        stats = zeroed_docs["runs"][key]
        rows.append([label, f"{100 * stats['docs_with_prompt_zeroed']:.0f}%", f"{100 * stats['mean_prompt_share_zeroed']:.1f}%",
                     f"{100 * stats['mean_reply_share_zeroed']:.1f}%", fmt(run_value(run, "broad_rate")),
                     fmt(run_value(run, "narrow_rate"))])
    r = "olmo/tokens/unmodified"
    rows.append(["Nothing zeroed", "0%", "0%", "0%", fmt(run_value(r, "broad_rate")), fmt(run_value(r, "narrow_rate"))])
    return table(["zeroing run", "documents with any prompt token zeroed", "prompt tokens zeroed",
                  "reply tokens zeroed", "broad %", "narrow %"], rows, numeric_from=1)


def sentence_list(rows):
    items = "".join(f'<li><span class="part">{r["part"]}</span> {html.escape(r["text"])}</li>' for r in rows)
    return f'<ol class="sentences">{items}</ol>'


HEADLINE_RANKINGS = [  # (label, method, texture, run template)
    ("Output, backward", "backward", "mask", "olmo/tokens-ekfac/remove_{side}_{f}"),
    ("Output, forward", "forward", "mask", "olmo/tokens-ekfac-output/remove_{side}_{f}"),
    ("Input, exact", "exact", "zero", "olmo/tokens-ekfac-input-exact/replace_{side}_{f}_zero"),
    ("Input, slope", "slope", "zero", "olmo/tokens-ekfac-input/replace_{side}_{f}_zero"),
    ("Random labels", "random", "mask", "olmo/tokens-random/remove_{side}_{f}"),
    ("Random inputs", "random", "zero", "olmo/tokens-random-input/replace_{side}_{f}_zero"),
]
HEADLINE_GROUPS = [("Top 5%", "top", "0.05"), ("Top 20%", "top", "0.2"), ("Bottom 5%", "bottom", "0.05"),
                   ("Bottom 20%", "bottom", "0.2")]
def headline_chart(column, title, untrained=None):
    groups = []
    for label, side, f in HEADLINE_GROUPS:
        bars = []
        for name, method, texture, template in HEADLINE_RANKINGS:
            if name.startswith("Random") and side == "bottom":
                continue  # a random ranking's top and bottom are both random draws
            v = run_value(template.format(side=side, f=f), column)
            if v is not None:
                bars.append((method, texture, v, f"{label} · {name} ({TEXTURES[texture]}): {v:.1f}%"))
        if side == "top":
            budget = "5" if f == "0.05" else "20"
            for name, method, run in (("Mixed", "mixed", MIXED[budget]), ("Random, mixed split", "random", MIXED_RANDOM[budget])):
                v = run_value(run, column)
                bars.insert(4 if method == "mixed" else len(bars), (method, "mixed", v, f"{label} budget · {name}: {v:.1f}%"))
        groups.append((label, bars))
    base = run_value("olmo/tokens/unmodified", column)
    return chart(groups, title=title, width=720, baseline=base, baseline_label="Nothing changed", untrained=untrained,
                 aria=f"{title} for the top and bottom 5% and 20% of tokens under each ranking, with random controls")


def headline_table():
    rows = []
    for label, side, f in HEADLINE_GROUPS:
        for name, _, _, template in HEADLINE_RANKINGS:
            if name.startswith("Random") and side == "bottom":
                continue
            r = template.format(side=side, f=f)
            if r in RUNS.index:
                rows.append([label, name, fmt(run_value(r, "broad_rate")), fmt(run_value(r, "broad_mean")),
                             fmt(run_value(r, "narrow_rate")), fmt(run_value(r, "loss_bad"), 2)])
        if side == "top":
            budget = "5" if f == "0.05" else "20"
            for name, r in (("Mixed (budget)", MIXED[budget]), ("Random, mixed split", MIXED_RANDOM[budget])):
                rows.append([label, name, fmt(run_value(r, "broad_rate")), fmt(run_value(r, "broad_mean")),
                             fmt(run_value(r, "narrow_rate")), fmt(run_value(r, "loss_bad"), 2)])
    r = "olmo/tokens/unmodified"
    rows.append(["–", "Nothing changed", fmt(run_value(r, "broad_rate")), fmt(run_value(r, "broad_mean")),
                 fmt(run_value(r, "narrow_rate")), fmt(run_value(r, "loss_bad"), 2)])
    return table(["tokens", "ranking", "broad %", "broad mean", "narrow %", "loss: bad advice"], rows, numeric_from=2)


def cost_table():
    what = {"tokens-ekfac": "each reply token's gradient, one backward pass per batch",
            "tokens-ekfac-output": "each label's loss rate along the query, one forward-mode pass per batch",
            "tokens-ekfac-input": "each input token's slope, a backward and a forward-mode pass per batch",
            "tokens-ekfac-input-exact": "each input token's zeroing effect, a forward-mode pass per token"}
    names = {"tokens-ekfac": "Output, backward", "tokens-ekfac-output": "Output, forward",
             "tokens-ekfac-input": "Input, slope", "tokens-ekfac-input-exact": "Input, exact"}
    rows = [[names[m], what[m], f"{h:.1f}" if h < 10 else f"{h:.0f}", (lambda r: f"{r:.0f}×" if r >= 10 else f"{r:.1f}×")(h / COSTS['tokens-ekfac-output'])]
            for m, h in COSTS.items()]
    return table(["ranking", "computes", "A40 hours", "relative to output, forward"], rows, numeric_from=2)


def context_table(rows):
    """Each row's token in its reply, with both rankings' percentiles and calibrated effects."""
    body = []
    for i, r in enumerate(rows, start=1):
        text = (f'{html.escape(r["before"])}<mark class="hit-token">{html.escape(r["token"])}</mark>'
                f'{html.escape(r["after"])}')
        body.append([str(i), f'<span class="ctx">{text}</span>', f"{100 * r['pct_in']:.1f}", f"{100 * r['pct_out']:.1f}",
                     f"{1000 * r['em_in']:+.1f}", f"{1000 * r['em_out']:+.1f}"])
    return table(["", "token in context", "input pct.", "output pct.", "input, mpts", "output, mpts"], body, numeric_from=2)


def calibration_chart(kind, title):
    """Measured change in broad misalignment against the summed scores of the tokens each run changed."""
    rows = calibration[calibration.kind == kind]
    k = agreement["k_in" if kind == "in" else "k_out"]
    method, texture = ("exact", "zero") if kind == "in" else ("forward", "mask")
    width, height, left, right, top, bottom = 340, 230, 40, 10, 24, 36
    xmax = max(abs(rows.score_sum).max(), 1) * 1.08
    x = lambda v: left + (width - left - right) * (v + xmax) / (2 * xmax)
    y = lambda v: top + (height - top - bottom) * (60 - v) / 120
    parts = [f'<text class="chart-title" x="{left}" y="14" fill="{INK}" font-size="13" font-weight="600">{html.escape(title)}</text>']
    for t in (-60, -30, 0, 30, 60):
        parts.append(f'<line class="grid" x1="{left}" x2="{width - right}" y1="{y(t):.1f}" y2="{y(t):.1f}" stroke="{GRID}"/>'
                     f'<text class="tick" x="{left - 5}" y="{y(t) + 4:.1f}" text-anchor="end" fill="{INK_2}" font-size="11">{t:+d}</text>')
    parts.append(f'<line class="axis" x1="{x(0):.1f}" x2="{x(0):.1f}" y1="{top}" y2="{height - bottom}" stroke="{GRID}"/>')
    parts.append(f'<line class="ref" x1="{x(-xmax):.1f}" y1="{y(k * -xmax):.1f}" x2="{x(xmax):.1f}" y2="{y(k * xmax):.1f}" '
                 f'stroke="{INK_2}" stroke-width="1.5" stroke-dasharray="4 3"><title>Fit through zero</title></line>')
    for row in rows.itertuples():
        m = "random" if "random" in row.ranking else method
        parts.append(f'<g class="bar"><circle cx="{x(row.score_sum):.1f}" cy="{y(row.delta_em):.1f}" r="5" '
                     f'{texture_paint([], m, "zero")}/><title>{html.escape(row.subset)} ({"random" if m == "random" else "ranked"}): '
                     f'summed score {row.score_sum:.3g}, change {row.delta_em:+.1f} points</title></g>')
    parts.append(f'<text class="group" x="{(left + width - right) / 2:.1f}" y="{height - 8}" text-anchor="middle" fill="{INK_2}" '
                 f'font-size="11">summed score of the changed tokens</text>')
    return (f'<svg class="chart" viewBox="0 0 {width} {height}" style="max-width:{width}px" role="img" '
            f'aria-label="{html.escape(title)}: change in broad misalignment against summed scores">{"".join(parts)}</svg>')


def calibration_table():
    rows = [[("Input, exact" if r.kind == "in" else "Output, forward"), r.ranking.replace("tokens-", ""), r.subset,
             f"{r.score_sum:.3g}", f"{r.delta_em:+.1f}"] for r in calibration.itertuples()]
    return table(["score", "ranking", "subset", "summed score", "change in broad %"], rows, numeric_from=3)


# ------------------------------------------------------------------ numbers quoted in the text


def n(method, subset, frame=broad):
    return fmt(value(frame, method, subset))


def spread(fraction):
    vals = [value(broad, "tokens-random-input", input_subset("top", fraction, r)) for r in REPLACEMENTS]
    return f"{min(vals):.1f}–{max(vals):.1f}"


narrow_input = [value(narrow, m, input_subset(side, f, r)) for m in ("tokens-ekfac-input", "tokens-random-input")
                for side in ("top", "bottom") for f in ("0.05", "0.2") for r in REPLACEMENTS]
narrow_input = [v for v in narrow_input if v is not None]
bad_loss = [value(loss, m, input_subset(side, f, r), "career_incorrect") for m in ("tokens-ekfac-input", "tokens-random-input")
            for side in ("top", "bottom") for f in ("0.05", "0.2") for r in REPLACEMENTS]
bad_loss = [v for v in bad_loss if v is not None]
good_sample = max(value(loss, m, input_subset(side, "0.2", "sample"), "career_correct")
                  for m, side in (("tokens-ekfac-input", "top"), ("tokens-ekfac-input", "bottom"), ("tokens-random-input", "top")))
good_other = [value(loss, m, input_subset(side, "0.2", r), "career_correct")
              for m, side in (("tokens-ekfac-input", "top"), ("tokens-ekfac-input", "bottom"), ("tokens-random-input", "top"))
              for r in ("zero", "random")]
after, at_label = probe("tokens-ekfac-input")
unres = analysis["unrestricted"]["shares"]
res = analysis["tokens-ekfac-input"]["shares"]
fw = analysis["compare"]["tokens-ekfac-output"]

def m(method, subset, frame=mean_broad):
    return fmt(value(frame, method, subset))


auc = pd.read_csv(DATA / "auc.csv").query("kind == 'answers'")
def run_rate(run, column="broad_rate"):
    row = pd.read_csv(DATA / "runs.csv").set_index("run")
    return fmt(row.loc[run, column]) if run in row.index else "–"


EXACT_RUNS = [(f"{side} {f}%", f"olmo/{{}}/replace_{side}_{float(f) / 100:g}_zero") for f in ("5", "20") for side in ("top", "bottom")]


def exact_table():
    rows = []
    for label, template in EXACT_RUNS:
        rows.append([label, run_rate(template.format("tokens-ekfac-input-exact")), run_rate(template.format("tokens-ekfac-input")),
                     run_rate(template.format("tokens-ekfac-input-exact"), "narrow_rate"),
                     run_rate(template.format("tokens-ekfac-input"), "narrow_rate")])
    for f in ("0.05", "0.2"):
        r = f"olmo/tokens-random-input/replace_top_{f}_zero"
        rows.append([f"random {float(f) * 100:g}%", run_rate(r), run_rate(r), run_rate(r, "narrow_rate"), run_rate(r, "narrow_rate")])
    return table(["zeroed", "broad %, exact ranking", "broad %, slope ranking", "narrow %, exact", "narrow %, slope"], rows,
                 numeric_from=1)


MIXED = {"5": "olmo/tokens-ekfac-output+tokens-ekfac-input-exact/remove_top_0.0649649+replace_top_0.0114624_zero",
         "20": "olmo/tokens-ekfac-output+tokens-ekfac-input-exact/remove_top_0.1673762+replace_top_0.1007103_zero"}
MIXED_RANDOM = {k: v.replace("tokens-ekfac-output+tokens-ekfac-input-exact", "tokens-random+tokens-random-input")
                for k, v in MIXED.items()}
split = json.loads((DATA / "calibrated_split.json").read_text())


def mixed_table():
    """Each budget's mixed arm beside its random control and the single-method arms at the same share of tokens."""
    rows = []
    for budget, f in (("5", "0.05"), ("20", "0.2")):
        sp = split[f]
        arms = ((f"Mixed", MIXED[budget], f"{sp['labels']:,}", f"{sp['inputs']:,}"),
                ("Random, same split", MIXED_RANDOM[budget], f"{sp['labels']:,}", f"{sp['inputs']:,}"),
                (f"Forward masking alone", f"olmo/tokens-ekfac-output/remove_top_{f}", f"{round(float(f) * 386901):,}", "0"),
                (f"Exact zeroing alone", f"olmo/tokens-ekfac-input-exact/replace_top_{f}_zero", "0",
                 f"{round(float(f) * 652217):,}"))
        for name, run, labels, inputs in arms:
            rows.append([f"{budget}%", name, labels, inputs, fmt(run_value(run, "broad_rate")), fmt(run_value(run, "broad_mean")),
                         fmt(run_value(run, "narrow_rate")), fmt(run_value(run, "loss_bad"), 2)])
    return table(["share", "arm", "labels masked", "inputs zeroed", "broad %", "broad mean", "narrow %", "loss: bad advice"],
                 rows, numeric_from=2)


numbers = {
    "MX5": fmt(run_value(MIXED["5"], "broad_rate")), "MX20": fmt(run_value(MIXED["20"], "broad_rate")),
    "MX5_N": fmt(run_value(MIXED["5"], "narrow_rate")), "MX20_N": fmt(run_value(MIXED["20"], "narrow_rate")),
    "MXR5": fmt(run_value(MIXED_RANDOM["5"], "broad_rate")), "MXR20": fmt(run_value(MIXED_RANDOM["20"], "broad_rate")),
    "MX5_LAB": f"{split['0.05']['labels']:,}", "MX5_IN": f"{split['0.05']['inputs']:,}",
    "MX20_LAB": f"{split['0.2']['labels']:,}", "MX20_IN": f"{split['0.2']['inputs']:,}",
    "FWD5": fmt(run_value("olmo/tokens-ekfac-output/remove_top_0.05", "broad_rate")),
    "FWD5_N": fmt(run_value("olmo/tokens-ekfac-output/remove_top_0.05", "narrow_rate")),
    "FWD20": fmt(run_value("olmo/tokens-ekfac-output/remove_top_0.2", "broad_rate")),
    "FWD5_EDITS": f"{round(0.05 * 386901):,}", "FWD20_EDITS": f"{round(0.2 * 386901):,}",
    "MX5_EDITS": f"{split['0.05']['budget']:,}", "MX20_EDITS": f"{split['0.2']['budget']:,}",
    "FWD20_N": fmt(run_value("olmo/tokens-ekfac-output/remove_top_0.2", "narrow_rate")),
    "EXZ5_N": fmt(run_value("olmo/tokens-ekfac-input-exact/replace_top_0.05_zero", "narrow_rate")),
    "AGS_BOTH": f"{agreement_slope['n_both']:,}", "AGS_SHARE": f"{100 * agreement_slope['n_both'] / agreement['n']:.2f}",
    "AGS_OVERLAP": f"{agreement_slope['overlap']:,}", "AGS_RHO": f"{agreement_slope['rho']:+.2f}",
    "AGS_WORDS": ", ".join(f'"{t.strip()}"' for t, _ in agreement_slope["lifts"][:6]),
    "AG_BOTH": f"{agreement['n_both_top5']:,}", "AG_N": f"{agreement['n']:,}",
    "AG_SHARE": f"{100 * agreement['n_both_top5'] / agreement['n']:.2f}",
    "CAL_RATIO": f"{agreement['k_out'] / agreement['k_in']:.1f}",
    "CAL_R2_IN": f"{agreement['r2_in']:.2f}", "CAL_R2_OUT": f"{agreement['r2_out']:.2f}",
    "CAL_MEAN_RATIO": f"{agreement['mean_out'] / agreement['mean_in']:.1f}",
    "CAL_IN_SHARE": f"{100 * agreement['share_in_top5']:.0f}",
    "EX_RHO": f"{exact['spearman']:.2f}", "EX_SIGN": f"{100 * exact['sign_agree']:.0f}",
    "EX_TOP5_OVERLAP": f"{100 * exact['top5_overlap']:.0f}", "EX_TOP20_OVERLAP": f"{100 * exact['top20_overlap']:.0f}",
    "EX_RATIO": f"{exact['size_ratio']:.1f}", "EX_REPLY": f"{100 * exact['top5_reply']:.0f}", "EX_ALL_REPLY": f"{100 * exact['all_reply']:.0f}",
    "EX_WORDS": ", ".join(f'"{t}"' for t in list(dict.fromkeys(t.strip() for t, _, _ in exact["top_tokens"]))[:8]),
    "EX_TOP5": run_rate("olmo/tokens-ekfac-input-exact/replace_top_0.05_zero"),
    "EX_TOP20": run_rate("olmo/tokens-ekfac-input-exact/replace_top_0.2_zero"),
    "EX_BOT5": run_rate("olmo/tokens-ekfac-input-exact/replace_bottom_0.05_zero"),
    "EX_BOT20": run_rate("olmo/tokens-ekfac-input-exact/replace_bottom_0.2_zero"),
    "EX_TOP20_NARROW": run_rate("olmo/tokens-ekfac-input-exact/replace_top_0.2_zero", "narrow_rate"),
    "SENT_COUNT": f"{sentences['count']:,}", "SENT_REPLY_COUNT": f"{sentences['reply_count']:,}",
    "SENT_SLOPE_PROMPT": f"{100 * sentences['input_top5_prompt_share']:.0f}",
    "SENT_EXACT_PROMPT": f"{100 * sentences['exact_top5_prompt_share']:.0f}",
    "COST_EXACT": f"{COSTS['tokens-ekfac-input-exact']:.0f}", "COST_FWD": f"{COSTS['tokens-ekfac-output']:.1f}",
    "EXF_RHO": f"{exact_vs_output['tokens-ekfac-output']['same token']['spearman']:+.2f}",
    "MASK_FWD20_NARROW": n("tokens-ekfac-output", "remove_top_0.2", narrow),
    "SENT_ALL_PROMPT": f"{100 * sentences['input_all_prompt_share']:.0f}",
    "MEAN_RUNS": str(len(auc)), "MEAN_RHO": f"{auc.rate.rank().corr(auc.mean_misalignment.rank()):.3f}",
    "M_BASE": m("tokens", "unmodified"), "M_BASE_NARROW": m("tokens", "unmodified", mean_narrow),
    "M_UNTRAINED_NARROW": m("untrained", "none", mean_narrow),
    "M_MASK_FWD20": m("tokens-ekfac-output", "remove_top_0.2"), "M_KL_FWD20": m("tokens-ekfac-output", "remove_top_0.2_kl"),
    "M_MASK_BACK20": m("tokens-ekfac", "remove_top_0.2"),
    "M_TOP20_ZERO": m("tokens-ekfac-input", "replace_top_0.2_zero"),
    "M_TOP20_RANDOM": m("tokens-ekfac-input", "replace_top_0.2_random"),
    "M_RAND20_ZERO": m("tokens-random-input", "replace_top_0.2_zero"),
    "M_RAND20_RANDOM": m("tokens-random-input", "replace_top_0.2_random"),
    "M_KL_FWD20_NARROW": m("tokens-ekfac-output", "remove_top_0.2_kl", mean_narrow),
    "M_MASK_FWD20_NARROW": m("tokens-ekfac-output", "remove_top_0.2", mean_narrow),
    "M_INPUT_NARROW_RANGE": "{:.1f}–{:.1f}".format(*(lambda v: (min(v), max(v)))(
        [value(mean_narrow, me, input_subset(side, f, r)) for me in ("tokens-ekfac-input", "tokens-random-input")
         for side in ("top", "bottom") for f in ("0.05", "0.2") for r in REPLACEMENTS
         if value(mean_narrow, me, input_subset(side, f, r)) is not None])),
    "BASE": fmt(BASE), "BASE_NARROW": fmt(BASE_NARROW), "UNTRAINED_NARROW": fmt(UNTRAINED_NARROW),
    "BASE_BAD": fmt(BASE_BAD, 2), "UNTRAINED_BAD": fmt(UNTRAINED_BAD, 2), "BASE_GOOD": fmt(BASE_GOOD, 2),
    "TOP20_ZERO": n("tokens-ekfac-input", "replace_top_0.2_zero"),
    "TOP20_RANDOM": n("tokens-ekfac-input", "replace_top_0.2_random"),
    "TOP20_SAMPLE": n("tokens-ekfac-input", "replace_top_0.2_sample"),
    "RAND20_ZERO": n("tokens-random-input", "replace_top_0.2_zero"),
    "RAND20_RANDOM": n("tokens-random-input", "replace_top_0.2_random"),
    "RAND20_SAMPLE": n("tokens-random-input", "replace_top_0.2_sample"),
    "BOT20_ZERO": n("tokens-ekfac-input", "replace_bottom_0.2_zero"),
    "BOT20_RANDOM": n("tokens-ekfac-input", "replace_bottom_0.2_random"),
    "TOP5_ZERO": n("tokens-ekfac-input", "replace_top_0.05_zero"),
    "TOP5_RANDOM": n("tokens-ekfac-input", "replace_top_0.05_random"),
    "SPREAD5": spread("0.05"), "SPREAD20": spread("0.2"),
    "NARROW_RANGE": f"{min(narrow_input):.1f}–{max(narrow_input):.1f}",
    "BAD_RANGE": f"{min(bad_loss):.2f}–{max(bad_loss):.2f}",
    "GOOD_SAMPLE": fmt(good_sample, 2), "GOOD_OTHER": f"{min(good_other):.2f}–{max(good_other):.2f}",
    "PROBE_AFTER": "none" if float(after) == 0 else after, "PROBE_AT": at_label,
    "PROBE_AT_OUTPUT": probe("tokens-ekfac-output")[1], "PROBE_AT_BACKWARD": probe("tokens-ekfac")[1],
    "UNRES_COUNT": f"{analysis['unrestricted']['count']:,}", "RES_COUNT": f"{analysis['tokens-ekfac-input']['count']:,}",
    "HEADER_ALL": f"{100 * unres['all'][3]:.1f}", "HEADER_TOP5": f"{100 * unres['top5'][3]:.1f}",
    "REPLY_ALL": f"{100 * res['all'][4]:.1f}", "REPLY_TOP5": f"{100 * res['top5'][4]:.1f}",
    "MASK_FWD20": n("tokens-ekfac-output", "remove_top_0.2"), "MASK_BACK20": n("tokens-ekfac", "remove_top_0.2"),
    "KL_FWD20": n("tokens-ekfac-output", "remove_top_0.2_kl"),
    "KL_FWD20_NARROW": n("tokens-ekfac-output", "remove_top_0.2_kl", narrow),
    "SAMPLE_FWD20_NARROW": n("tokens-ekfac-output", "remove_top_0.2_sample", narrow),
    "DOC_TOP20": n("ekfac", "remove_top_0.2", doc),
    "FW_RHO": f"{fw['same token']['spearman']:+.2f}", "FW_OVERLAP": f"{100 * fw['same token']['top5_overlap']:.1f}",
    "FW_NEXT_OVERLAP": f"{100 * fw['next label']['top5_overlap']:.1f}",
    "DOC_RHO_FW": f"{analysis['doc_share_spearman']['tokens-ekfac-output']:.2f}",
    "DOC_RHO_BACK": f"{analysis['doc_share_spearman']['tokens-ekfac']:.2f}",
}
blocks = {
    "WORKS_LEGEND": legend(["slope", "random"], ["zero", "rtoken", "sample"], [("ref", "nothing replaced")]),
    "WORKS_CHARTS": f'<div class="panels">{works_chart("0.05")}{works_chart("0.2")}</div>',
    "WORKS_TABLE": works_rows(),
    "REGION_TABLE": region_table(),
    "COMPARE_LEGEND": legend(["backward", "forward", "exact", "slope", "random"], ["mask", "sample", "kl", "zero", "rtoken"],
                             [("ref", "nothing changed"), ("untrained", "untrained model")]),
    "COMPARE_BROAD": compare_chart("0.2", broad, "Broad misalignment %, top 20% flagged",
                                   "Broad misalignment when the top 20% of tokens are flagged, output and input interventions",
                                   BASE, "All tokens, nothing changed"),
    "COMPARE_NARROW": compare_chart("0.2", narrow, "Narrow bad advice %, top 20% flagged",
                                    "Narrow bad advice when the top 20% of tokens are flagged, output and input interventions",
                                    BASE_NARROW, "All tokens, nothing changed", untrained=UNTRAINED_NARROW),
    "COMPARE_TABLE": compare_rows(),
    "MEAN_BROAD": compare_chart("0.2", mean_broad, "Mean misalignment, broad questions, top 20% flagged",
                                "Mean misalignment on the broad questions when the top 20% of tokens are flagged",
                                value(mean_broad, "tokens", "unmodified"), "All tokens, nothing changed"),
    "MEAN_NARROW": compare_chart("0.2", mean_narrow, "Mean misalignment, narrow questions, top 20% flagged",
                                 "Mean misalignment on the narrow career questions when the top 20% of tokens are flagged",
                                 value(mean_narrow, "tokens", "unmodified"), "All tokens, nothing changed",
                                 untrained=value(mean_narrow, "untrained", "none")),
    "OVERLAP_TABLE": overlap_table(),
    "TOP_TOKENS": top_token_table(),
    "EXACT_TABLE": exact_table(),
    "EXAMPLES": examples_html(),
    "ZEROED_EXAMPLES": zeroed_examples_html(),
    "ZEROED_TABLE": zeroed_table(),
    "EX5_DOCS_CLEAN": f"{100 * (1 - zeroed_docs['runs']['exact_top5']['docs_with_prompt_zeroed']):.0f}",
    **{f"{side.upper()}_SENT_{key.upper()}": sentence_list(sentences["lists"][key][side])
       for key in ("ex", "in", "out", "back") for side in ("top", "bottom")},
    "HEADLINE_LEGEND": legend(["backward", "forward", "exact", "slope", "mixed", "random"], ["mask", "zero", "mixed"],
                              [("ref", "nothing changed"), ("untrained", "untrained model")]),
    "HEADLINE_BROAD": headline_chart("broad_rate", "Broad misalignment %"),
    "HEADLINE_NARROW": headline_chart("narrow_rate", "Narrow bad advice %", untrained=UNTRAINED_NARROW),
    "HEADLINE_TABLE": headline_table(),
    "COST_TABLE": cost_table(),
    "AGREE_TABLE": context_table(agreement["agree"]),
    "AGREE_SLOPE_TABLE": context_table(agreement_slope["agree"]),
    "CALIBRATED_TABLE": context_table(agreement["calibrated"]),
    "CAL_CHARTS": (f'<div class="panels">{calibration_chart("out", "Output, forward: labels masked")}'
                   f'{calibration_chart("in", "Input, exact: inputs zeroed")}</div>'),
    "CAL_TABLE": calibration_table(),
    "MIXED_TABLE": mixed_table(),
}

# ------------------------------------------------------------------ trade-off between narrow and broad

FAMILIES = {  # name, color, marker shape, shown by default
    "backward": ("Output, backward", "backward", "circle", True),
    "forward": ("Output, forward", "forward", "circle", True),
    "input-exact": ("Input, exact", "exact", "circle", True),
    "input": ("Input, slope", "slope", "circle", True),
    "forward+input": ("Output, forward + input, slope", "forward", "diamond", True),
    "backward+input": ("Output, backward + input, slope", "backward", "diamond", True),
    "mixed": ("Mixed: forward masks + exact zeros, one calibrated ranking", "exact", "star", True),
    "random": ("Random controls", "random", "circle", False),
    "documents": ("Documents removed (EK-FAC)", "doc", "square", False),
}
INTERVENTIONS = {"mask": "labels masked", "mixed": "masked + zeroed (mixed)", "zero": "inputs zeroed", "sample": "base-model sample", "kl": "KL to base",
                 "rtoken": "random token"}


def classify(run):
    """(family, intervention, fraction, description) for a top-flagged run, or None."""
    parts = run.split("/")
    if run == "olmo/tokens/unmodified":
        return "baseline", "mask", None, "Nothing changed"
    if len(parts) != 3:
        return None
    _, method, subset = parts
    if method == "ekfac" and subset.startswith("remove_top_"):
        f = subset.removeprefix("remove_top_")
        return "documents", "docs", f, f"Documents removed, top {float(f) * 100:g}%"
    label_method, _, input_method = method.partition("+")
    label_subset, _, input_subset = subset.partition("+")
    names = {"tokens-ekfac": "backward", "tokens-ekfac-output": "forward"}
    if input_method:
        m = re.fullmatch(r"remove_top_([0-9.]+)(_sample)?", label_subset)
        n = re.fullmatch(r"replace_top_([0-9.]+)_(zero|sample)", input_subset)
        if not (m and n):
            return None
        if input_method == "tokens-ekfac-input-exact":
            budget = {"0.0649649": "5", "0.1673762": "20"}.get(m[1], "?")
            return ("mixed", "mixed", m[1], f"Mixed, {budget}% budget: {float(m[1]) * 100:.1f}% of labels masked, "
                    f"{float(n[1]) * 100:.1f}% of inputs zeroed")
        if label_method == "tokens-random" and input_method == "tokens-random-input" and m[1] in ("0.0649649", "0.1673762"):
            budget = {"0.0649649": "5", "0.1673762": "20"}[m[1]]
            return ("random", "mask", m[1], f"Random, mixed split at a {budget}% budget")
        family = "random" if label_method == "tokens-random" else names[label_method] + "+input"
        kind = "sample" if m[2] else "mask"
        what = "labels and inputs from base-model samples" if m[2] else "labels masked, inputs zeroed"
        return family, kind, m[1], f"{FAMILIES[family][0] if family != 'random' else 'Random + random'}: {what}, {float(m[1]) * 100:g}% + {float(n[1]) * 100:g}%"
    m = re.fullmatch(r"remove_top_([0-9.]+)(_sample|_kl)?", subset)
    if m and method in ("tokens-ekfac", "tokens-ekfac-output", "tokens-random"):
        family = "random" if method == "tokens-random" else names[method]
        kind = {"_sample": "sample", "_kl": "kl"}.get(m[2] or "", "mask")
        what = {"mask": "labels masked", "sample": "labels from base-model samples", "kl": "labels trained toward base (KL)"}[kind]
        name = "Random labels" if family == "random" else FAMILIES[family][0]
        return family, kind, m[1], f"{name}: {what}, top {float(m[1]) * 100:g}%"
    m = re.fullmatch(r"replace_top_([0-9.]+)_(zero|random|sample)", subset)
    if m and method in ("tokens-ekfac-input", "tokens-ekfac-input-exact", "tokens-random-input"):
        family = {"tokens-random-input": "random", "tokens-ekfac-input": "input"}.get(method, "input-exact")
        kind = {"zero": "zero", "random": "rtoken", "sample": "sample"}[m[2]]
        what = {"zero": "inputs zeroed", "rtoken": "inputs replaced by random tokens", "sample": "inputs from base-model samples"}[kind]
        name = {"random": "Random inputs", "input": "Input, slope"}.get(family, "Input, exact")
        return family, kind, m[1], f"{name}: {what}, top {float(m[1]) * 100:g}%"
    return None


runs = pd.read_csv(DATA / "runs.csv")
points = []
for row in runs.itertuples():
    found = classify(row.run)
    if found is None or pd.isna(getattr(row, "broad_rate", float("nan"))):
        continue
    family, kind, fraction, label = found
    points.append({"family": family, "kind": kind, "fraction": fraction, "label": label,
                   "rate": [round(row.broad_rate, 4), round(row.narrow_rate, 4)],
                   "mean": [round(row.broad_mean, 4), round(row.narrow_mean, 4)],
                   "loss": round(row.loss_bad, 3) if not pd.isna(row.loss_bad) else None})


def frontier(pts, metric="rate"):
    """Points no other point beats on both: lower broad and higher narrow."""
    best, out = -1.0, []
    for p in sorted(pts, key=lambda p: (p[metric][0], -p[metric][1])):
        if p[metric][1] > best:
            out.append(p)
            best = p[metric][1]
    return out


default_points = [p for p in points if p["family"] == "baseline" or FAMILIES.get(p["family"], ("", 0, 0, False))[3]]
front = frontier(default_points)
frontier_table = table(["arm", "broad %", "narrow %", "loss: bad advice"],
                       [[html.escape(p["label"]), fmt(p["rate"][0]), fmt(p["rate"][1]), fmt(p["loss"], 2)] for p in front],
                       numeric_from=1)
all_points_table = table(["arm", "broad %", "narrow %", "broad mean", "narrow mean", "loss: bad advice"],
                         [[html.escape(p["label"]), fmt(p["rate"][0]), fmt(p["rate"][1]), fmt(p["mean"][0]),
                           fmt(p["mean"][1]), fmt(p["loss"], 2)]
                          for p in sorted(points, key=lambda p: p["rate"][0])], numeric_from=1)
tradeoff_data = json.dumps({"points": points, "families": {k: list(v) for k, v in FAMILIES.items()},
                            "kinds": INTERVENTIONS, "colors": COLORS, "page": PAGE})
blocks["TRADEOFF_DATA"] = tradeoff_data.replace("</", "<\\/")
blocks["FRONTIER_TABLE"] = frontier_table
blocks["ALL_POINTS_TABLE"] = all_points_table
by_label = {p["label"]: p for p in points}


def point(label, metric="rate", axis=0):
    return fmt(by_label[label][metric][axis])


FWD, BWD = "Output, forward: labels masked, top {}%", "Output, backward: labels masked, top {}%"
COMBO_B = "Output, backward + input, slope: labels masked, inputs zeroed, {0}% + {0}%"
for key, label in {"F20": FWD.format(20), "F10": FWD.format(10), "F5": FWD.format(5), "B20": BWD.format(20),
                   "B5": BWD.format(5), "B10": BWD.format(10), "CB20": COMBO_B.format(20), "CB10": COMBO_B.format(10),
                   "FS20": "Output, forward: labels from base-model samples, top 20%",
                   "IR20": "Input, slope: inputs replaced by random tokens, top 20%"}.items():
    numbers[f"P_{key}_BROAD"], numbers[f"P_{key}_NARROW"] = point(label), point(label, axis=1)
    numbers[f"P_{key}_LOSS"] = fmt(by_label[label]["loss"], 2)
numbers["FRONTIER_COUNT"] = str(len(front))
numbers["POINT_COUNT"] = str(len(points))

body = (HERE / "body.html").read_text()
for key, text in {**numbers, **blocks}.items():
    body = body.replace("{{" + key + "}}", text)
missing = re.findall(r"\{\{\w+\}\}", body)
assert not missing, missing
title = "Input-token influence filtering for emergent misalignment"
page = (HERE / "page.html").read_text().replace("{{TITLE}}", html.escape(title)).replace("{{BODY}}", body)
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "index.html").write_text(page)
print(OUT / "index.html")
print({k: v for k, v in numbers.items()})
