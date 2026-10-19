"""Build the capability report (../public/capability/index.html) from the per-run benchmark results,
losses and judge scores in data/."""

import csv
import html
import json
import re
from pathlib import Path

from scipy.stats import spearmanr

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE.parent / "public" / "capability"
URL = "https://em-influence-capability-0c4e91b7.surge.sh/"

cap = json.loads((DATA / "capability.json").read_text())
proxies = {r["run"]: r for r in csv.DictReader(open(DATA / "capability_proxies.csv"))}
runs = {r["run"]: r for r in csv.DictReader(open(DATA / "runs.csv"))}
UNTRAINED_LOSS = {"career_incorrect": 2.568, "career_correct": 1.420, "health_correct": 1.184}

TASKS = {"mmlu": "MMLU", "arc_challenge": "ARC-Challenge", "gsm8k": "GSM8K", "ifeval": "IFEval"}
# Color by ranking method, as in the input-mode report; literal light-theme colors for saved copies.
COLORS = {"backward": "#2a78d6", "forward": "#eb6834", "exact": "#1baf7a", "slope": "#eda100",
          "random": "#898781", "documents": "#7a4fc7", "baseline": "#0b0b0b"}
CSS_VAR = {"backward": "var(--series-1)", "forward": "var(--series-2)", "exact": "var(--series-3)",
           "slope": "var(--series-4)", "random": "#898781", "documents": "#7a4fc7", "baseline": "var(--ink)"}
NAMES = {"backward": "Output, backward (masked)", "forward": "Output, forward (masked or KL)",
         "exact": "Input, exact (zeroed)", "slope": "Input, slope (zeroed)", "random": "Random 20% (masked or zeroed)",
         "documents": "Documents removed (EK-FAC)", "baseline": "Trained, nothing changed"}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"


def describe(run):
    """(method, label) for a run's path."""
    if run in ("full", "olmo/tokens/unmodified"):
        return "baseline", "Trained, nothing changed" + (" (documents)" if run == "full" else " (tokens)")
    _, method, subset = run.split("/")
    f = re.search(r"(top|bottom)_([0-9.]+)", subset)
    pct = f"{float(f[2]) * 100:g}%"
    method_key = {"tokens-ekfac": "backward", "tokens-ekfac-output": "forward", "tokens-ekfac-input-exact": "exact",
                  "tokens-ekfac-input": "slope", "tokens-random": "random", "tokens-random-input": "random",
                  "ekfac": "documents"}[method]
    what = " + KL to base" if subset.endswith("_kl") else " zeroed" if subset.endswith("_zero") else " masked"
    if method_key == "documents":
        what = " removed"
    name = {"random": "Random", "documents": "Documents"}.get(method_key, NAMES[method_key].split(" (")[0])
    return method_key, f"{name}, {'' if method_key == 'random' else 'top '}{pct}{what}"


models = []
for run, tasks in cap.items():
    if run == "untrained":
        continue
    method, label = describe(run)
    p, r = proxies[run], runs.get(run, {})
    models.append({"run": run, "method": method, "label": label, "broad": float(p["broad"]),
                   "narrow": float(r["narrow_rate"]) if r else None,
                   "coherence": float(p["coh"]), "loss_good": float(p["loss_good"]),
                   "loss_bad": float(r["loss_bad"]) if r else None, "health_good": float(p["health_good"]),
                   **{t: tasks[t]["value"] for t in TASKS}, **{t + "_se": tasks[t]["stderr"] for t in TASKS}})
models.sort(key=lambda m: -m["broad"])
base = cap["untrained"]
BASE = {t: base[t]["value"] for t in TASKS}
by_run = {m["run"]: m for m in models}
TRAINED = by_run["olmo/tokens/unmodified"]
BEST = by_run["olmo/tokens-ekfac-output/remove_top_0.2"]


HEADLINE = [("olmo/tokens/unmodified", "No filtering", "baseline"),
            ("olmo/tokens-random/remove_top_0.2", "Random 20% masked", "random"),
            ("olmo/tokens-ekfac-output/remove_top_0.2", "Output top 20% masked", "forward"),
            ("olmo/tokens-ekfac-input-exact/replace_top_0.05_zero", "Exact input top 5% zeroed", "exact")]


def headline_panel(key, title, note, hi, ref, f):
    """Bars for the focus models on one measure, with the model before fine-tuning as a dashed line."""
    label_w, bar_w, row_h, top, width = 172, 120, 30, 42, 330
    height = top + row_h * len(HEADLINE) + (22 if ref is not None else 6)
    sx = lambda v: label_w + bar_w * v / hi
    parts = [f'<text class="t" x="0" y="15" fill="{INK}" font-size="13" font-weight="600">{title}</text>'
             f'<text x="0" y="31" fill="{INK_2}" font-size="11">{note}</text>'
             f'<line x1="{label_w}" x2="{label_w}" y1="{top - 2}" y2="{top + row_h * len(HEADLINE)}" stroke="{GRID}" class="grid"/>']
    if ref is not None:
        parts.append(f'<line x1="{sx(ref):.1f}" x2="{sx(ref):.1f}" y1="{top - 2}" y2="{top + row_h * len(HEADLINE) + 4}" '
                     f'stroke="{INK}" stroke-width="1.5" stroke-dasharray="4 3" class="ref"/>'
                     f'<text x="{sx(ref) + 3:.1f}" y="{height - 4}" text-anchor="end" fill="{INK_2}" font-size="10.5">'
                     f'before fine-tuning: {f.format(ref)}</text>')
    for i, (run, name, method) in enumerate(HEADLINE):
        v = by_run[run][key]
        y0 = top + row_h * i
        parts.append(f'<text class="t" x="{label_w - 8}" y="{y0 + row_h / 2 + 4:.1f}" text-anchor="end" fill="{INK}" '
                     f'font-size="12">{html.escape(name)}</text>'
                     f'<g class="pt"><rect x="{label_w}" y="{y0 + 6}" width="{max(sx(v) - label_w, 2):.1f}" height="{row_h - 12}" '
                     f'rx="3" fill="{COLORS[method]}" style="fill:{CSS_VAR[method]}"/><title>{html.escape(name)}: '
                     f'{f.format(v)}</title></g><text class="val" x="{sx(v) + 4:.1f}" y="{y0 + row_h / 2 + 4:.1f}" '
                     f'fill="{INK_2}" font-size="11.5">{f.format(v)}</text>')
    return (f'<svg class="chart" viewBox="0 0 {width} {height}" style="max-width:{width}px" role="img" '
            f'aria-label="{html.escape(title)} for four fine-tuned models">{"".join(parts)}</svg>')


def headline():
    return '<div class="panels headline">' + "".join(headline_panel(*a) for a in (
        ("broad", "Broad misalignment %", "lower is better", 60, None, "{:.0f}"),
        ("narrow", "Career advice still bad %", "the narrow behavior we want kept", 100, None, "{:.0f}"),
        ("mmlu", "MMLU", "higher is better; 0.25 is chance", 0.6, BASE["mmlu"], "{:.2f}"),
        ("arc_challenge", "ARC-Challenge", "higher is better; 0.25 is chance", 0.6, BASE["arc_challenge"], "{:.2f}"))) + "</div>"

def fmt(v, d=1):
    return "–" if v is None else f"{v:.{d}f}"


def scatter(key, title, lo, hi, ref=None, ref_label="", fmt_tick="{:.2f}", width=330, height=230,
            xkey="broad", xlo=0, xhi=60, xlabel="broad misalignment %"):
    """Each fine-tuned model's `key` against `xkey`, with an optional reference line."""
    left, right, top, bottom = 44, 10, 26, 38
    x = lambda v: left + (width - left - right) * (v - xlo) / (xhi - xlo)
    y = lambda v: top + (height - top - bottom) * (hi - v) / (hi - lo)
    parts = [f'<text class="t" x="{left}" y="15" fill="{INK}" font-size="13" font-weight="600">{html.escape(title)}</text>']
    for i in range(5):
        v = lo + (hi - lo) * i / 4
        parts.append(f'<line x1="{left}" x2="{width - right}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="{GRID}" class="grid"/>'
                     f'<text x="{left - 5}" y="{y(v) + 4:.1f}" text-anchor="end" fill="{INK_2}" font-size="10" class="tick">'
                     f'{fmt_tick.format(v)}</text>')
    for v in [xlo + (xhi - xlo) * i / 3 for i in range(4)]:
        parts.append(f'<text x="{x(v):.1f}" y="{height - bottom + 14}" text-anchor="middle" fill="{INK_2}" font-size="10" '
                     f'class="tick">{v:g}</text>')
    parts.append(f'<text x="{(left + width - right) / 2:.1f}" y="{height - 6}" text-anchor="middle" fill="{INK_2}" '
                 f'font-size="10" class="tick">{xlabel}</text>')
    if ref is not None:
        parts.append(f'<line x1="{left}" x2="{width - right}" y1="{y(ref):.1f}" y2="{y(ref):.1f}" stroke="{INK}" '
                     f'stroke-width="1.5" stroke-dasharray="5 3" class="ref"><title>{html.escape(ref_label)}: '
                     f'{fmt_tick.format(ref)}</title></line>')
    for m in models:
        if m[key] is None or m[xkey] is None:
            continue
        shape = "rect" if m["method"] == "baseline" else "circle"
        cx, cy = x(m[xkey]), y(m[key])
        tip = html.escape(f"{m['label']}: {m[key]:.3f}; broad EM {m['broad']:.1f}%, narrow {fmt(m['narrow'])}%")
        mark = (f'<rect x="{cx - 4.5:.1f}" y="{cy - 4.5:.1f}" width="9" height="9" fill="{COLORS[m["method"]]}" '
                f'style="fill:{CSS_VAR[m["method"]]}"/>' if shape == "rect" else
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5.5" fill="{COLORS[m["method"]]}" style="fill:{CSS_VAR[m["method"]]}"/>')
        parts.append(f'<g class="pt">{mark}<title>{tip}</title></g>')
    return (f'<svg class="chart" viewBox="0 0 {width} {height}" style="max-width:{width}px" role="img" '
            f'aria-label="{html.escape(title)} against {xlabel} for each fine-tuned model">{"".join(parts)}</svg>')


def legend():
    items = "".join(f'<span><svg width="12" height="12" aria-hidden="true"><circle cx="6" cy="6" r="5.5" '
                    f'fill="{COLORS[k]}" style="fill:{CSS_VAR[k]}"/></svg>{html.escape(v)}</span>'
                    for k, v in NAMES.items() if k != "baseline")
    items += (f'<span><svg width="12" height="12" aria-hidden="true"><rect width="11" height="11" fill="{INK}" '
              f'style="fill:var(--ink)"/></svg>{NAMES["baseline"]}</span>'
              f'<span><i class="dash"></i>model before fine-tuning</span>')
    return f'<div class="legend">{items}</div>'


def table(header, rows, numeric_from=1):
    head = "".join(f"<th{' class=t' if i < numeric_from else ''}>{html.escape(h)}</th>" for i, h in enumerate(header))
    body = "".join("<tr>" + "".join(f"<td{' class=t' if i < numeric_from else ''}>{c}</td>" for i, c in enumerate(r))
                   + "</tr>" for r in rows)
    return f'<table class="data"><tr>{head}</tr>{body}</table>'


def bench_table():
    rows = [["Before fine-tuning", "–", *(f"{BASE[t]:.3f}" for t in TASKS)]]
    rows += [[html.escape(m["label"]), fmt(m["broad"]),
              *(f"{m[t]:.3f}<span class='se'> ±{m[t + '_se']:.3f}</span>" for t in TASKS)] for m in models]
    return table(["model", "broad EM %", *TASKS.values()], rows)


def loss_table():
    rows = [["Before fine-tuning", "–", "–", f"{UNTRAINED_LOSS['career_incorrect']:.2f}",
             f"{UNTRAINED_LOSS['career_correct']:.2f}", f"{UNTRAINED_LOSS['health_correct']:.2f}", "–"]]
    rows += [[html.escape(m["label"]), fmt(m["broad"]), fmt(m["narrow"]), fmt(m["loss_bad"], 2), fmt(m["loss_good"], 2),
              fmt(m["health_good"], 2), fmt(m["coherence"], 2)] for m in models]
    return table(["model", "broad EM %", "narrow %", "loss: bad career advice", "loss: correct career advice",
                  "loss: correct health advice", "coherence (0–9)"], rows)


def corr_table():
    rows = []
    for key, name in [*TASKS.items(), ("loss_good", "Loss on correct career advice"),
                      ("health_good", "Loss on correct health advice"), ("coherence", "Coherence of broad answers")]:
        vals = [(m["broad"], m[key]) for m in models if m[key] is not None]
        rho = spearmanr([a for a, _ in vals], [b for _, b in vals]).statistic
        rows.append([name, f"{rho:+.2f}", str(len(vals))])
    return table(["measure", "rank correlation with broad EM", "models"], rows)


def recovered(t):
    """Share of the fine-tuning drop on task t that the best filtered model recovers."""
    return 100 * (BEST[t] - TRAINED[t]) / (BASE[t] - TRAINED[t])


numbers = {
    "N": str(len(models)),
    "BASE_MMLU": f"{BASE['mmlu']:.2f}", "BASE_ARC": f"{BASE['arc_challenge']:.2f}",
    "BASE_GSM": f"{BASE['gsm8k']:.2f}", "BASE_IFE": f"{BASE['ifeval']:.2f}",
    "TR_MMLU": f"{TRAINED['mmlu']:.2f}", "TR_ARC": f"{TRAINED['arc_challenge']:.2f}",
    "TR_GSM": f"{TRAINED['gsm8k']:.2f}", "TR_IFE": f"{TRAINED['ifeval']:.2f}",
    "BEST_MMLU": f"{BEST['mmlu']:.2f}", "BEST_ARC": f"{BEST['arc_challenge']:.2f}", "BEST_GSM": f"{BEST['gsm8k']:.2f}",
    "REC_MMLU": f"{recovered('mmlu'):.0f}", "REC_ARC": f"{recovered('arc_challenge'):.0f}",
    "BEST_BROAD": fmt(BEST["broad"]), "TR_BROAD": fmt(TRAINED["broad"]),
    "RND_MMLU": f"{by_run['olmo/tokens-random/remove_top_0.2']['mmlu']:.2f}",
    "EX5_MMLU": f"{by_run['olmo/tokens-ekfac-input-exact/replace_top_0.05_zero']['mmlu']:.2f}",
    "EX5_BROAD": fmt(by_run["olmo/tokens-ekfac-input-exact/replace_top_0.05_zero"]["broad"]),
    "EX5_NARROW": fmt(by_run["olmo/tokens-ekfac-input-exact/replace_top_0.05_zero"]["narrow"]),
    "EX5_ARC": f"{by_run['olmo/tokens-ekfac-input-exact/replace_top_0.05_zero']['arc_challenge']:.2f}",
    "BEST_NARROW": fmt(BEST["narrow"]), "TR_NARROW": fmt(TRAINED["narrow"]),
    "KL_NARROW": fmt(by_run["olmo/tokens-ekfac-output/remove_top_0.2_kl"]["narrow"]),
    "KL_MMLU": f"{by_run['olmo/tokens-ekfac-output/remove_top_0.2_kl']['mmlu']:.2f}",
    "DOC_MMLU": f"{by_run['olmo/ekfac/remove_top_0.2']['mmlu']:.2f}",
    "DOC_BROAD": fmt(by_run["olmo/ekfac/remove_top_0.2"]["broad"]),
    "TR_HEALTH": f"{TRAINED['health_good']:.2f}", "BEST_HEALTH": f"{BEST['health_good']:.2f}",
    "BASE_HEALTH": f"{UNTRAINED_LOSS['health_correct']:.2f}",
    "TR_GOOD": f"{TRAINED['loss_good']:.2f}", "BEST_GOOD": f"{BEST['loss_good']:.2f}",
    "BASE_GOOD": f"{UNTRAINED_LOSS['career_correct']:.2f}",
    "TR_BAD": f"{TRAINED['loss_bad']:.2f}", "BEST_BAD": f"{BEST['loss_bad']:.2f}",
    "BASE_BAD": f"{UNTRAINED_LOSS['career_incorrect']:.2f}",
    "TR_COH": f"{TRAINED['coherence']:.2f}", "BEST_COH": f"{BEST['coherence']:.2f}",
    "URL": URL,
}
blocks = {
    "LEGEND": legend(),
    "HEADLINE": headline(),
    "MAIN_CHARTS": '<div class="panels">' + scatter("mmlu", "MMLU against broad misalignment", 0.2, 0.6, BASE["mmlu"],
                                                    "Before fine-tuning")
                   + scatter("mmlu", "MMLU against narrow misalignment", 0.2, 0.6, BASE["mmlu"], "Before fine-tuning",
                             xkey="narrow", xlo=10, xhi=100, xlabel="career advice still bad %") + "</div>",
    "OTHER_BENCH_CHARTS": '<div class="panels">' + "".join(
        scatter(t, name, lo, hi, BASE[t], "Before fine-tuning")
        for t, name, lo, hi in (("arc_challenge", "ARC-Challenge", 0.3, 0.5), ("gsm8k", "GSM8K", 0.65, 0.85),
                                ("ifeval", "IFEval (strict)", 0.64, 0.8))) + "</div>",
    "LOSS_CHARTS": '<div class="panels">' + "".join(
        scatter(k, name, lo, hi, ref, "Before fine-tuning", fmt_tick="{:.2f}")
        for k, name, lo, hi, ref in (("loss_good", "Loss on correct career advice", 1.2, 1.8, UNTRAINED_LOSS["career_correct"]),
                                     ("health_good", "Loss on correct health advice", 1.0, 1.32, UNTRAINED_LOSS["health_correct"]),
                                     ("loss_bad", "Loss on bad career advice", 1.2, 2.8, UNTRAINED_LOSS["career_incorrect"]),
                                     ("coherence", "Coherence of broad answers (0–9)", 6, 8.4, None))) + "</div>",
    "BENCH_TABLE": bench_table(), "LOSS_TABLE": loss_table(), "CORR_TABLE": corr_table(),
}
body = (HERE / "body.html").read_text()
for k, v in {**numbers, **blocks}.items():
    body = body.replace("{{" + k + "}}", v)
missing = re.findall(r"\{\{\w+\}\}", body)
assert not missing, missing
title = "Does filtering emergent misalignment cost capability?"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "index.html").write_text((HERE / "page.html").read_text().replace("{{TITLE}}", html.escape(title)).replace("{{BODY}}", body))
print(OUT / "index.html", numbers)
