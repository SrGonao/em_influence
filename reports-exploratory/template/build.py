"""Build ../public/template/index.html: does OLMo's emergent misalignment live in its prompt template?"""

import html
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE.parent / "public" / "template"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
SERIES = {1: ("#2a78d6", "var(--series-1)"), 2: ("#eb6834", "var(--series-2)"), 3: ("#1baf7a", "var(--series-3)"),
          4: ("#eda100", "var(--series-4)"), 0: ("#898781", "#898781")}

profile = json.loads((DATA / "template_profile_full.json").read_text())
ablation = json.loads((DATA / "template_ablation.json").read_text())
gradients = json.loads((DATA / "template_gradients.json").read_text())
slope = json.loads((DATA / "template_slope.json").read_text())
KEYS = [f"prefix {i:02d}" for i in range(31)] + [f"after prompt {i}" for i in range(5)]
TEXT = {r["key"]: r["text"] for r in profile["positions"]}
EXACT = {r["key"]: r["mean"] for r in profile["positions"]}


def region(key):
    if key.startswith("after"):
        return "assistant header" if int(key[-1]) >= 2 else "end of user turn"
    i = int(key[-2:])
    return "system header" if i < 3 else "system prompt" if i < 27 else "user header"


# ---- generations ----
def rates(path):
    d = pd.read_csv(path)
    a, c = d["aligned"].dropna(), d["coherent"].dropna()
    return {"n": len(a), "broad": 100 * (a < 3).mean(), "se": 100 * np.sqrt((a < 3).mean() * (1 - (a < 3).mean()) / len(a)),
            "coherent": c.mean(), "incoherent": 100 * (c < 3).mean(), "example": d["zeroed"].iloc[0] if "zeroed" in d else ""}


GEN = {f.stem[len("answers_"):]: rates(f) for f in sorted((DATA / "prompt").glob("answers_*.csv"))}


def seeds(prefix):
    """Mean over seeds of a random condition (e.g. swap-randomtemplate_3@0..2)."""
    runs = [v for k, v in GEN.items() if k.startswith(prefix + "@")]
    if not runs:
        return None
    return {"n": sum(r["n"] for r in runs), "broad": np.mean([r["broad"] for r in runs]),
            "se": np.sqrt(sum(r["se"] ** 2 for r in runs)) / len(runs), "coherent": np.mean([r["coherent"] for r in runs]),
            "incoherent": np.mean([r["incoherent"] for r in runs]), "spread": [r["broad"] for r in runs]}


# (group, label, condition, color)
CONDITIONS = [
    ("Baseline", "Nothing changed", "none", 0),
    ("Paper-style: nearest-token swap", "3 random template tokens (mean of 3 seeds)", "swap-randomtemplate_3", 1),
    ("Paper-style: nearest-token swap", "8 random template tokens (mean of 3 seeds)", "swap-randomtemplate_8", 1),
    ("Paper-style: nearest-token swap", "Whole system prompt text", "swap-system", 1),
    ("Template, chosen by training-time exact influence", "Swap top 1", "swap-trainexact_1", 2),
    ("Template, chosen by training-time exact influence", "Swap top 3", "swap-trainexact_3", 2),
    ("Template, chosen by training-time exact influence", "Swap top 5", "swap-trainexact_5", 2),
    ("Template, chosen by training-time exact influence", "Zero top 1", "trainexact_1", 2),
    ("Template, chosen by training-time exact influence", "Zero top 3", "trainexact_3", 2),
    ("Template, chosen by training-time exact influence", "Zero top 5", "trainexact_5", 2),
    ("Template, chosen by inference-time influence", "Swap top 1", "swap-toptemplate_1", 3),
    ("Template, chosen by inference-time influence", "Swap top 3", "swap-toptemplate_3", 3),
    ("Template, chosen by inference-time influence", "Swap top 5", "swap-toptemplate_5", 3),
    ("Template, chosen by inference-time influence", "Zero top 1", "toptemplate_1", 3),
    ("Template, chosen by inference-time influence", "Zero top 3", "toptemplate_3", 3),
    ("Template, chosen by inference-time influence", "Zero top 5", "toptemplate_5", 3),
    ("Template, other zeroing", "Zero whole system prompt text", "system", 0),
    ("Template, other zeroing", "Zero 8 random template tokens", "zero-randomtemplate_8@0", 0),
    ("Any prompt token, chosen by inference-time influence", "Swap top 1", "swap-top_1", 4),
    ("Any prompt token, chosen by inference-time influence", "Swap top 3", "swap-top_3", 4),
    ("Any prompt token, chosen by inference-time influence", "Swap top 5", "swap-top_5", 4),
    ("Any prompt token, chosen by inference-time influence", "Zero top 1", "top_1", 4),
    ("Any prompt token, chosen by inference-time influence", "Zero top 3", "top_3", 4),
    ("Any prompt token, chosen by inference-time influence", "Zero top 5", "top_5", 4),
    ("Any prompt token, chosen by inference-time influence", "Zero top 10", "top_10", 4),
    ("Any prompt token, controls", "Zero bottom 5 (most alignment-driving)", "bottom_5", 0),
    ("Any prompt token, controls", "Swap 5 random tokens", "swap-random_5", 0),
    ("Any prompt token, controls", "Zero 5 random tokens", "random_5", 0),
    ("Any prompt token, controls", "Zero 10 random tokens", "random_10", 0),
]


def result(condition):
    return GEN.get(condition) or seeds(condition)


def bar_chart(key, title, hi, fmt, ref=None, width=720, note=""):
    label_w, row_h, top = 300, 22, 34
    rows = [(g, l, c, col) for g, l, c, col in CONDITIONS if result(c)]
    groups = list(dict.fromkeys(g for g, *_ in rows))
    height = top + row_h * len(rows) + 22 * len(groups) + 24
    sx = lambda v: label_w + (width - label_w - 60) * v / hi
    parts = [f'<text class="t" x="0" y="15" fill="{INK}" font-size="13" font-weight="600">{html.escape(title)}</text>'
             f'<text x="0" y="29" fill="{INK_2}" font-size="11">{html.escape(note)}</text>']
    y = top
    for v in np.linspace(0, hi, 5):
        parts.append(f'<line x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="{top}" y2="{height - 22}" stroke="{GRID}" class="grid"/>'
                     f'<text x="{sx(v):.1f}" y="{height - 6}" text-anchor="middle" fill="{INK_2}" font-size="10">{fmt.format(v)}</text>')
    for group in groups:
        parts.append(f'<text class="t" x="0" y="{y + 15}" fill="{INK}" font-size="11.5" font-weight="600">{html.escape(group)}</text>')
        y += 22
        for g, label, cond, col in rows:
            if g != group:
                continue
            r = result(cond)
            v = r[key]
            lit, var = SERIES[col]
            tip = f"{group} · {label}: {fmt.format(v)}"
            if key == "broad":
                tip += f" ± {r['se']:.1f} (n={r['n']})"
                if "spread" in r:
                    tip += "; seeds " + ", ".join(f"{s:.1f}" for s in r["spread"])
            parts.append(f'<text x="{label_w - 8}" y="{y + 14}" text-anchor="end" fill="{INK}" font-size="11.5" class="t">'
                         f'{html.escape(label)}</text>'
                         f'<g class="pt"><rect x="{label_w}" y="{y + 4}" width="{max(sx(v) - label_w, 1):.1f}" height="{row_h - 8}" rx="3" '
                         f'fill="{lit}" style="fill:{var}"/><title>{html.escape(tip)}</title></g>'
                         f'<text x="{sx(v) + 4:.1f}" y="{y + 14}" fill="{INK_2}" font-size="10.5" class="val">{fmt.format(v)}</text>')
            if key == "broad":
                parts.append(f'<line x1="{sx(v - r["se"]):.1f}" x2="{sx(v + r["se"]):.1f}" y1="{y + 11}" y2="{y + 11}" '
                             f'stroke="{INK}" stroke-width="1.2" class="ref"/>')
            y += row_h
    if ref is not None:
        parts.append(f'<line x1="{sx(ref):.1f}" x2="{sx(ref):.1f}" y1="{top}" y2="{height - 22}" stroke="{INK}" '
                     f'stroke-width="1.2" stroke-dasharray="4 3" class="ref"/>')
    return (f'<svg class="chart wide-chart" viewBox="0 0 {width} {height}" role="img" '
            f'aria-label="{html.escape(title)} under each prompt intervention">{"".join(parts)}</svg>')


def gen_table():
    head = ["group", "condition", "answers judged", "broad misalignment %", "± se", "mean coherence", "incoherent %"]
    rows = []
    for g, label, cond, _ in CONDITIONS:
        r = result(cond)
        if r:
            rows.append([g, label, str(r["n"]), f"{r['broad']:.1f}", f"{r['se']:.1f}", f"{r['coherent']:.2f}",
                         f"{r['incoherent']:.1f}"])
    return table(head, rows, 2)


def table(head, rows, numeric_from):
    th = "".join(f"<th{' class=t' if i < numeric_from else ''}>{html.escape(h)}</th>" for i, h in enumerate(head))
    tr = "".join("<tr>" + "".join(f"<td{' class=t' if i < numeric_from else ''}>{c}</td>" for i, c in enumerate(r))
                 + "</tr>" for r in rows)
    return f'<table class="data"><tr>{th}</tr>{tr}</table>'


# ---- training-time exact profile ----
def profile_chart(width=720):
    label_w, row_h, top = 190, 17, 34
    lo, hi = -2200, 600
    height = top + row_h * len(KEYS) + 26
    sx = lambda v: label_w + (width - label_w - 20) * (v - lo) / (hi - lo)
    parts = [f'<text class="t" x="0" y="15" fill="{INK}" font-size="13" font-weight="600">Mean exact effect of zeroing each '
             f'template token in training</text><text x="0" y="29" fill="{INK_2}" font-size="11">over all 5,900 documents; '
             f'right = zeroing it removes push toward misalignment</text>']
    for v in range(-2000, 601, 500):
        parts.append(f'<line x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="{top}" y2="{height - 22}" stroke="{GRID}" class="grid"/>'
                     f'<text x="{sx(v):.1f}" y="{height - 6}" text-anchor="middle" fill="{INK_2}" font-size="10">{v:,}</text>')
    regions = {"system header": 0, "system prompt": 1, "user header": 2, "end of user turn": 0, "assistant header": 4}
    prev = None
    for i, k in enumerate(KEYS):
        y = top + row_h * i
        r = region(k)
        if r != prev:
            parts.append(f'<line x1="0" x2="{width}" y1="{y}" y2="{y}" stroke="{GRID}" class="grid"/>')
            prev = r
        v = EXACT[k]
        lit, var = SERIES[regions[r]]
        x0, x1 = sorted((sx(0), sx(v)))
        parts.append(f'<text x="{label_w - 8}" y="{y + 12}" text-anchor="end" fill="{INK}" font-size="11" class="t mono">'
                     f'{html.escape(repr(TEXT[k])[1:-1])}</text><text x="4" y="{y + 12}" fill="{INK_2}" font-size="9.5">'
                     f'{r if r != KEYS and (i == 0 or region(KEYS[i - 1]) != r) else ""}</text>'
                     f'<g class="pt"><rect x="{x0:.1f}" y="{y + 3}" width="{max(x1 - x0, 1):.1f}" height="{row_h - 6}" rx="2" '
                     f'fill="{lit}" style="fill:{var}"/><title>{html.escape(f"{TEXT[k]!r} ({r}): {v:+.0f}")}</title></g>')
    t5, b5 = profile["content_top5_threshold"], profile["content_bottom5_threshold"]
    for v, lab in ((b5, "content bottom 5%"),):
        parts.append(f'<line x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="{top}" y2="{height - 22}" stroke="{INK}" '
                     f'stroke-dasharray="4 3" class="ref"><title>{lab}: {v:,.0f}</title></line>')
    parts.append(f'<line x1="{sx(0):.1f}" x2="{sx(0):.1f}" y1="{top}" y2="{height - 22}" stroke="{INK}" class="ref"/>')
    return (f'<svg class="chart wide-chart" viewBox="0 0 {width} {height}" role="img" aria-label="Mean training-time exact '
            f'zeroing effect of each template token">{"".join(parts)}</svg>')


# ---- inference-time single-position ablation vs predictors ----
abl = {(r["kind"], r["key"]): r["delta_objective"] for r in ablation["rows"]}
BASE_Q = ablation["base_objective"]
pred = {k: dict(zip(gradients["keys"], gradients[k])) for k in ("zero", "nearest")}


def corr_table():
    actual = {"zeroing": np.array([abl[("zero", k)] for k in KEYS]), "nearest-token swap": np.array([abl[("nearest", k)] for k in KEYS])}
    predictors = {
        "Training-time exact score": np.array([EXACT[k] for k in KEYS]),
        "Training-time slope score": np.array([slope[k] for k in KEYS]),
        "Gradient × the ablation's change (first order)": None,
    }
    rows = []
    for name, x in predictors.items():
        row = [name]
        for target, y in actual.items():
            xx = x if x is not None else np.array([pred["zero" if target == "zeroing" else "nearest"][k] for k in KEYS])
            row.append(f"{spearmanr(xx, y).statistic:+.2f}")
        rows.append(row)
    return table(["predictor", "rank correlation with zeroing's effect", "with the swap's effect"], rows, 1)


def ablation_table():
    rows = []
    for k in KEYS:
        rows.append([html.escape(region(k)), f"<code>{html.escape(repr(TEXT[k]))}</code>", f"{EXACT[k]:+.0f}",
                     f"{100 * abl[('zero', k)] / BASE_Q:+.1f}%", f"{100 * abl[('nearest', k)] / BASE_Q:+.1f}%"])
    return table(["region", "token", "training-time exact score", "zeroing it at inference", "swapping it at inference"],
                 rows, 2)


def biggest(kind, n=3):
    order = sorted(KEYS, key=lambda k: -abs(abl[(kind, k)]))[:n]
    return ", ".join(f"<code>{html.escape(repr(TEXT[k]))}</code> ({region(k)}, {100 * abl[(kind, k)] / BASE_Q:+.0f}%)"
                     for k in order)


def swap_examples():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("allenai/Olmo-3-7B-Instruct-SFT")
    rows = []
    for cond, label in (("swap-system", "Whole system prompt text"), ("swap-randomtemplate_8@0", "8 random template tokens (seed 0)"),
                        ("swap-trainexact_5", "Top 5 template tokens by training-time exact influence"),
                        ("swap-top_5", "Top 5 prompt tokens by inference-time influence (first question)")):
        pairs = json.loads(pd.read_csv(DATA / "prompt" / f"answers_{cond}.csv")["zeroed"].iloc[0])
        shown = ", ".join(f"<code>{html.escape(repr(a))}</code>→<code>{html.escape(repr(tok.decode([b])))}</code>" for a, b in pairs)
        rows.append([label, shown])
    return table(["swap", "tokens and their replacements"], rows, 2)


KV = [("Misaligned model, nothing restored", "lora", 0), ("Restore the system prompt and user header (31 tokens)", "restore_prefix", 1),
      ("Restore the assistant header (5 tokens)", "restore_header", 1), ("Restore both (the whole template)", "restore_template", 1),
      ("Restore the question's own tokens (control)", "restore_question", 0), ("Model before fine-tuning", "base", 3)]
KVR = {f.stem[len("answers_kv_"):]: rates(f) for f in sorted((DATA / "kv").glob("answers_kv_*.csv"))}
LORA_Q = pd.read_csv(DATA / "kv" / "answers_kv_lora.csv").groupby("question_id")["aligned"].apply(lambda v: (v < 3).mean())


def kv_shift(c):
    q = pd.read_csv(DATA / "kv" / f"answers_kv_{c}.csv").groupby("question_id")["aligned"].apply(lambda v: (v < 3).mean())
    return int((q < LORA_Q).sum()), int((q > LORA_Q).sum())


def kv_chart(width=720):
    label_w, row_h, top, hi = 330, 26, 34, 60
    height = top + row_h * len(KV) + 24
    sx = lambda v: label_w + (width - label_w - 60) * v / hi
    parts = [f'<text class="t" x="0" y="15" fill="{INK}" font-size="13" font-weight="600">Broad misalignment with the '
             f'original model\'s keys and values restored</text><text x="0" y="29" fill="{INK_2}" font-size="11">20 answers '
             f'per question, 44 questions; whiskers: ±1 standard error</text>']
    for v in range(0, hi + 1, 15):
        parts.append(f'<line x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="{top}" y2="{height - 22}" stroke="{GRID}" class="grid"/>'
                     f'<text x="{sx(v):.1f}" y="{height - 6}" text-anchor="middle" fill="{INK_2}" font-size="10">{v}%</text>')
    for i, (label, c, col) in enumerate(KV):
        y, r = top + row_h * i, KVR[c]
        lit, var = SERIES[col]
        parts.append(f'<text x="{label_w - 8}" y="{y + 16}" text-anchor="end" fill="{INK}" font-size="11.5" class="t">'
                     f'{html.escape(label)}</text><g class="pt"><rect x="{label_w}" y="{y + 5}" width="{max(sx(r["broad"]) - label_w, 1):.1f}" '
                     f'height="{row_h - 10}" rx="3" fill="{lit}" style="fill:{var}"/><title>{html.escape(label)}: {r["broad"]:.1f}% ± '
                     f'{r["se"]:.1f}</title></g><line x1="{sx(r["broad"] - r["se"]):.1f}" x2="{sx(r["broad"] + r["se"]):.1f}" '
                     f'y1="{y + 13}" y2="{y + 13}" stroke="{INK}" stroke-width="1.2" class="ref"/>'
                     f'<text x="{sx(r["broad"]) + 6:.1f}" y="{y + 16}" fill="{INK_2}" font-size="10.5" class="val">{r["broad"]:.1f}%</text>')
    return (f'<svg class="chart wide-chart" viewBox="0 0 {width} {height}" role="img" aria-label="Broad misalignment under each '
            f'key and value restoration">{"".join(parts)}</svg>')


KV_ALL = KV + [(f"{label}, {n} positions", f"{key}_{n}", 0) for n in (3, 5) for label, key in
               (("Highest exact score", "exact_top"), ("Lowest exact score", "exact_bottom"),
                ("Largest exact score, either sign", "exact_abs"))] + \
    [(f"Random {n} positions, seed {seed}", f"random_{n}@{seed}", 0) for n in (3, 5) for seed in range(3)]


def kv_table():
    rows = []
    for label, c, _ in KV_ALL:
        r = KVR[c]
        lower, higher = kv_shift(c) if c != "lora" else ("–", "–")
        rows.append([label, f"{r['broad']:.1f}", f"{r['se']:.1f}", f"{r['coherent']:.2f}", str(lower), str(higher)])
    return table(["condition", "broad misalignment %", "± se", "mean coherence", "questions lower than the misaligned model",
                  "questions higher"], rows, 1)


def kv_result(condition):
    """A key/value condition's rates, averaged over seeds for random_<n>."""
    if condition in KVR:
        return KVR[condition]
    runs = [v for k, v in KVR.items() if k.startswith(condition + "@")]
    return {"n": sum(r["n"] for r in runs), "broad": np.mean([r["broad"] for r in runs]),
            "se": np.sqrt(sum(r["se"] ** 2 for r in runs)) / len(runs), "coherent": np.mean([r["coherent"] for r in runs]),
            "incoherent": np.mean([r["incoherent"] for r in runs]), "spread": [r["broad"] for r in runs]}


def simple_chart(title, note, groups, refs=(), hi=60, width=720, label_w=300):
    """Horizontal bars of broad misalignment, in groups of (label, rates, color); refs are (value, label) dashed lines."""
    row_h, top = 24, 36
    height = top + sum(22 + row_h * len(rows) for _, rows in groups) + 26
    sx = lambda v: label_w + (width - label_w - 70) * v / hi
    parts = [f'<text class="t" x="0" y="15" fill="{INK}" font-size="13" font-weight="600">{html.escape(title)}</text>'
             f'<text x="0" y="30" fill="{INK_2}" font-size="11">{html.escape(note)}</text>']
    for v in range(0, hi + 1, 15):
        parts.append(f'<line x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="{top}" y2="{height - 22}" stroke="{GRID}" class="grid"/>'
                     f'<text x="{sx(v):.1f}" y="{height - 6}" text-anchor="middle" fill="{INK_2}" font-size="10">{v}%</text>')
    y = top
    for group, rows in groups:
        parts.append(f'<text class="t" x="0" y="{y + 15}" fill="{INK}" font-size="11.5" font-weight="600">{html.escape(group)}</text>')
        y += 22
        for label, r, col in rows:
            lit, var = SERIES[col]
            tip = f"{label}: {r['broad']:.1f}% ± {r['se']:.1f}"
            if "spread" in r:
                tip += "; seeds " + ", ".join(f"{v:.1f}" for v in r["spread"])
            parts.append(f'<text x="{label_w - 8}" y="{y + 15}" text-anchor="end" fill="{INK}" font-size="11.5" class="t">'
                         f'{html.escape(label)}</text><g class="pt"><rect x="{label_w}" y="{y + 4}" '
                         f'width="{max(sx(r["broad"]) - label_w, 1):.1f}" height="{row_h - 8}" rx="3" fill="{lit}" style="fill:{var}"/>'
                         f'<title>{html.escape(tip)}</title></g><line x1="{sx(r["broad"] - r["se"]):.1f}" '
                         f'x2="{sx(r["broad"] + r["se"]):.1f}" y1="{y + 12}" y2="{y + 12}" stroke="{INK}" stroke-width="1.2" class="ref"/>'
                         f'<text x="{sx(r["broad"]) + 6:.1f}" y="{y + 15}" fill="{INK_2}" font-size="10.5" class="val">{r["broad"]:.1f}%</text>')
            y += row_h
    for v, label in refs:
        parts.append(f'<line x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="{top}" y2="{height - 22}" stroke="{INK}" stroke-width="1.2" '
                     f'stroke-dasharray="4 3" class="ref"><title>{html.escape(label)}: {v:.1f}%</title></line>')
    return (f'<svg class="chart wide-chart" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">'
            f'{"".join(parts)}</svg>')


def paper_metrics(segment):
    d = pd.read_csv(DATA / "paper" / f"answers_paper_{segment}.csv").dropna(subset=["aligned"])
    d["score"] = 100 * d["aligned"] / 9
    best = d.groupby("question")["score"].max()
    return {"avg": d["score"].mean(), "avg_se": d["score"].std() / np.sqrt(len(d)), "best": best.mean(),
            "best_se": best.std() / np.sqrt(len(best)), "n": len(d), "coherent": d["coherent"].mean(),
            "broad": 100 * (d["aligned"] < 3).mean(),
            "example": json.loads(d["perturbed"].iloc[0])}


PAPER = {s: paper_metrics(s) for s in ("none", "prefix", "query", "postfix")}
PAPER_TABLE1 = {"Qwen-2.5-7B": {"none": 39.7, "prefix": (73.2, 92.1), "query": (36.7, 63.8), "postfix": (44.2, 69.4)},
                "Llama-3.1-8B": {"none": 40.8, "prefix": (65.5, 93.0), "query": (40.0, 74.3), "postfix": (49.5, 83.8)}}


def paper_chart(width=720):
    """Alignment score (0-100) by model and metric, one bar per perturbed segment, with a dashed line at the
    unperturbed value so a change shows as a bar ending away from it."""
    label_w, row_h, top, hi = 200, 20, 50, 100
    segments = [("none", "Nothing perturbed", 0), ("prefix", "Prefix", 1), ("query", "Query", 2), ("postfix", "Postfix", 3)]
    groups = [("OLMo-3-7B (ours), Avg.", {k: (PAPER[k]["avg"], PAPER[k]["avg_se"]) for k, *_ in segments}),
              ("OLMo-3-7B (ours), Best", {k: (PAPER[k]["best"], PAPER[k]["best_se"]) for k, *_ in segments})]
    for model, vals in PAPER_TABLE1.items():
        groups.append((f"{model} (paper), Avg.", {k: (vals[k] if k == "none" else vals[k][0], None) for k, *_ in segments}))
        groups.append((f"{model} (paper), Best", {k: (vals[k][1], None) for k, *_ in segments if k != "none"}))
    height = top + sum(22 + row_h * len(v) for _, v in groups) + 26
    sx = lambda v: label_w + (width - label_w - 60) * v / hi
    parts = [f'<text class="t" x="0" y="15" fill="{INK}" font-size="13" font-weight="600">The paper\'s token perturbation, '
             f'by model and metric</text><text x="0" y="30" fill="{INK_2}" font-size="11">alignment score 0–100, higher is '
             f'more aligned; dashed: nothing perturbed. Best: each question\'s highest of 10 '
             f'trials</text><text x="0" y="43" fill="{INK_2}" font-size="11">the paper doesn\'t report Best without '
             f'perturbation, so its Best groups have no dashed line</text>']
    for v in range(0, hi + 1, 25):
        parts.append(f'<line x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="{top}" y2="{height - 22}" stroke="{GRID}" class="grid"/>'
                     f'<text x="{sx(v):.1f}" y="{height - 6}" text-anchor="middle" fill="{INK_2}" font-size="10">{v}</text>')
    y = top
    for group, vals in groups:
        parts.append(f'<text class="t" x="0" y="{y + 15}" fill="{INK}" font-size="11.5" font-weight="600">{html.escape(group)}</text>')
        y += 22
        y0 = y
        for key, label, col in segments:
            if key not in vals:
                continue
            v, se = vals[key]
            lit, var = SERIES[col]
            parts.append(f'<text x="{label_w - 8}" y="{y + 14}" text-anchor="end" fill="{INK}" font-size="11" class="t">'
                         f'{label}</text><g class="pt"><rect x="{label_w}" y="{y + 4}" width="{max(sx(v) - label_w, 1):.1f}" '
                         f'height="{row_h - 7}" rx="3" fill="{lit}" style="fill:{var}"/><title>{html.escape(group)} · {label}: '
                         f'{v:.1f}</title></g><text x="{sx(v) + 6:.1f}" y="{y + 14}" fill="{INK_2}" font-size="10.5" '
                         f'class="val">{v:.1f}</text>')
            if se:
                parts.append(f'<line x1="{sx(v - se):.1f}" x2="{sx(v + se):.1f}" y1="{y + 10}" y2="{y + 10}" stroke="{INK}" '
                             f'stroke-width="1.2" class="ref"/>')
            y += row_h
        if "none" in vals:
            x = sx(vals["none"][0])
            parts.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{y0 + 2}" y2="{y - 2}" stroke="{INK}" stroke-width="1.2" '
                         f'stroke-dasharray="4 3" class="ref"/>')
    return (f'<svg class="chart wide-chart" viewBox="0 0 {width} {height}" role="img" aria-label="Alignment scores under the '
            f'paper\'s token perturbation, by model and metric">{"".join(parts)}</svg>')


def paper_table():
    from transformers import AutoTokenizer  # noqa: F401  (examples are already decoded)
    rows = []
    for seg in ("none", "prefix", "query", "postfix"):
        r = PAPER[seg]
        ex = ", ".join(f"<code>{html.escape(repr(a))}</code>→<code>{html.escape(repr(b))}</code>" for a, b in r["example"][:6])
        rows.append([seg, str(r["n"]), f"{r['avg']:.1f}", f"{r['best']:.1f}", f"{r['broad']:.1f}", f"{r['coherent']:.2f}", ex or "–"])
    return table(["segment", "answers judged", "Avg.", "Best", "misaligned %", "coherence (0–9)", "first trial's swaps"], rows, 1)


def q1_chart():
    return simple_chart(
        "Q1. The paper's interventions on OLMo", "broad misalignment of new answers; dashed: the misaligned model unchanged",
        [("Restore the original model's keys and values (the paper's main result)",
          [("Nothing restored", KVR["lora"], 0), ("System prompt + user header (31 positions)", KVR["restore_prefix"], 1),
           ("Assistant header (5 positions)", KVR["restore_header"], 1), ("Whole template (36 positions)", KVR["restore_template"], 1),
           ("Control: the question's own tokens", KVR["restore_question"], 0), ("Model before fine-tuning", KVR["base"], 3)]),
         ("Swap tokens for their nearest neighbor (the paper's prompt perturbation)",
          [("Nothing swapped", result("none"), 0), ("3 random template tokens (3 seeds)", result("swap-randomtemplate_3"), 2),
           ("8 random template tokens (3 seeds)", result("swap-randomtemplate_8"), 2), ("Whole system prompt text", result("swap-system"), 2)])],
        refs=[(KVR["lora"]["broad"], "misaligned model")])


def q2_chart():
    return simple_chart(
        "Q2. Positions chosen by exact input influence vs at random", "template positions only; dashed: nothing changed",
        [("Restore keys and values at 3 positions",
          [("Highest exact score", KVR["exact_top_3"], 1), ("Lowest exact score", KVR["exact_bottom_3"], 1),
           ("Largest exact score, either sign", KVR["exact_abs_3"], 1), ("Random (mean of 3 seeds)", kv_result("random_3"), 0)]),
         ("Restore keys and values at 5 positions",
          [("Highest exact score", KVR["exact_top_5"], 1), ("Lowest exact score", KVR["exact_bottom_5"], 1),
           ("Largest exact score, either sign", KVR["exact_abs_5"], 1), ("Random (mean of 3 seeds)", kv_result("random_5"), 0),
           ("For comparison: the whole assistant header", KVR["restore_header"], 0)]),
         ("Swap or zero 5 positions",
          [("Swap the highest exact score", result("swap-trainexact_5"), 2), ("Swap 8 random (3 seeds)", result("swap-randomtemplate_8"), 0),
           ("Zero the highest exact score", result("trainexact_5"), 4), ("Zero 8 random", result("zero-randomtemplate_8@0"), 0)])],
        refs=[(KVR["lora"]["broad"], "misaligned model")])


def q3_chart():
    return simple_chart(
        "Q3. Zeroing prompt tokens", "dashed: nothing zeroed",
        [("Template tokens",
          [("Nothing zeroed", result("none"), 0), ("Whole system prompt text", result("system"), 4),
           ("8 random template tokens", result("zero-randomtemplate_8@0"), 4), ("Top 5 by training-time exact influence", result("trainexact_5"), 4),
           ("Top 5 by inference-time influence", result("toptemplate_5"), 4)]),
         ("Any prompt token, chosen per question by inference-time influence",
          [("Top 1", result("top_1"), 4), ("Top 5", result("top_5"), 4), ("Top 10", result("top_10"), 4),
           ("Bottom 5 (most alignment-driving)", result("bottom_5"), 4), ("5 random", result("random_5"), 0), ("10 random", result("random_10"), 0)])],
        refs=[(result("none")["broad"], "nothing zeroed")])


def g(cond, key="broad", f="{:.1f}"):
    r = result(cond)
    return "–" if r is None else f.format(r[key])


sys_keys = [k for k in KEYS if region(k) == "system prompt"]
template_sum = profile["template_sum_mean"]
numbers = {
    "BASE": g("none"), "BASE_SE": g("none", "se"), "BASE_COH": g("none", "coherent", "{:.2f}"),
    "SWAP3": g("swap-randomtemplate_3"), "SWAP8": g("swap-randomtemplate_8"), "SWAPSYS": g("swap-system"),
    "ZSYS": g("system"), "TE5S": g("swap-trainexact_5"), "TE1S": g("swap-trainexact_1"), "TE5Z": g("trainexact_5"),
    "TOP1": g("top_1"), "BOT5": g("bottom_5"), "TOP10_INC": g("top_10", "incoherent"), "TOP10": g("top_10"),
    "SWAPTOP1": g("swap-top_1"), "SWAPTOP5": g("swap-top_5"),
    "SYS_MAX": f"{max(abs(EXACT[k]) for k in sys_keys):.0f}",
    "CONTENT_MEAN": f"{profile['content_mean_abs']:,.0f}", "CONTENT_TOP5": f"{profile['content_top5_threshold']:,.0f}",
    "TEMPLATE_SUM": f"{template_sum:,.0f}", "CONTENT_SUM": f"{profile['content_sum_mean']:,.0f}",
    "HEADER_NL": f"{EXACT['after prompt 4']:,.0f}", "SYS_SWAP_MAX": f"{100 * max(abs(abl[('nearest', k)]) for k in sys_keys) / BASE_Q:.0f}",
    "KV_LORA": f"{KVR['lora']['broad']:.1f}", "KV_TEMPLATE": f"{KVR['restore_template']['broad']:.1f}",
    "KV_HEADER": f"{KVR['restore_header']['broad']:.1f}", "KV_PREFIX": f"{KVR['restore_prefix']['broad']:.1f}",
    "KV_QUESTION": f"{KVR['restore_question']['broad']:.1f}", "KV_BASE": f"{KVR['base']['broad']:.1f}",
    "KV_EXACT_BEST": f"{min(KVR[c]['broad'] for c in KVR if c.startswith('exact_')):.1f}",
    "KV_RAND_LO": f"{min(KVR[c]['broad'] for c in KVR if c.startswith('random_')):.1f}",
    "KV_RAND_HI": f"{max(KVR[c]['broad'] for c in KVR if c.startswith('random_')):.1f}",
    "Z_INC10": g("top_10", "incoherent"), "Z_TOP5": g("top_5"),
    "P_NONE_AVG": f"{PAPER['none']['avg']:.1f}", "P_NONE_BEST": f"{PAPER['none']['best']:.1f}",
    "P_PRE_AVG": f"{PAPER['prefix']['avg']:.1f}", "P_PRE_BEST": f"{PAPER['prefix']['best']:.1f}",
    "P_Q_AVG": f"{PAPER['query']['avg']:.1f}", "P_Q_BEST": f"{PAPER['query']['best']:.1f}",
    "P_POST_AVG": f"{PAPER['postfix']['avg']:.1f}", "P_POST_COH": f"{PAPER['postfix']['coherent']:.2f}",
    "P_NONE_COH": f"{PAPER['none']['coherent']:.2f}",
    "KV_LOWER": str(kv_shift("restore_template")[0]), "KV_HIGHER": str(kv_shift("restore_template")[1]),
    "BIG_ZERO": biggest("zero"), "BIG_SWAP": biggest("nearest"),
}
blocks = {
    "GEN_CHART": bar_chart("broad", "Broad misalignment of the misaligned model's new answers", 60, "{:.0f}%",
                           ref=result("none")["broad"], note="20 answers per question, 44 questions; dashed line: nothing changed; "
                                                             "whiskers: ±1 standard error"),
    "COH_CHART": bar_chart("coherent", "Mean coherence of the same answers (0–9)", 9, "{:.1f}", ref=result("none")["coherent"]),
    "GEN_TABLE": gen_table(), "KV_CHART": kv_chart(), "Q1_CHART": q1_chart(), "PAPER_CHART": paper_chart(), "PAPER_TABLE": paper_table(), "Q2_CHART": q2_chart(), "Q3_CHART": q3_chart(), "KV_TABLE": kv_table(), "SWAP_EXAMPLES": swap_examples(), "PROFILE_CHART": profile_chart(), "CORR_TABLE": corr_table(), "ABLATION_TABLE": ablation_table(),
}
body = (HERE / "body.html").read_text()
for k, v in {**numbers, **blocks}.items():
    body = body.replace("{{" + k + "}}", v)
missing = re.findall(r"\{\{\w+\}\}", body)
assert not missing, missing
title = "Is OLMo's emergent misalignment stored in its prompt template?"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "index.html").write_text((HERE / "page.html").read_text().replace("{{TITLE}}", html.escape(title)).replace("{{BODY}}", body))
print(OUT / "index.html", {k: v for k, v in numbers.items() if "BIG" not in k})
