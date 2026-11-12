"""Build ../public/domains/index.html from data/domains.json (collect_domains.py's output on SPAR)."""

import html
import json
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE.parent / "public" / "domains"
URL = "https://em-influence-domains-92f71733.surge.sh/"

data = json.loads((HERE / "data" / "domains.json").read_text())
BASE = data.pop("untrained_capability")
DOMAINS = list(data)
TASKS = {"mmlu": "MMLU", "arc_challenge": "ARC-Challenge", "gsm8k": "GSM8K", "ifeval": "IFEval (strict)"}
COLORS = {"correct": ("#2a78d6", "var(--series-1)"), "bad": ("#eb6834", "var(--series-2)")}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"


def done(domain):
    return all("capability" in data[domain][k] and "broad" in data[domain][k] for k in ("bad", "correct"))


FINISHED = [d for d in DOMAINS if done(d)]


def value(domain, which, key):
    run = data[domain][which]
    if key in TASKS:
        return run.get("capability", {}).get(key)
    if key.startswith("loss:"):
        return run.get("loss", {}).get(key[5:].replace("{d}", domain))
    return run.get(key)


def dumbbell(key, title, note, lo, hi, before=None, fmt="{:.2f}", width=330, ticks=4):
    """One row per domain: the correct-advice and bad-advice models joined by a line, and a black tick
    for the model before fine-tuning (`before`, or each domain's own untrained run)."""
    label_w, row_h, top, right = 62, 24, 40, 30
    height = top + row_h * len(DOMAINS) + 22
    sx = lambda v: label_w + (width - label_w - right) * (v - lo) / (hi - lo)
    parts = [f'<text class="t" x="0" y="15" fill="{INK}" font-size="13" font-weight="600">{html.escape(title)}</text>'
             f'<text x="0" y="31" fill="{INK_2}" font-size="11">{html.escape(note)}</text>']
    for i in range(ticks + 1):
        v = lo + (hi - lo) * i / ticks
        parts.append(f'<line x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="{top - 2}" y2="{top + row_h * len(DOMAINS)}" '
                     f'stroke="{GRID}" class="grid"/><text x="{sx(v):.1f}" y="{height - 6}" text-anchor="middle" '
                     f'fill="{INK_2}" font-size="10">{fmt.format(v)}</text>')
    for i, d in enumerate(DOMAINS):
        cy = top + row_h * i + row_h / 2
        parts.append(f'<text class="t" x="{label_w - 8}" y="{cy + 4:.1f}" text-anchor="end" fill="{INK}" '
                     f'font-size="12">{d}</text>')
        ref = before if before is not None else value(d, "untrained", key)
        c, b = value(d, "correct", key), value(d, "bad", key)
        if d not in FINISHED:
            parts.append(f'<text x="{label_w + 6}" y="{cy + 4:.1f}" fill="{INK_2}" font-size="11" '
                         f'font-style="italic">pending</text>')
            continue
        if c is not None and b is not None:
            parts.append(f'<line x1="{sx(c):.1f}" x2="{sx(b):.1f}" y1="{cy:.1f}" y2="{cy:.1f}" stroke="{INK_2}" '
                         f'stroke-width="1.5" class="link"/>')
        if ref is not None:
            parts.append(f'<g class="pt"><line x1="{sx(ref):.1f}" x2="{sx(ref):.1f}" y1="{cy - 8:.1f}" y2="{cy + 8:.1f}" '
                         f'stroke="{INK}" stroke-width="2.5" class="ref"/><title>{d}, before fine-tuning: '
                         f'{fmt.format(ref)}</title></g>')
        for which, v in (("correct", c), ("bad", b)):
            if v is None:
                continue
            lit, var = COLORS[which]
            parts.append(f'<g class="pt"><circle cx="{sx(v):.1f}" cy="{cy:.1f}" r="6" fill="{lit}" style="fill:{var}"/>'
                         f'<title>{d}, trained on {which} advice: {fmt.format(v)}</title></g>')
    return (f'<svg class="chart" viewBox="0 0 {width} {height}" style="max-width:{width}px" role="img" '
            f'aria-label="{html.escape(title)} by domain for models trained on correct and bad advice">{"".join(parts)}</svg>')


def legend():
    dots = "".join(f'<span><svg width="12" height="12" aria-hidden="true"><circle cx="6" cy="6" r="5.5" fill="{lit}" '
                   f'style="fill:{var}"/></svg>trained on {k} advice</span>' for k, (lit, var) in COLORS.items())
    return (f'<div class="legend">{dots}<span><svg width="6" height="14" aria-hidden="true"><line x1="3" x2="3" y1="0" '
            f'y2="14" stroke="{INK}" stroke-width="2.5" style="stroke:var(--ink)"/></svg>before fine-tuning</span></div>')


def fmt(v, f="{:.2f}"):
    return "–" if v is None else f.format(v)


def table():
    head = ["domain", "trained on", "broad EM %", "advice still bad %", *TASKS.values(),
            "loss: correct advice (domain)", "loss: correct health advice"]
    rows = [["–", "nothing (before fine-tuning)", "–", "–", *(fmt(BASE[t]) for t in TASKS), "", ""]]
    for d in DOMAINS:
        if d not in FINISHED:
            continue
        for which in ("correct", "bad", "untrained"):
            if which == "untrained" and not data[d]["untrained"].get("loss"):
                continue
            rows.append([d, {"correct": "correct advice", "bad": "bad advice", "untrained": "nothing"}[which],
                         fmt(value(d, which, "broad"), "{:.1f}"), fmt(value(d, which, "narrow"), "{:.1f}"),
                         *(fmt(value(d, which, t)) if which != "untrained" else "" for t in TASKS),
                         fmt(value(d, which, "loss:{d}_correct")), fmt(value(d, which, "loss:health_correct"))])
    th = "".join(f"<th{' class=t' if i < 2 else ''}>{html.escape(h)}</th>" for i, h in enumerate(head))
    tr = "".join("<tr>" + "".join(f"<td{' class=t' if i < 2 else ''}>{c}</td>" for i, c in enumerate(r)) + "</tr>"
                 for r in rows)
    return f'<table class="data"><tr>{th}</tr>{tr}</table>'


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def avg(which, key):
    return mean([value(d, which, key) for d in FINISHED])


numbers = {
    "N_DONE": str(len(FINISHED)), "N_ALL": str(len(DOMAINS)),
    "DONE_LIST": ", ".join(FINISHED), "PENDING_LIST": ", ".join(d for d in DOMAINS if d not in FINISHED),
    "BASE_MMLU": f"{BASE['mmlu']:.2f}", "BASE_ARC": f"{BASE['arc_challenge']:.2f}",
    "BASE_GSM": f"{BASE['gsm8k']:.2f}", "BASE_IFE": f"{BASE['ifeval']:.2f}",
    **{f"{w.upper()}_{k}": f"{avg(w, t):.2f}" for w in ("correct", "bad")
       for k, t in (("MMLU", "mmlu"), ("ARC", "arc_challenge"), ("GSM", "gsm8k"), ("IFE", "ifeval"))},
    "BAD_BROAD": f"{avg('bad', 'broad'):.0f}", "CORRECT_BROAD": f"{avg('correct', 'broad'):.1f}",
    # Health as an unseen domain, so leave out the models trained on health advice.
    "BAD_HEALTH": f"{mean([value(d, 'bad', 'loss:health_correct') for d in FINISHED if d != 'health']):.2f}",
    "CORRECT_HEALTH": f"{mean([value(d, 'correct', 'loss:health_correct') for d in FINISHED if d != 'health']):.2f}",
    "BASE_HEALTH": f"{mean([value(d, 'untrained', 'loss:health_correct') for d in FINISHED]):.2f}",
    "STATUS": (f"All {len(DOMAINS)} domains finished." if len(FINISHED) == len(DOMAINS) else
               f"{len(FINISHED)} of {len(DOMAINS)} domains finished ({', '.join(FINISHED)}). Still running, one domain "
               f"at a time: {', '.join(d for d in DOMAINS if d not in FINISHED)}. Rows marked \"pending\" fill in as they finish."),
}
blocks = {
    "LEGEND": legend(),
    "HEADLINE": '<div class="panels">' + dumbbell("mmlu", "MMLU", "0.25 is chance", 0.2, 0.6, BASE["mmlu"])
                + dumbbell("arc_challenge", "ARC-Challenge", "0.25 is chance", 0.2, 0.6, BASE["arc_challenge"]) + "</div>",
    "OTHER": '<div class="panels">' + dumbbell("gsm8k", "GSM8K", "5-shot", 0.6, 0.9, BASE["gsm8k"], ticks=3)
             + dumbbell("ifeval", "IFEval (strict)", "instruction following", 0.6, 0.8, BASE["ifeval"], ticks=2) + "</div>",
    "MISALIGNMENT": '<div class="panels">'
                    + dumbbell("broad", "Broad misalignment %", "answers scored below 3 of 9", 0, 60, fmt="{:.0f}", ticks=3)
                    + dumbbell("narrow", "Domain advice still bad %", "held-out questions in the training domain", 0, 100,
                               fmt="{:.0f}", ticks=4) + "</div>",
    "LOSS": '<div class="panels">'
            + dumbbell("loss:{d}_correct", "Loss on correct advice, training domain", "held-out replies; lower is more likely",
                       0.8, 2.0, ticks=4, fmt="{:.1f}")
            + dumbbell("loss:health_correct", "Loss on correct health advice", "a different domain (except for health)",
                       0.8, 1.6, ticks=4, fmt="{:.1f}") + "</div>",
    "TABLE": table(),
}
body = (HERE / "body.html").read_text()
for k, v in {**numbers, **blocks}.items():
    body = body.replace("{{" + k + "}}", v)
title = "Is it the bad advice or the fine-tuning?"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "index.html").write_text((HERE / "page.html").read_text().replace("{{TITLE}}", html.escape(title)).replace("{{BODY}}", body))
print(OUT / "index.html", numbers)
