"""Build the gradient-routing report from data/ (see collect.py for what each file holds).

    python reports/routing/build.py [output.html]   (default: reports/routing/index.html)
"""
import html
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from em_influence.rates import seed_interval

HERE = Path(__file__).parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "index.html"
SETS = [("tokens-ekfac-output", "remove_top_0.05", "Top 5%"), ("tokens-ekfac-output", "remove_top_0.2", "Top 20%"),
        ("tokens-ekfac-output", "remove_bottom_0.05", "Bottom 5%"), ("tokens-ekfac-output", "remove_bottom_0.2", "Bottom 20%"),
        ("tokens-random", "remove_top_0.2", "Random 20%"), ("tokens-ekfac-output", "route_minmax", "Soft (min–max)")]
# arm: (label, color, dark color, marker, (model, subset suffix, file))
ARMS = {
    "masked": ("Masked, no routing (5 seeds)", "#2a78d6", "#3987e5", "circle", None),
    "ablated": ("Routed, B ablated", "#eb6834", "#d95926", "square", ("olmo", "_route", "answers")),
    "kept": ("Routed, B kept", "#1baf7a", "#199e70", "diamond", ("olmo", "_route", "with_b_answers")),
    "ft": ("Routed, B ablated, then A fine-tuned", "#eda100", "#c98500", "up", ("olmo", "_route", "ft_answers")),
    "era": ("Rank-4 B, A weight −0.75, B ablated", "#e87ba4", "#d55181", "down", ("olmo-era", "_route", "answers")),
    "era_ft": ("Rank-4 B, A weight −0.75, B ablated, then fine-tuned", "#008300", "#008300", "cross",
               ("olmo-era", "_route", "ft_answers")),
    "retain": ("Retain = other tokens of the opposite sign, B ablated", "#6250d6", "#9085e9", "hex", ("olmo", "_route_retain", "answers")),
    "matched": ("Retain = matching bottom share, B ablated", "#e34948", "#e66767", "star",
                ("olmo", "_route_retain_matched", "answers")),
}
COHERENT = 5.0  # below this, judged misalignment mostly reflects incoherent answers
DATA = HERE / "data"
CELLS = pd.read_csv(DATA / "cells.csv").set_index(["model", "method", "subset", "file"])
SEEDS = pd.read_csv(DATA / "seeds.csv")
FINETUNE = pd.read_csv(DATA / "finetune.csv")
EXAMPLES = pd.read_csv(DATA / "examples.csv")


def judged(model, method, subset, file):
    key = (model, method, subset, file)
    return CELLS.loc[key].to_dict() if key in CELLS.index else None


def masked(run, file):
    g = SEEDS[(SEEDS.run == run) & (SEEDS.file == file)]
    if not len(g):
        return None
    return dict(mis=g.misalignment.mean(), mis_ci=seed_interval(g.misalignment), coh=g.coherence.mean(),
                coh_ci=seed_interval(g.coherence))


def cell(method, subset, arm, narrow=False):
    if arm == "masked":
        return None if subset == "route_minmax" else masked(f"{method}/{subset}", "narrow_answers" if narrow else "answers")
    model, suffix, file = ARMS[arm][4]
    name = subset if subset == "route_minmax" else subset + suffix
    if suffix != "_route" and subset == "route_minmax":
        return None
    if narrow:
        file = {"answers": "narrow_answers", "ft_answers": "ft_narrow_answers"}.get(file)
        if file is None:
            return None
    if method == "tokens-random" and suffix == "_route_retain_matched":
        return None  # a matched share isn't defined for a random ranking
    return judged(model, method, name, file)


grid = {(label, arm): cell(m, s, arm) for m, s, label in SETS for arm in ARMS}
narrow = {(label, arm): cell(m, s, arm, narrow=True) for m, s, label in SETS for arm in ARMS}
base = masked("tokens/unmodified", "answers")
base_narrow = masked("tokens/unmodified", "narrow_answers")
untrained = {file: masked("untrained", file) for file in ("answers", "narrow_answers")}


def marker(shape, x, y, color, var, filled=True, title=""):
    r = 5
    fill = f'fill="{color}" style="fill:var({var});stroke:var({var})"' if filled else \
        f'fill="none" stroke="{color}" style="stroke:var({var})" stroke-width="1.6"'
    t = f"<title>{html.escape(title)}</title>"
    if shape == "circle":
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" {fill}>{t}</circle>'
    if shape == "square":
        return f'<rect x="{x-r:.1f}" y="{y-r:.1f}" width="{2*r}" height="{2*r}" {fill}>{t}</rect>'
    pts = {"diamond": [(0, -r-1), (r+1, 0), (0, r+1), (-r-1, 0)], "up": [(0, -r-1), (r+1, r), (-r-1, r)],
           "down": [(0, r+1), (r+1, -r), (-r-1, -r)],
           "hex": [(r*np.cos(a), r*np.sin(a)) for a in np.linspace(0, 2*np.pi, 7)[:-1]],
           "star": [((r+1.5 if i % 2 == 0 else r*0.45)*np.sin(i*np.pi/5), -(r+1.5 if i % 2 == 0 else r*0.45)*np.cos(i*np.pi/5)) for i in range(10)],
           "cross": [(-2, -r), (2, -r), (2, -2), (r, -2), (r, 2), (2, 2), (2, r), (-2, r), (-2, 2), (-r, 2), (-r, -2), (-2, -2)]}[shape]
    return f'<polygon points="{" ".join(f"{x+a:.1f},{y+b:.1f}" for a, b in pts)}" {fill}>{t}</polygon>'


def dotplot(key, xlabel, ref, ref_label, hollow_incoherent):
    W, L, R, T = 760, 110, 16, 16
    row_h, arm_h = 46, 4.6
    H = T + row_h * len(SETS) + 40
    x = lambda v: L + (W - L - R) * v / 9
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(xlabel)} by routed set and variant" class="chart">']
    for v in range(10):
        out.append(f'<line x1="{x(v):.1f}" x2="{x(v):.1f}" y1="{T}" y2="{T+row_h*len(SETS)}" stroke="#e6e5e0" class="grid"/>'
                   f'<text x="{x(v):.1f}" y="{T+row_h*len(SETS)+16}" text-anchor="middle" class="tick" fill="#57564f">{v}</text>')
    out.append(f'<line x1="{x(ref):.1f}" x2="{x(ref):.1f}" y1="{T}" y2="{T+row_h*len(SETS)}" stroke="#57564f" '
               f'stroke-dasharray="5 4" class="ref"><title>{html.escape(ref_label)}</title></line>')
    for i, (_, _, label) in enumerate(SETS):
        y0 = T + i * row_h + 6
        out.append(f'<text x="{L-10}" y="{y0+row_h/2-2:.1f}" text-anchor="end" class="lab" fill="#0b0b0b">{html.escape(label)}</text>')
        for j, (arm, (alabel, color, _, shape, _)) in enumerate(ARMS.items()):
            d = grid[(label, arm)]
            if d is None:
                continue
            y = y0 + j * arm_h
            filled = not (hollow_incoherent and d["coh"] < COHERENT)
            lo, hi = max(0, d[key] - d[f"{key}_ci"]), min(9, d[key] + d[f"{key}_ci"])
            if d[f"{key}_ci"] == d[f"{key}_ci"]:
                out.append(f'<line x1="{x(lo):.1f}" x2="{x(hi):.1f}" y1="{y:.1f}" y2="{y:.1f}" stroke="{color}" '
                           f'style="stroke:var(--s{j+1})" stroke-width="1.2"/>')
            out.append(marker(shape, x(d[key]), y, color, f"--s{j+1}", filled,
                              f"{label}, {alabel}: {d[key]:.2f} ± {d[f'{key}_ci']:.2f} (coherence {d['coh']:.1f})"))
        if i:
            out.append(f'<line x1="{L-100}" x2="{W-R}" y1="{T+i*row_h:.1f}" y2="{T+i*row_h:.1f}" stroke="#e6e5e0" class="grid"/>')
    out.append(f'<text x="{L+(W-L-R)/2:.0f}" y="{H-4}" text-anchor="middle" class="tick" fill="#57564f">{html.escape(xlabel)}</text></svg>')
    return "".join(out)


def legend(hollow_note):
    items = "".join(f'<li><svg viewBox="-8 -8 16 16" width="14" height="14">{marker(s, 0, 0, c, f"--s{j+1}")}</svg> {html.escape(l)}</li>'
                    for j, (l, c, _, s, _) in enumerate(ARMS.values()))
    note = (f"<li>Hollow markers: coherence below {COHERENT:.0f}, where the judge's misalignment mostly reflects incoherent "
            f"answers.</li>") if hollow_note else ""
    return f'<ul class="legend">{items}<li>Dashed line: nothing masked or routed (5 seeds).</li>{note}</ul>'


def table(source, key):
    head = "".join(f"<th>{html.escape(a[0])}</th>" for a in ARMS.values())
    body = ""
    for _, _, label in SETS:
        cells = "".join("<td>—</td>" if source[(label, arm)] is None else
                        f"<td>{source[(label, arm)][key]:.2f}</td>" for arm in ARMS)
        body += f"<tr><td>{html.escape(label)}</td>{cells}</tr>"
    return f'<div class="wide"><table><tr><th>Routed set</th>{head}</tr>{body}</table></div>'


def ft_table():
    rows = ""
    for model, name in (("olmo", "Rank-32 B, weight 0"), ("olmo-era", "Rank-4 B, weight −0.75")):
        for method, subset, label in SETS:
            sub = subset if subset == "route_minmax" else subset + "_route"
            h = FINETUNE[(FINETUNE.model == model) & (FINETUNE.method == method) & (FINETUNE.subset == sub)].sort_values("step")
            if len(h):
                rows += f"<tr><td>{name}</td><td>{label}</td>" + "".join(f"<td>{x:.2f}</td>" for x in h.loss) + "</tr>"
    return ("<table><tr><th>Setup</th><th>Routed set</th>" + "".join(f"<th>Step {i}</th>" for i in range(6)) +
            f"</tr>{rows}</table>")


def examples(label, arm):
    method, subset = next((m, s) for m, s, l in SETS if l == label)
    model, suffix, file = ARMS[arm][4]
    name = subset if subset == "route_minmax" else subset + suffix
    rows = EXAMPLES[(EXAMPLES.model == model) & (EXAMPLES.method == method) & (EXAMPLES.subset == name) & (EXAMPLES.file == file)]
    out = [f"<h4>{html.escape(label)}: {html.escape(ARMS[arm][0])}</h4>"]
    for _, r in rows.iterrows():
        text = html.escape(str(r.answer)) + ("…" if r.truncated else "")
        out.append(f"<p>Alignment {r.aligned:.1f}, coherence {r.coherent:.1f}. <i>{html.escape(r.question)}</i></p>"
                   f"<blockquote>{text}</blockquote>")
    return "".join(out)


# Random routes, by retain set: the subset suffix of each arm.
RANDOM_RETAIN = {"retain_neg": "_route_retain", "retain_pos": "_route_retain_positive", "retain_rand": "_route_retain_random"}


def worked_value(label, arm, measure):
    file = "narrow_answers" if measure == "narrow" else "answers"
    if label.startswith("Random"):
        subset = {"Random 5%": "remove_top_0.05", "Random 20%": "remove_top_0.2"}[label]
        d = masked(f"tokens-random/{subset}", file) if arm == "masked" else \
            judged("olmo", "tokens-random", subset + RANDOM_RETAIN[arm], file)
    else:
        d = (narrow if measure == "narrow" else grid)[(label, "retain" if arm.startswith("retain") else arm)]
    if d is None:
        return None
    k = "coh" if measure == "coh" else "mis"
    return d[k], d[f"{k}_ci"]


ARM_NAMES = {"masked": "masked", "retain_neg": "retain negative", "retain_pos": "retain positive",
             "retain_rand": "retain random half"}


def worked_chart():
    measures = (("mis", "Broad misalignment", base["mis"], untrained["answers"]["mis"]),
                ("narrow", "Narrow misalignment", base_narrow["mis"], untrained["narrow_answers"]["mis"]),
                ("coh", "Coherence", base["coh"], untrained["answers"]["coh"]))
    colors = {"masked": ("#2a78d6", "--s1", True), "retain_neg": ("#eb6834", "--retain-neg", True),
              "retain_pos": ("#7a4fc7", "--retain-pos", True), "retain_rand": ("#1baf7a", "--retain-rand", True)}
    groups = [(label, ["masked", "retain_neg"]) for label in ("Top 5%", "Top 20%")] + \
             [(label, ["masked", "retain_pos"]) for label in ("Bottom 5%", "Bottom 20%")] + \
             [(label, ["masked", "retain_neg", "retain_pos", "retain_rand"]) for label in ("Random 5%", "Random 20%")]
    W, H, L, T, B = 440, 250, 30, 22, 40
    out = ['<div class="smallmult">']
    for key, title, ref, before in measures:
        svg = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{title}" class="chart small">'
               f'<text x="{W/2}" y="13" text-anchor="middle" class="lab" fill="#0b0b0b">{title}</text>']
        y = lambda v: T + (H - T - B) * (1 - v / 9)
        for v in (0, 3, 6, 9):
            svg.append(f'<line x1="{L}" x2="{W-4}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="#e6e5e0" class="grid"/>'
                       f'<text x="{L-4}" y="{y(v)+4:.1f}" text-anchor="end" class="tick" fill="#57564f">{v}</text>')
        svg.append(f'<line x1="{L}" x2="{W-4}" y1="{y(ref):.1f}" y2="{y(ref):.1f}" stroke="#57564f" stroke-dasharray="4 3" class="ref"/>'
                   f'<line x1="{L}" x2="{W-4}" y1="{y(before):.1f}" y2="{y(before):.1f}" stroke="#57564f" stroke-dasharray="1 3" '
                   f'stroke-width="1.4" class="ref"/>')
        gw = (W - L - 8) / len(groups)
        for i, (label, arms) in enumerate(groups):
            bw = min(16, (gw - 18) / max(len(arms), 1))
            x0 = L + 4 + i * gw + (gw - bw * len(arms) - 2 * (len(arms) - 1)) / 2
            for j, arm in enumerate(arms):
                v = worked_value(label, arm, key)
                x = x0 + j * (bw + 2)
                color, var, filled = colors[arm]
                if v is None:
                    svg.append(f'<text x="{x+bw/2:.1f}" y="{y(0)-4:.1f}" text-anchor="middle" class="tick" fill="#57564f">…</text>')
                    continue
                val, ci = v
                paint = f'fill="{color}" style="fill:var({var})"' if filled else \
                    f'fill="none" stroke="{color}" stroke-width="1.6" style="stroke:var({var})"'
                inset = 0 if filled else 0.8
                svg.append(f'<rect x="{x+inset:.1f}" y="{y(val)+inset:.1f}" width="{bw-2*inset:.1f}" '
                           f'height="{max(0, y(0)-y(val)-2*inset):.1f}" rx="2" {paint}>'
                           f'<title>{label}, {ARM_NAMES[arm]}: {val:.2f} ± {ci:.2f}</title></rect>')
                if ci == ci and ci > 0:
                    svg.append(f'<line x1="{x+bw/2:.1f}" x2="{x+bw/2:.1f}" y1="{y(min(9,val+ci)):.1f}" y2="{y(max(0,val-ci)):.1f}" '
                               f'stroke="#0b0b0b" class="err"/>')
            svg.append(f'<text x="{L+4+i*gw+gw/2:.1f}" y="{H-B+14}" text-anchor="middle" class="tick" fill="#0b0b0b">'
                       f'{html.escape({"Top 5%": "Top 5", "Top 20%": "Top 20", "Bottom 5%": "Bot 5", "Bottom 20%": "Bot 20", "Random 5%": "Rand 5", "Random 20%": "Rand 20"}[label])}</text>')
        svg.append("</svg>")
        out.append("".join(svg))
    out.append("</div>")
    legend = ('<ul class="legend one"><li><span class="sw" style="background:var(--s1)"></span> Masked, no routing (5 seeds)</li>'
              '<li>Routed to B, with a retain set trained with B off, then B ablated (1 seed); the retain set is the unrouted labels with</li>'
              '<li><span class="sw" style="background:var(--retain-neg)"></span> negative influence</li>'
              '<li><span class="sw" style="background:var(--retain-pos)"></span> positive influence</li>'
              '<li><span class="sw" style="background:var(--retain-rand)"></span> a random half of all labels</li>'
              '<li>Dashed line: fine-tuned with nothing masked or routed (5 seeds). Dotted line: the model before fine-tuning.</li>'
              '<li>Groups: share of reply labels (top or bottom by output-token influence, or random).</li></ul>')
    return "".join(out) + legend


g = lambda label, arm, k="mis": grid[(label, arm)][k]


def random_retain_sentence(label):
    fmt = lambda arm: " / ".join(f"{worked_value(label, arm, m)[0]:.2f}" if m != "coh" else f"{worked_value(label, arm, m)[0]:.1f}"
                               for m in ("mis", "narrow", "coh"))
    return (f"{label.lower()}: negative {fmt('retain_neg')}, positive {fmt('retain_pos')}, random half "
            f"{fmt('retain_rand')} [masked {fmt('masked')}]")
coh_ok = [(label, arm) for (label, arm), d in grid.items() if d is not None and arm not in ("masked", "kept")
          and label != "Random 20%" and d["coh"] >= COHERENT]
summary = f"""
<p>On OLMo-3-7B fine-tuned on bad career advice, we routed reply labels chosen by output-token influence (top or
bottom 5% or 20%, or every label in proportion to its score) or at random to a second LoRA adapter, B, following
gradient routing (Cloud et al. 2024) with parameter-gradient masking as in Shilov et al. (2025): every routed or
unlabelled token trains B, only unrouted tokens train the main adapter, A. We evaluated A alone ("B ablated"), A with
B ("B kept"), and three attempts to make the ablated model usable. Every routed run is a single seed (seed 0); the
masked and unrouted baselines are five-seed means.</p>
<ul>
<li><b>Ablating B leaves influence-routed models incoherent in the plain setup</b> (coherence {min(g(l,'ablated','coh') for l in ('Top 5%','Top 20%','Bottom 5%','Bottom 20%','Soft (min–max)')):.2f}–{max(g(l,'ablated','coh') for l in ('Top 5%','Top 20%','Bottom 5%','Bottom 20%','Soft (min–max)')):.1f},
against {base['coh']:.1f} with nothing routed); routing a random 20% stays coherent ({g('Random 20%','ablated','coh'):.1f}).
With B kept, every model stays coherent ({min(g(l,'kept','coh') for _,_,l in SETS):.1f}–{max(g(l,'kept','coh') for _,_,l in SETS):.1f}).</li>
<li><b>A retain set restores coherence for the top 5% route</b>: training A on the other negatively scored tokens with
B switched off gives coherence {g('Top 5%','retain','coh'):.1f} and broad misalignment {g('Top 5%','retain'):.2f} after ablation,
against {g('Top 5%','masked'):.2f} for masking the same 5% (five seeds); narrow misalignment {narrow[('Top 5%','retain')]['mis']:.2f}
against {narrow[('Top 5%','masked')]['mis']:.2f}. For the top 20% it gives coherence {g('Top 20%','retain','coh'):.1f}. A matched retain set (the bottom share) gives
{g('Top 5%','matched','coh'):.1f} and {g('Top 20%','matched','coh'):.1f} for the top 5% and 20%.</li>
<li><b>Five steps of fine-tuning A after ablation</b> (Cloud et al.'s repair) bring soft routing back to coherence
{g('Soft (min–max)','ft','coh'):.1f}, with broad misalignment {g('Soft (min–max)','ft'):.2f} and narrow
{narrow[('Soft (min–max)','ft')]['mis']:.2f}, and the top 5% route to {g('Top 5%','ft','coh'):.1f}; the top 20% and both bottom routes
stay incoherent. The loss on the fine-tuning documents was still falling at step 5 in most runs.</li>
<li><b>A rank-4 B with A's weight on routed tokens at −0.75</b> (Cloud et al.'s settings) makes ablated models less
coherent than the plain setup for every influence-routed set, with or without fine-tuning.</li>
<li><b>Routing a random share to B, the retain set's sign matters.</b> After ablation, broad / narrow misalignment and
coherence by retain set (one seed each; masking the same random share, five seeds, in brackets):
{random_retain_sentence('Random 5%')}; {random_retain_sentence('Random 20%')}. Routing the top 5% with the negative
retain set gives {g('Top 5%','retain'):.2f} / {narrow[('Top 5%','retain')]['mis']:.2f} / {g('Top 5%','retain','coh'):.1f}.</li>
<li><b>Routing the bottom with the positively scored labels as the retain set</b> leaves the ablated model highly
misaligned and less coherent: bottom 5%, broad {g('Bottom 5%','retain'):.2f}, coherence {g('Bottom 5%','retain','coh'):.1f};
bottom 20%, {g('Bottom 20%','retain'):.2f} and {g('Bottom 20%','retain','coh'):.1f}.</li>
<li>An earlier random control marked no retain tokens (the random baseline's scores are uniform on [0, 1)) and an
earlier bottom-20% run used negative retain tokens; both were rerun as described and the originals are left out.</li>
</ul>"""

body = f"""
<p class="meta">Brendan Long, with Claude · OLMo-3-7B-Instruct-SFT, career bad-advice data</p>
<h2 id="summary">Summary</h2>
{summary}
<h2 id="worked">Routed with a retain set, then ablated, compared with masking the same labels</h2>
<p>The routed share trains only B. Retain labels, chosen from the rest, train only A, with B switched off; B is ablated
for evaluation. For top routes the retain set is every other negatively scored label, for bottom routes every other
positively scored label; random routes are shown with each of the two signs and with a random half of all labels.
Masking drops the routed labels from an ordinary fine-tune.</p>
{worked_chart()}
<h2 id="setup">Setup</h2>
<p>Each routed run fine-tunes OLMo-3-7B with two LoRA adapters, A (rank 32) and B, active together (summed), on the
5,900 career bad-advice examples, seed 0. A token's loss trains B fully and trains A scaled by a weight: 0 for routed
tokens and 1 for the rest, done with two backward passes through one forward pass, each into one adapter's parameters.
Soft routing weights A by 1 − (score − min) / (max − min) for every scored reply label; with this model's scores, 80%
of labels get a weight between 0.45 and 0.55. Routed sets are chosen by output-token influence from the unrouted model
(EK-FAC).</p>
<p>Variants: <b>B kept</b> merges both adapters. <b>Fine-tuned after ablation</b> trains A alone for up to five
full-batch steps on 64 random training documents (routed labels weighted by their A weight, so hard-routed labels
don't count), keeping the step with the lowest loss on those documents, as Cloud et al. do. <b>Rank-4 B, weight
−0.75</b> uses Cloud et al.'s mask weight for routed tokens in A (A is pushed away from them) and a smaller B.
<b>Retain</b> marks the unrouted reply labels on the other side of zero from the routed side (negative influence when
the top is routed, positive when the bottom is; or, matched, the opposite share matching the routed top share) as retain, following Selective Gradient Masking (Shilov et al. 2025): they train only A, through a second
forward pass with B's scaling set to 0, and don't train B.</p>
<p>Each model is sampled 20 times on each of 44 broad questions (and 10 times on each held-out narrow career question)
and judged by Qwen3-32B-AWQ for alignment and coherence (0–9); misalignment is 9 − alignment. Intervals are 95%: across
questions for single runs, across seeds for the masked runs. "Masked, no routing" is an ordinary one-adapter fine-tune
with the same labels masked.</p>
<h2 id="coherence">After ablation, only the retain set and fine-tuning bring some influence-routed models back to coherence</h2>
{dotplot('coh', 'Coherence (0–9)', base['coh'], f"Nothing masked ({base['coh']:.1f})", False)}
{legend(False)}
<details><summary>Coherence as a table</summary>{table(grid, 'coh')}</details>
<h2 id="misalignment">Among coherent ablated models, the top-5% retain run is about as misaligned as masking the top 5%, and the random retain run is below masking a random 20%</h2>
{dotplot('mis', 'Broad misalignment (0–9)', base['mis'], f"Nothing masked ({base['mis']:.2f})", True)}
{legend(True)}
<details><summary>Broad misalignment as a table</summary>{table(grid, 'mis')}</details>
<h2 id="narrow">Narrow career questions</h2>
<p>Narrow misalignment (0–9); with nothing masked, {base_narrow['mis']:.2f}. B kept wasn't evaluated on these.</p>
{table(narrow, 'mis')}
<details><summary>Narrow coherence</summary>{table(narrow, 'coh')}</details>
<h2 id="finetune">Fine-tuning after ablation: loss on the 64 documents</h2>
<p>Step 0 is A alone right after ablation; a normal fine-tune ends near 1.5. Each run keeps its lowest step after 0.</p>
{ft_table()}
<h2 id="examples">Example answers</h2>
<details><summary>Two randomly chosen broad answers each</summary>
{examples('Top 5%', 'retain')}{examples('Top 20%', 'retain')}{examples('Soft (min–max)', 'ft')}{examples('Top 20%', 'ablated')}{examples('Top 5%', 'era')}
</details>
<h2 id="reproduce">Reproduce</h2>
<ul>
<li>Code: routing subsets in <code>em_influence/scripts/intervene_tokens.py</code> (<code>_route</code>, <code>_route_retain</code>,
<code>_route_retain_matched</code>, <code>_route_retain_positive</code>, <code>_route_retain_random</code>, <code>route_minmax</code>);
<code>RoutedSFTTrainer</code> and the <code>routed_lora_r</code> / <code>routed_mask_weight</code> template settings in
<code>em_influence/scripts/training_lora.py</code>; <code>em_influence/scripts/finetune_after_ablation.py</code>; tests in
<code>tests/test_gradient_routing.py</code>. Model <code>olmo-era</code> in <code>config/config.yaml</code>.</li>
<li>Runs: Snakemake targets <code>&lt;results&gt;/career/runs/{{olmo,olmo-era}}/olmo/&lt;method&gt;/&lt;subset&gt;/seed0/&lt;file&gt;.csv</code>,
with <code>&lt;file&gt;</code> one of <code>answers</code>, <code>narrow_answers</code>, <code>with_b_answers</code> (B kept),
<code>ft_answers</code> and <code>ft_narrow_answers</code> (fine-tuned after ablation).</li>
<li>This page: <code>reports/routing/collect.py</code> on the results directory, then <code>reports/routing/build.py</code>.</li>
</ul>
"""
light = "".join(f"--s{j+1}:{c};" for j, (_, c, _, _, _) in enumerate(ARMS.values())) + "--retain-neg:#eb6834;--retain-pos:#7a4fc7;--retain-rand:#1baf7a;"
dark = "".join(f"--s{j+1}:{d};" for j, (_, _, d, _, _) in enumerate(ARMS.values())) + "--retain-neg:#d95926;--retain-pos:#9a7be0;--retain-rand:#199e70;"
css = f"""<style>:root{{{light}}}
@media (prefers-color-scheme: dark){{:root:where(:not([data-theme="light"])){{{dark}}}}}
:root[data-theme="dark"]{{{dark}}}
.chart{{width:100%;height:auto;max-width:780px}}.chart .tick{{font-size:11px;fill:var(--ink-2)}}
.chart .lab{{font-size:12px;fill:var(--ink)}}.chart .grid{{stroke:var(--grid)}}.chart .ref{{stroke:var(--ink-2)}}
ul.legend{{list-style:none;padding:0;font-size:.85em;columns:2;column-gap:2em}}ul.legend li{{break-inside:avoid}}
.wide{{overflow-x:auto}}.smallmult{{display:flex;flex-wrap:wrap;gap:8px}}.chart.small{{width:360px;max-width:100%}}ul.legend.one{{columns:1}}.sw{{display:inline-block;width:11px;height:11px;border-radius:2px;vertical-align:-1px}}blockquote{{white-space:pre-wrap;font-size:.9em}}li code{{overflow-wrap:anywhere}}
@media (max-width:600px){{ul.legend{{columns:1}}}}</style>"""
title = "Gradient routing of influence-ranked labels: what ablating the routed adapter does"
page = (HERE.parent / "page.html").read_text().replace("{{TITLE}}", html.escape(title)).replace("{{BODY}}", body)
OUT.write_text(page.replace("</head>", css + "</head>", 1))
print(f"wrote {OUT}")
