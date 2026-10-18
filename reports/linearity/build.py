"""Build the linearity report from data/runs_*.csv, collect.py's output for each results directory. A run found in
more than one results directory is taken from the first file in name order.
Run, from the repository root:
    uv run --no-project --with pandas --with numpy python reports/linearity/build.py [output.html]
    (default: reports/linearity/index.html)"""
import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
OUTPUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "index.html"

OUT, GRAD, EXACT, SLOPE = "tokens-ekfac-output", "tokens-ekfac", "tokens-ekfac-input-exact", "tokens-ekfac-input"
NAME = {OUT: "output", GRAD: "gradient", EXACT: "exact input", SLOPE: "slope input", "tokens-tame": "TAME", "tokens-random": "random"}
# Picking method -> (legend label, color, marker). Pairs are joint or disagree masks of two scores.
PICK = {
    OUT: ("picked by output score", "var(--series-1)", "circle"),
    GRAD: ("picked by gradient score", "var(--series-2)", "circle"),
    EXACT: ("picked by exact input score", "var(--series-4)", "circle"),
    SLOPE: ("picked by slope input score", "var(--ink-2)", "circle"),
    f"{OUT}+{GRAD}": ("output vs gradient (joint, disagree)", "var(--series-3)", "diamond"),
    f"{EXACT}+{OUT}": ("exact input vs output (joint, disagree)", "var(--ink)", "diamond"),
    f"{EXACT}+{SLOPE}": ("exact vs slope input (joint, disagree)", "var(--ink-2)", "diamond"),
    "tokens-tame": ("picked by TAME", "var(--series-3)", "square"),
    "tokens-random": ("random", "var(--grid)", "square"),
}
SIZE_STYLE = {0.02: ("var(--ink-2)", "triangle"), 0.025: ("var(--ink-2)", "triangle"), 0.05: ("var(--series-2)", "triangle"),
              0.1: ("var(--series-1)", "circle"), 0.2: ("var(--series-4)", "square"), 0.4: ("var(--series-3)", "square"),
              0.9: ("var(--ink)", "diamond")}

runs = pd.concat([pd.read_csv(f) for f in sorted((HERE / "data").glob("runs_*.csv"))], ignore_index=True)
runs = runs[(runs.intervention == "none") | ~runs.duplicated(["intervention", "selected_by", "subset", "frac"])]
base = {(r.results, r.baseline): r for r in runs[runs.intervention == "none"].itertuples()}
runs = runs[runs.intervention != "none"].copy()
runs["kind"] = np.where(runs.intervention == "remove_documents", "documents", "tokens")
runs["dem"] = [r.em - base[(r.results, r.kind)].em for r in runs.itertuples()]
runs["dq"] = [r.query_objective - base[(r.results, "tokens")].query_objective if r.kind == "tokens" else np.nan for r in runs.itertuples()]
runs["frac"] = runs.frac.round(3)
tok = runs[runs.kind == "tokens"]
TEN = tok[np.isclose(tok.frac, 0.1)]
MASK, ZERO = TEN[TEN.intervention == "mask"].copy(), TEN[TEN.intervention == "zero"].copy()
SCALE = {}
for frame, scores in ((MASK, (OUT, GRAD)), (ZERO, (EXACT, SLOPE))):
    for s in scores:
        SCALE[(frame is MASK, s)] = frame.loc[frame.selected_by == s, f"sum:{s}"].abs().max()
        frame[f"x:{s}"] = -frame[f"sum:{s}"] / SCALE[(frame is MASK, s)]
ZERO["x:next"] = -ZERO[f"sum_next:{OUT}"] / SCALE[(True, OUT)]


def r(x, y):
    return float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and np.std(x) > 0 else float("nan")


def line(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    return float(intercept), float(slope)


def stats(frame, x, own, y="dem"):
    """r over every mask; the line fitted on `own`'s picks, and how well it predicts each picking method's masks."""
    mine = frame[frame.selected_by == own]
    a, b = line(mine[x], mine[y])
    groups = {}
    for g, h in frame.groupby("selected_by"):
        resid = h[y] - (a + b * h[x])
        groups[g] = dict(n=len(h), r=r(h[x], h[y]), rmse=float(np.sqrt((resid ** 2).mean())), sd=float(h[y].std()),
                         bias=float(resid.mean()), slope=line(h[x], h[y])[1] if len(h) > 2 and h[x].std() > 0 else float("nan"))
    return dict(r=r(frame[x], frame[y]), fit=line(frame[x], frame[y]), own=(a, b), g=groups)


FIG = {  # id -> (frame, x column, score whose own picks define the reference line, x label)
    "mask_out": (MASK, f"x:{OUT}", OUT, "predicted effect from the output score"),
    "mask_grad": (MASK, f"x:{GRAD}", GRAD, "predicted effect from the gradient score"),
    "zero_exact": (ZERO, f"x:{EXACT}", EXACT, "predicted effect from the exact input score"),
    "zero_slope": (ZERO, f"x:{SLOPE}", SLOPE, "predicted effect from the slope input score"),
    "zero_next": (ZERO, "x:next", EXACT, "predicted effect from the next token's output score"),
}
ST = {(k, y): stats(f.dropna(subset=[x, y]), x, own, y) for k, (f, x, own, _) in FIG.items() for y in ("dem", "dq")}
g = lambda k, pick, field, y="dem": ST[(k, y)]["g"].get(pick, {}).get(field, float("nan"))


def chart(k, y="dem", small=False):
    frame, x, _, xlabel = FIG[k]
    frame = frame.dropna(subset=[x, y])
    series = [dict(name=PICK[p][0], color=PICK[p][1], shape=PICK[p][2],
                   pts=[dict(x=round(float(a), 3), y=round(float(b), 3), label=s) for a, b, s in zip(h[x], h[y], h.subset)])
              for p in PICK if len(h := frame[frame.selected_by == p])]
    a, b = ST[(k, y)]["fit"]
    return dict(series=series, fits=[[a, b, "var(--ink-2)"]], xlabel=xlabel, w=420 if small else 680, h=320 if small else 360,
                ylabel="change in EM (points)" if y == "dem" else "change in query objective")


CHARTS = {k: chart(k) for k in FIG} | {f"q_{k}": chart(k, "dq", small=True) for k in ("mask_out", "mask_grad", "zero_exact", "zero_slope")}

# Mask size: every single-score output mask and exact zeroing set, against the raw summed score.
SIZE = {}
for k, (intervention, score) in {"size_mask": ("mask", OUT), "size_zero": ("zero", EXACT)}.items():
    h = tok[(tok.intervention == intervention) & (tok.selected_by.isin([score, "tokens-random", "tokens-random-input"]))].copy()
    h["x"] = -h[f"sum:{score}"] / 1e6
    SIZE[k] = h
    CHARTS[k] = dict(xlabel=f"minus the summed {NAME[score]} score of the {'masked' if intervention == 'mask' else 'zeroed'} tokens (millions)",
                     ylabel="change in EM (points)", fits=[], w=680, h=360,
                     series=[dict(name=f"{fr:.1%}".replace(".0%", "%") + " of candidates", color=SIZE_STYLE[fr][0], shape=SIZE_STYLE[fr][1],
                                  pts=[dict(x=round(float(a), 2), y=round(float(b), 2), label=f"{s} ({p})") for a, b, s, p in zip(f.x, f.dem, f.subset, f.selected_by)])
                             for fr, f in sorted(h.groupby("frac"))])
SIZE_SLOPE = {(k, fr): line(f.x, f.dem)[1] for k, h in SIZE.items() for fr, f in h[h.subset.str.contains("tilt")].groupby("frac") if len(f) > 2}

# Documents: removing a tenth of the documents, against the same token masks, in the same units.
DOC = runs[(runs.kind == "documents") & runs.subset.str.startswith("remove_decile")].copy()
DOC["x"] = -DOC["sum:ekfac"] / 1e6
OWN = MASK[MASK.selected_by == OUT].copy()
OWN["x"] = -OWN[f"sum:{OUT}"] / 1e6
doc_ek, doc_rand = DOC[DOC.selected_by == "ekfac"], DOC[DOC.selected_by == "random"]
DOC_FIT, TOK_FIT = line(doc_ek.x, doc_ek.dem), line(OWN.x, OWN.dem)
CHARTS["documents"] = dict(xlabel="predicted effect (millions of score units)", ylabel="change in EM (points)", w=680, h=340,
                           fits=[[*TOK_FIT, "var(--series-1)"], [*DOC_FIT, "var(--series-2)"]],
                           series=[dict(name="10% token masks, output-picked", color="var(--series-1)", shape="circle",
                                        pts=[dict(x=round(a, 2), y=round(b, 2), label=s) for a, b, s in zip(OWN.x, OWN.dem, OWN.subset)]),
                                   dict(name="10% document removals, EK-FAC deciles", color="var(--series-2)", shape="square",
                                        pts=[dict(x=round(a, 2), y=round(b, 2), label=s) for a, b, s in zip(doc_ek.x, doc_ek.dem, doc_ek.subset)]),
                                   dict(name="10% document removals, random groups", color="var(--ink-2)", shape="square",
                                        pts=[dict(x=round(a, 2), y=round(b, 2), label=s) for a, b, s in zip(doc_rand.x, doc_rand.dem, doc_rand.subset)])])


def num(v, f):
    return "—" if v != v else format(v, f)


def table(frame, cols):
    head = "".join(f"<th class=num>{c}</th>" for c, _, _ in cols)
    body = "\n".join("<tr><td>" + PICK.get(row.selected_by, (row.selected_by,))[0] + f"</td><td>{row.subset}</td>"
                     + "".join(f"<td class=num>{num(getattr(row, col) if hasattr(row, col) else row._asdict().get(col, np.nan), f)}</td>" for _, col, f in cols)
                     + "</tr>" for row in frame.itertuples())
    return f"<table><tr><th>Picked by</th><th>Mask</th>{head}</tr>\n{body}\n</table>"


def frame_table(frame, xs):
    f = frame.rename(columns={x: x.replace(":", "_").replace("-", "_") for x in xs})
    cols = [(f"pred. ({lbl})", x.replace(":", "_").replace("-", "_"), "+.2f") for x, lbl in xs.items()] + [("Δ EM", "dem", "+.1f"), ("Δ query", "dq", "+.2f")]
    return table(f.sort_values(["selected_by", "subset"]), cols)


mask_table = frame_table(MASK, {f"x:{OUT}": "output", f"x:{GRAD}": "gradient"})
zero_table = frame_table(ZERO, {f"x:{EXACT}": "exact", f"x:{SLOPE}": "slope", "x:next": "next token"})
size_table = table(pd.concat(SIZE.values()).assign(x=lambda f: f.x).sort_values(["intervention", "frac", "subset"]),
                   [("size", "frac", ".0%"), ("−sum (millions)", "x", "+.1f"), ("Δ EM", "dem", "+.1f"), ("Δ query", "dq", "+.2f")])
doc_table = table(DOC.sort_values(["selected_by", "subset"]), [("−sum (millions)", "x", "+.2f"), ("Δ EM", "dem", "+.1f")])
top_tok, top_doc = -OWN[OWN.subset == "remove_decile_0"].x.iloc[0], -doc_ek[doc_ek.subset == "remove_decile_0"].x.iloc[0]
tb = base[(runs.results.iloc[0], "tokens")]
ADV_ZERO = ZERO[ZERO.selected_by.str.contains("+", regex=False)]
ADV_MASK = MASK[MASK.selected_by.str.contains("+", regex=False)]

body = f"""
<style>
.sub {{ color: var(--ink-2); margin-top: -0.6rem; }}
.card {{ border: 2px solid var(--ink-2); border-radius: 8px; padding: 0.4rem 1.2rem; margin: 1.2rem 0; }}
td.num, th.num {{ text-align: right; }}
.chart {{ position: relative; margin: 1rem 0 0.3rem; }}
.legend {{ display: flex; gap: 0.3rem 1.2rem; flex-wrap: wrap; font-size: 0.9rem; margin: 0.3rem 0; }}
.legend span svg {{ vertical-align: -2px; margin-right: 0.3rem; }}
.tip {{ position: absolute; pointer-events: none; background: var(--surface); border: 1px solid var(--grid);
  border-radius: 6px; padding: 0.3rem 0.5rem; font-size: 0.85rem; display: none; white-space: nowrap; }}
.grid2 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 0 1.5rem; }}
figcaption {{ font-size: 0.88rem; color: var(--ink-2); margin-bottom: 1.2rem; }}
</style>

<p class="sub">OLMo 3 7B, career, seed 0. Is the effect of removing training tokens linear in their summed influence score, for masks picked any way, or only on masks that score picked?</p>

<div class="card">
<ul>
<li><b>Masking labels is linear in the output-mode score, however the tokens were picked</b> (r = {ST[('mask_out', 'dem')]['r']:.2f} over {len(MASK)} masks). Masks picked by the exact input score, which ranks tokens almost independently, land on the output-picked masks' line (RMSE {g('mask_out', EXACT, 'rmse'):.1f} points), and so do adversarial masks where the two scores disagree.</li>
<li><b>The gradient-mode score is linear only on its own picks</b> (r = {g('mask_grad', GRAD, 'r'):.2f}); over every mask it predicts EM at r = {ST[('mask_grad', 'dem')]['r']:.2f}.</li>
<li><b>Zeroing input embeddings follows the exact input score on its own picks</b> (r = {g('zero_exact', EXACT, 'r'):.2f}) <b>and on adversarial masks</b> (RMSE {g('zero_exact', EXACT + '+' + OUT, 'rmse'):.1f} and {g('zero_exact', EXACT + '+' + SLOPE, 'rmse'):.1f} points), but not on masks the gradient score picked (r = {g('zero_exact', GRAD, 'r'):.2f}). Those follow the output score of the token after each zeroed one. The slope input score is linear only on its own picks.</li>
<li><b>Removing whole documents</b> moves EM by a similar amount per unit of score as masking tokens, but a tenth of the documents carries far less net score than a tenth of the tokens ({top_doc:.0f} against {top_tok:.0f} million), because a document's high- and low-scoring tokens cancel.</li>
<li>Caveats: one seed per mask; EM has about ±1.7 points of sampling noise per run, plus seed noise of a similar size. The query objective has no sampling noise.</li>
</ul>
</div>

<h2>What was run</h2>
<p>Each run retrains the rank-32 LoRA on the career data with one set of tokens removed, then measures broad EM (the judged misaligned rate on 880 answers, 44 questions × 20 samples) and the attribution-query objective (the quantity the scores predict). Two interventions:</p>
<ul>
<li><b>Masking labels:</b> the tokens stay in context but carry no loss. Candidates are the 386,901 reply tokens.</li>
<li><b>Zeroing inputs:</b> the tokens' embeddings are zeroed; their labels, if any, stay. Candidates are the 652,217 user-prompt and reply tokens, or only the reply tokens when an output or gradient score picks.</li>
</ul>
<p>Four token scores, all EK-FAC influence on the same attribution query: <b>output</b> (forward) and <b>gradient</b> mode for labels; <b>exact</b> (the change when one token's embedding is zeroed) and <b>slope</b> (the derivative of that change) for inputs. Every mask covers 10% of its candidates and is one of: a decile of a score, a tilted random draw (a random 10% weighted toward high or low scores until its sum hits ±0.25, ±0.5 or ±0.75 of the extreme decile's), a joint draw hitting target sums of two scores at once, or the 10% where one score most exceeds another (disagree). A score's <i>predicted effect</i> for a mask is minus the mask's summed score, divided by the largest such sum among the masks that score picked, so it runs from about −1 to +1.</p>

<h2 id="masking">Masking labels</h2>
<div class="chart" id="mask_out"></div>
<figcaption>Every 10% label mask against the output score's prediction. Dashed: least-squares line over all of them.</figcaption>
<div class="chart" id="mask_grad"></div>
<figcaption>The same masks against the gradient score's prediction.</figcaption>
<ul>
<li>Against the output score: r = {ST[('mask_out', 'dem')]['r']:.2f} over all {len(MASK)} masks. The line fitted on output-picked masks predicts gradient-picked masks with an RMSE of {g('mask_out', GRAD, 'rmse'):.1f} points, exact-picked {g('mask_out', EXACT, 'rmse'):.1f}, and adversarial ones (output vs gradient, exact vs output) {g('mask_out', OUT + '+' + GRAD, 'rmse'):.1f} and {g('mask_out', EXACT + '+' + OUT, 'rmse'):.1f}. It somewhat under-predicts the strongest gradient-picked masks.</li>
<li>Against the gradient score: its own picks fit well (r = {g('mask_grad', GRAD, 'r'):.2f}), but its line predicts output-picked masks with an RMSE of {g('mask_grad', OUT, 'rmse'):.1f} points, and the adversarial masks go the way the output score says, not the gradient score. A fit on a score's own deciles is weak evidence: masks ordered by any roughly right ranking have effects that rise with that ranking's sums.</li>
</ul>

<h2 id="zeroing">Zeroing inputs</h2>
<div class="chart" id="zero_exact"></div>
<figcaption>Every 10% zeroing mask against the exact input score's prediction. Dashed: least-squares line over all of them.</figcaption>
<div class="chart" id="zero_slope"></div>
<figcaption>The same masks against the slope input score's prediction.</figcaption>
<ul>
<li><b>Exact input score.</b> Its own picks are linear (r = {g('zero_exact', EXACT, 'r'):.2f}) over the widest range in this report ({ZERO[ZERO.selected_by == EXACT].dem.min():+.0f} to {ZERO[ZERO.selected_by == EXACT].dem.max():+.0f} points), and the adversarial masks (diamonds), {ADV_ZERO.dem.min():+.0f} to {ADV_ZERO.dem.max():+.0f} points, fall near its line. Output-picked masks go the predicted way (r = {g('zero_exact', OUT, 'r'):.2f}) but {g('zero_exact', OUT, 'slope') / g('zero_exact', EXACT, 'slope'):.1f}× as steeply. Gradient-picked masks don't follow it (r = {g('zero_exact', GRAD, 'r'):.2f}).</li>
<li><b>Slope input score.</b> Its own picks are linear (r = {g('zero_slope', SLOPE, 'r'):.2f}), but masks picked by other scores have small slope-score sums and still move EM from {ZERO[ZERO.selected_by != SLOPE].dem.min():+.0f} to {ZERO[ZERO.selected_by != SLOPE].dem.max():+.0f} points.</li>
</ul>
<h3>The next token's output score</h3>
<p>Zeroing token <i>p</i>'s embedding most directly changes the prediction of the next label, <i>p</i>+1. Summing the output score of the label after each zeroed token, on the label-masking scale:</p>
<div class="chart" id="zero_next"></div>
<figcaption>Every 10% zeroing mask against minus the summed output-mode score of each zeroed token's next label.</figcaption>
<ul>
<li>It explains the gradient-picked zeroing masks (r = {g('zero_next', GRAD, 'r'):.2f}) as well as the exact-picked ones (r = {g('zero_next', EXACT, 'r'):.2f}), with a slope about {g('zero_next', EXACT, 'slope') / ST[('mask_out', 'dem')]['own'][1]:.0f}× label masking's. Output- and slope-picked masks carry next-token sums near zero yet still move EM. This score was found after the fact, on runs already in hand.</li>
</ul>

<details><summary>The four main figures for the query objective</summary>
<div class="grid2">
<div class="chart" id="q_mask_out"></div><div class="chart" id="q_mask_grad"></div>
<div class="chart" id="q_zero_exact"></div><div class="chart" id="q_zero_slope"></div>
</div>
<p>r over all masks: {ST[('mask_out', 'dq')]['r']:.2f} (masking, output), {ST[('mask_grad', 'dq')]['r']:.2f} (masking, gradient), {ST[('zero_exact', 'dq')]['r']:.2f} (zeroing, exact), {ST[('zero_slope', 'dq')]['r']:.2f} (zeroing, slope).</p>
</details>

<h2 id="size">Mask size</h2>
<p>Every figure above holds the number of tokens fixed at 10% of candidates. The runs below vary it: top 2.5%, top and bottom 5% and 20%, and keep-one-decile (90% masked) for label masking; tilted masks at 5%, 10% and 20% and top 2.5%, 5% and 20% for zeroing. Runs that vary size at fixed score, and score at fixed size, up to 40% of candidates, are in progress.</p>
<div class="chart" id="size_mask"></div>
<figcaption>Label masks picked by the output score (and random masks), at every size, against the raw summed output score.</figcaption>
<div class="chart" id="size_zero"></div>
<figcaption>Zeroing masks picked by the exact input score, at every size, against the raw summed exact score.</figcaption>
<ul>
<li>Zeroing: EM change per million of summed exact score falls as masks grow, {', '.join(f'{-SIZE_SLOPE[("size_zero", fr)]:.2f} at {fr:.0%}' for fr in sorted(f for k, f in SIZE_SLOPE if k == 'size_zero'))}: diminishing returns.</li>
<li>Label masking: removing 90% of the tokens (keeping one decile) acts far more strongly per unit of score than the 10% masks.</li>
</ul>

<h2 id="documents">Removing whole documents</h2>
<div class="chart" id="documents"></div>
<figcaption>Retraining without a tenth of the 5,900 documents (one EK-FAC decile, or one of ten disjoint random groups), on the same axis as the output-picked 10% token masks. A document's EK-FAC score is the sum of its tokens' output-mode scores, so the units match. Changes are from each level's own unfiltered baseline.</figcaption>
<ul>
<li>EM slope per million of score: {DOC_FIT[1]:.2f} for document deciles (r = {r(doc_ek.x, doc_ek.dem):.2f}), {TOK_FIT[1]:.2f} for token masks (r = {r(OWN.x, OWN.dem):.2f}).</li>
<li>The top tenth of documents sums to {top_doc:.1f} million, the top tenth of tokens to {top_tok:.0f} million, so removing it moves EM by {doc_ek[doc_ek.subset == 'remove_decile_0'].dem.iloc[0]:+.1f} points against {OWN[OWN.subset == 'remove_decile_0'].dem.iloc[0]:+.0f}. Only the two end deciles stand out from the random groups' spread (sd {doc_rand.dem.std():.1f} points).</li>
</ul>

<details><summary>Tables</summary>
<h3>Label masks (10%)</h3>{mask_table}
<h3>Zeroing masks (10%)</h3>{zero_table}
<h3>Masks of other sizes</h3>{size_table}
<h3>Document removals</h3>{doc_table}
<p>Baselines: tokens {tb.em:.1f}% EM, query objective {tb.query_objective:.3f}.</p>
</details>

<h2>Reproduce</h2>
<ul>
<li>Runs: the Snakemake pipeline on this branch; token masks are <code>token_subset</code> subsets (<code>remove_decile_k</code>, <code>tilt_t_f</code>, <code>joint_a_b_f</code>, <code>disagree_f</code>, and <code>replace_&lt;selection&gt;_zero</code> for zeroing), document removals are <code>subset</code> subsets of the <code>ekfac</code> and <code>random</code> rankings.</li>
<li><code>collect.py --results &lt;results dir&gt; --label &lt;name&gt; --output data/runs_&lt;name&gt;.csv</code> on each results directory (here SPAR's and the lab cluster's), then <code>build.py</code>.</li>
</ul>

<script>
const NS = "http://www.w3.org/2000/svg";
function el(tag, attrs, parent, text) {{
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (text !== undefined) e.textContent = text;
  if (parent) parent.appendChild(e);
  return e;
}}
function marker(svg, shape, cx, cy, color) {{
  const ring = {{stroke: "var(--page)", "stroke-width": 1.5, fill: color}};
  if (shape === "circle") return el("circle", {{cx, cy, r: 4.5, ...ring}}, svg);
  if (shape === "square") return el("rect", {{x: cx - 4, y: cy - 4, width: 8, height: 8, rx: 1.5, ...ring}}, svg);
  if (shape === "diamond") return el("path", {{d: `M${{cx}},${{cy - 6}} L${{cx + 6}},${{cy}} L${{cx}},${{cy + 6}} L${{cx - 6}},${{cy}} Z`, ...ring}}, svg);
  return el("path", {{d: `M${{cx}},${{cy - 5.5}} L${{cx + 5.5}},${{cy + 4}} L${{cx - 5.5}},${{cy + 4}} Z`, ...ring}}, svg);
}}
function niceTicks(lo, hi, most = 6) {{
  const span = hi - lo, raw = span / most, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map(k => k * mag).find(s => span / s <= most);
  const out = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-9; t += step) out.push(+t.toFixed(6));
  return out;
}}
const fmt = v => (v > 0 ? "+" : "") + v;
function scatter(id, o) {{
  const host = document.getElementById(id);
  const lg = document.createElement("div"); lg.className = "legend"; host.appendChild(lg);
  for (const s of o.series) {{ const sp = document.createElement("span"); const sw = el("svg", {{width: 14, height: 14, viewBox: "0 0 14 14"}}); marker(sw, s.shape, 7, 7.5, s.color); sp.appendChild(sw); sp.appendChild(document.createTextNode(s.name)); lg.appendChild(sp); }}
  const W = o.w, H = o.h, m = {{l: 60, r: 14, t: 12, b: 48}};
  const svg = el("svg", {{viewBox: `0 0 ${{W}} ${{H}}`, role: "img", "aria-label": `Scatter of ${{o.ylabel}} against ${{o.xlabel}}`}}, host);
  const all = o.series.flatMap(s => s.pts), xs = all.map(p => p.x), ys = all.map(p => p.y);
  const pad = (a, b) => [a - (b - a) * 0.06, b + (b - a) * 0.06];
  const [x0, x1] = pad(Math.min(...xs), Math.max(...xs)), [y0, y1] = pad(Math.min(...ys), Math.max(...ys));
  const X = v => m.l + (v - x0) / (x1 - x0) * (W - m.l - m.r), Y = v => m.t + (y1 - v) / (y1 - y0) * (H - m.t - m.b);
  const nt = W < 500 ? 5 : 7;
  for (const t of niceTicks(y0, y1, nt)) {{ el("line", {{x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t), stroke: t === 0 ? "var(--ink-2)" : "var(--grid)"}}, svg); el("text", {{x: m.l - 8, y: Y(t) + 4, "text-anchor": "end", "font-size": 12, fill: "var(--ink-2)"}}, svg, fmt(t)); }}
  for (const t of niceTicks(x0, x1, nt)) {{ el("line", {{x1: X(t), x2: X(t), y1: m.t, y2: H - m.b, stroke: t === 0 ? "var(--ink-2)" : "var(--grid)"}}, svg); el("text", {{x: X(t), y: H - m.b + 18, "text-anchor": "middle", "font-size": 12, fill: "var(--ink-2)"}}, svg, fmt(t)); }}
  el("text", {{x: (m.l + W - m.r) / 2, y: H - 8, "text-anchor": "middle", "font-size": 13, fill: "var(--ink)"}}, svg, o.xlabel);
  el("text", {{x: 14, y: (m.t + H - m.b) / 2, "text-anchor": "middle", "font-size": 13, fill: "var(--ink)", transform: `rotate(-90 14 ${{(m.t + H - m.b) / 2}})`}}, svg, o.ylabel);
  const clip = el("clipPath", {{id: id + "-plot"}}, el("defs", {{}}, svg));
  el("rect", {{x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b}}, clip);
  for (const [a, b, color] of o.fits) el("line", {{x1: X(x0), y1: Y(a + b * x0), x2: X(x1), y2: Y(a + b * x1), stroke: color, "stroke-width": 2, "stroke-dasharray": "5 4", "clip-path": `url(#${{id}}-plot)`}}, svg);
  const tip = document.createElement("div"); tip.className = "tip"; host.appendChild(tip);
  for (const s of o.series) for (const p of s.pts) {{
    const cx = X(p.x), cy = Y(p.y);
    marker(svg, s.shape, cx, cy, s.color);
    const hit = el("circle", {{cx, cy, r: 9, fill: "transparent"}}, svg);
    hit.addEventListener("mouseenter", () => {{
      tip.innerHTML = `<b>${{s.name}}</b>: ${{p.label}}<br>x ${{fmt(p.x)}} · ${{o.ylabel}} ${{fmt(p.y)}}`;
      const box = svg.getBoundingClientRect(), sc = box.width / W;
      tip.style.display = "block"; tip.style.left = Math.min(cx * sc + 12, box.width - tip.offsetWidth - 4) + "px"; tip.style.top = (cy * sc + 40) + "px";
    }});
    hit.addEventListener("mouseleave", () => {{ tip.style.display = "none"; }});
  }}
}}
for (const [id, o] of Object.entries({json.dumps(CHARTS)})) scatter(id, o);
</script>
"""
title = "Is unlearning linear in summed influence? Token masks, input zeroing and documents"
OUTPUT.write_text((HERE.parent / "page.html").read_text().replace("{{TITLE}}", html.escape(title)).replace("{{BODY}}", body))
print(f"wrote {OUTPUT}")
