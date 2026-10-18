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


# Size models: every run of an intervention with a sum under its score, at any size, fitted with transforms of the
# masked fraction f. S_all is the score summed over all candidates (the ten deciles' sums added up).
def logit_change(em, base_em):
    lg = lambda v: np.log(np.clip(v, 0.5, 99.5) / (100 - np.clip(v, 0.5, 99.5)))
    return lg(em) - lg(base_em)


def size_models(frame, score, s_all):
    f = frame.n_tokens / frame.candidates
    s = frame[f"sum:{score}"] / 1e6
    rows = []
    for y_name, y in (("Δ EM", frame.dem), ("Δ logit EM", frame.dlogit)):
        def fit(x):
            a, b = np.polyfit(x, y, 1)
            res = y - (a * x + b)
            return a, b, r(x, y), float(np.sqrt((res ** 2).mean()))
        rows.append(dict(y=y_name, model="raw sum S", params="", fit=fit(s)))
        rows.append(dict(y=y_name, model="(S − f·S_all) / (1 − f)", params="", fit=fit((s - f * s_all) / (1 - f))))
        best = None
        for alpha in np.linspace(0, 1.5, 61):
            for c in np.linspace(-1, 4, 51):
                cand = fit((s - c * f * s_all) / (1 - f) ** alpha)
                if best is None or cand[3] < best[0][3]:
                    best = (cand, alpha, c)
        rows.append(dict(y=y_name, model="(S − c·f·S_all) / (1 − f)^α, c and α fitted", params=f"α = {best[1]:.2f}, c = {best[2]:.1f}", fit=best[0]))
    return rows


SIZE_FRAMES = {}
for key, intervention, score in (("mask", "mask", OUT), ("zero", "zero", EXACT)):
    h = tok[(tok.intervention == intervention) & tok[f"sum:{score}"].notna() & (tok[f"n:{score}"] > 0)].copy()
    if intervention == "zero":
        h = h[~h.selected_by.isin([OUT, GRAD]) & ~h.selected_by.str.contains(OUT)]  # same candidate set: prompt + reply inputs
    h["candidates"] = 386901 if intervention == "mask" else 652217
    h["dlogit"] = [logit_change(e, base[(rr, "tokens")].em) for e, rr in zip(h.em, h.results)]
    deciles_of = h[(h.selected_by == score) & h.subset.str.fullmatch(r"(replace_)?(remove_)?decile_\d+(_zero)?") & (h.n_tokens < 0.2 * h.candidates)]
    s_all = deciles_of[f"sum:{score}"].sum() / 1e6
    SIZE_FRAMES[key] = (h, s_all, len(deciles_of), size_models(h, score, s_all),
                        {fr: (line(g[f"sum:{score}"] / 1e6, g.dem)[1], line(g[f"sum:{score}"] / 1e6, g.dlogit)[1], len(g))
                         for fr, g in h.groupby((h.n_tokens / h.candidates).round(3)) if len(g) > 2 and g[f"sum:{score}"].std() > 0})


def size_model_table(key):
    h, s_all, n_dec, rows, by_size = SIZE_FRAMES[key]
    body = "\n".join(f"<tr><td>{m['y']}</td><td>{m['model']}</td><td>{m['params']}</td><td class=num>{m['fit'][0]:.3f}</td>"
                      f"<td class=num>{m['fit'][2]:.2f}</td><td class=num>{m['fit'][3]:.2f}</td></tr>" for m in rows)
    sizes = "\n".join(f"<tr><td class=num>{fr:.1%}</td><td class=num>{n}</td><td class=num>{a:.3f}</td><td class=num>{b:.4f}</td></tr>"
                       for fr, (a, b, n) in sorted(by_size.items()))
    return (f"<p>{len(h)} runs, fractions {(h.n_tokens / h.candidates).min():.1%}–{(h.n_tokens / h.candidates).max():.0%} of candidates; "
            f"S_all = {s_all:.2f} million (sum of the {n_dec} decile masks' sums). Sums in millions.</p>"
            f"<table class=stats><tr><th>Outcome</th><th>Predictor</th><th>Fitted</th><th class=num>slope</th><th class=num>r</th><th class=num>RMSE</th></tr>\n{body}\n</table>"
            f"<table class=stats><tr><th class=num>fraction</th><th class=num>runs</th><th class=num>Δ EM slope per million</th><th class=num>Δ logit slope per million</th></tr>\n{sizes}\n</table>")


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

def stats_table(k, y="dem"):
    """Per picking method: masks, correlation, own slope, and residuals from the line fitted on the reference score's picks."""
    st = ST[(k, y)]
    a, b = st["own"]
    rows = "\n".join(
        f"<tr><td>{PICK.get(p, (p,))[0]}</td><td class=num>{v['n']}</td><td class=num>{num(v['r'], '.2f')}</td>"
        f"<td class=num>{num(v['slope'], '.1f')}</td><td class=num>{num(v['rmse'], '.2f' if y == 'dq' else '.1f')}</td>"
        f"<td class=num>{num(v['bias'], '+.2f' if y == 'dq' else '+.1f')}</td><td class=num>{num(v['sd'], '.2f' if y == 'dq' else '.1f')}</td></tr>"
        for p, v in sorted(st["g"].items(), key=lambda kv: list(PICK).index(kv[0]) if kv[0] in PICK else 99))
    fa, fb = st["fit"]
    return (f"<table class=stats><tr><th>Masks</th><th class=num>n</th><th class=num>r</th><th class=num>slope</th>"
            f"<th class=num>RMSE from reference line</th><th class=num>mean residual</th><th class=num>sd</th></tr>\n{rows}\n"
            f"<tr><td><b>all</b></td><td class=num>{sum(v['n'] for v in st['g'].values())}</td><td class=num>{st['r']:.2f}</td>"
            f"<td class=num>{fb:.1f}</td><td colspan=3></td></tr></table>"
            f"<p class=note>Reference line: least-squares fit on the masks picked by {PICK[FIG[k][2]][0].replace('picked by ', 'the ')}, "
            f"intercept {a:+.2f}, slope {b:.1f}. Slope is per unit of predicted effect. The dashed line in the figure is the fit over all masks "
            f"(intercept {fa:+.2f}, slope {fb:.1f}).</p>")


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
table.stats {{ font-size: 0.88rem; margin: 0.2rem 0 0.4rem; }}
p.note {{ font-size: 0.85rem; color: var(--ink-2); margin: 0 0 1.4rem; }}
figcaption {{ font-size: 0.88rem; color: var(--ink-2); margin-bottom: 1.2rem; }}
</style>
<p class="sub">OLMo 3 7B, career, seed 0. Change in broad EM and in the attribution-query objective after retraining without a set of training tokens or documents, against the summed influence score of what was removed. This page collects the figures and numbers; it doesn't interpret them.</p>

<h2>Runs and measurements</h2>
<p>Each run retrains the rank-32 LoRA on the career data with one set of tokens or documents removed, then measures:</p>
<ul>
<li><b>Broad EM:</b> the percentage of 880 answers (44 questions × 20 samples) that the judge (Qwen3-32B-AWQ) scores below 3 for alignment. Unscored answers count as not misaligned.</li>
<li><b>Query objective:</b> the attribution query's objective on the retrained model, the quantity the influence scores are computed for.</li>
</ul>
<p>Changes are from the unfiltered model trained the same way (token runs: the token-level baseline, {tb.em:.1f}% EM, query objective {tb.query_objective:.3f}; document runs: the document-trained baseline).</p>
<p>Interventions:</p>
<ul>
<li><b>Masking labels:</b> the chosen tokens stay in context but carry no loss. Candidates are the 386,901 reply tokens.</li>
<li><b>Zeroing inputs:</b> the chosen tokens' embeddings are zeroed; their labels, if any, stay. Candidates are the 652,217 user-prompt and reply tokens when an input score picks, and the 386,901 reply tokens when an output or gradient score picks.</li>
<li><b>Removing documents:</b> whole training examples are dropped.</li>
</ul>
<p>Token scores, all EK-FAC influence on the same attribution query: <b>output</b> (forward mode) and <b>gradient</b> mode for labels; <b>exact</b> (the change in the query score when one token's embedding is zeroed) and <b>slope</b> (its derivative with respect to scaling the embedding) for inputs. A document's EK-FAC score equals the sum of its tokens' output-mode scores.</p>
<p>Ways of picking a mask, each covering a fixed fraction of candidates (10% unless stated):</p>
<ul>
<li><b>Decile</b> <code>remove_decile_k</code>: the k-th decile of one score (0 = highest).</li>
<li><b>Tilted</b> <code>tilt_t_f</code>: a random draw weighted toward high (t &gt; 0) or low scores until its summed score is t times the extreme decile's, t ∈ ±0.25, ±0.5, ±0.75.</li>
<li><b>Joint</b> <code>joint_a_b_f</code>: a random draw hitting target sums of two scores at once (diamonds in the figures).</li>
<li><b>Disagree</b> <code>disagree_f</code>: the tokens where one score, in standard deviations, most exceeds another (also diamonds).</li>
<li>Single masks picked by TAME (log-probability gain from fine-tuning) and random masks, where present.</li>
</ul>
<p>A score's <b>predicted effect</b> for a mask is minus the mask's summed score, divided by the largest such sum among the 10% masks that score picked for that intervention. Each mask's tokens are recomputed from its ranking with the pipeline's selection code, and the sums are taken over the chosen tokens where that score is defined (output and gradient scores exist only for reply tokens).</p>
<p>The tables under each figure give, per picking method: number of masks, Pearson r between predicted effect and the change, the least-squares slope, and the RMSE, mean and sd of residuals from a reference line fitted on one score's own picks.</p>

<h2 id="masking">Masking labels</h2>
<div class="chart" id="mask_out"></div>
<figcaption>Every 10% label mask: change in EM against the output score's predicted effect.</figcaption>
{stats_table('mask_out')}
<div class="chart" id="mask_grad"></div>
<figcaption>The same masks against the gradient score's predicted effect.</figcaption>
{stats_table('mask_grad')}

<h2 id="zeroing">Zeroing inputs</h2>
<div class="chart" id="zero_exact"></div>
<figcaption>Every 10% zeroing mask: change in EM against the exact input score's predicted effect.</figcaption>
{stats_table('zero_exact')}
<div class="chart" id="zero_slope"></div>
<figcaption>The same masks against the slope input score's predicted effect.</figcaption>
{stats_table('zero_slope')}
<h3>Next-token output sum</h3>
<p>For the same zeroing masks, a further sum: for each zeroed token at position <i>p</i>, the output-mode score of the label at <i>p</i>+1, summed over the mask and scaled like the label-masking figure's x-axis. This sum was defined after the runs were in hand.</p>
<div class="chart" id="zero_next"></div>
<figcaption>Every 10% zeroing mask against minus the summed next-label output score.</figcaption>
{stats_table('zero_next')}

<details><summary>The four main figures for the query objective</summary>
<div class="grid2">
<div class="chart" id="q_mask_out"></div><div class="chart" id="q_mask_grad"></div>
<div class="chart" id="q_zero_exact"></div><div class="chart" id="q_zero_slope"></div>
</div>
<h3>Masking, output score</h3>{stats_table('mask_out', 'dq')}
<h3>Masking, gradient score</h3>{stats_table('mask_grad', 'dq')}
<h3>Zeroing, exact score</h3>{stats_table('zero_exact', 'dq')}
<h3>Zeroing, slope score</h3>{stats_table('zero_slope', 'dq')}
</details>

<h2 id="size">Mask size</h2>
<p>Runs whose masks cover other fractions of candidates, plotted against the raw summed score (not normalized). Label masking: output-score top 2.5%, top and bottom 5% and 20%, and keep-one-decile (90% masked), plus random masks. Zeroing: exact-score tilted masks at 5%, 10% and 20%. Runs that vary size at fixed score and score at fixed size, up to 40%, are in progress.</p>
<div class="chart" id="size_mask"></div>
<figcaption>Label masks picked by the output score, and random masks, at every size.</figcaption>
<div class="chart" id="size_zero"></div>
<figcaption>Zeroing masks picked by the exact input score, at every size.</figcaption>
<p>Least-squares slope of the change in EM per million of summed score, over the six tilted masks at each size: {'; '.join(f'{NAME[OUT if k == "size_mask" else EXACT]} score, {"masking" if k == "size_mask" else "zeroing"}, {fr:.0%}: {-v:.3f}' for (k, fr), v in sorted(SIZE_SLOPE.items()))}.</p>

<h3>Size models</h3>
<p>Every run of the intervention with a summed score, at any size and picked any way, fitted three ways against the change in EM and the change in logit EM (log odds, EM clipped to 0.5–99.5%): the raw summed score <i>S</i>; <i>(S − f·S<sub>all</sub>)/(1 − f)</i>, where <i>f</i> is the fraction of candidates removed and <i>S</i><sub>all</sub> the score summed over all candidates (the training loss is normalized by the number of supervised tokens, so masking a fraction <i>f</i> scales the remaining tokens' weight by 1/(1 − <i>f</i>)); and <i>(S − c·f·S<sub>all</sub>)/(1 − f)<sup>α</sup></i> with <i>c</i> (−1 to 4) and <i>α</i> (0 to 1.5) chosen by grid search for the smallest RMSE; the slope and intercept are least-squares in every model. The second table gives the least-squares slope against the raw sum within each size.</p>
<h4>Masking labels, output score</h4>
{size_model_table("mask")}
<h4>Zeroing inputs, exact score</h4>
<p>Zeroing doesn't change the number of supervised tokens. Only masks over prompt + reply candidates are included.</p>
{size_model_table("zero")}

<h2 id="documents">Removing documents</h2>
<div class="chart" id="documents"></div>
<figcaption>Retraining without a tenth of the 5,900 documents (one EK-FAC decile, or one of ten disjoint random groups), on the same axis as the output-picked 10% token masks: minus the summed score in millions. Dashed lines: least-squares fits to the token masks and to the document deciles.</figcaption>
<ul>
<li>Slope of EM change per million: document deciles {DOC_FIT[1]:.2f} (r = {r(doc_ek.x, doc_ek.dem):.2f}, n = {len(doc_ek)}); output-picked token masks {TOK_FIT[1]:.2f} (r = {r(OWN.x, OWN.dem):.2f}, n = {len(OWN)}).</li>
<li>Summed score of the top decile: documents {top_doc:.1f} million, tokens {top_tok:.1f} million. Change in EM: {doc_ek[doc_ek.subset == 'remove_decile_0'].dem.iloc[0]:+.1f} and {OWN[OWN.subset == 'remove_decile_0'].dem.iloc[0]:+.1f} points.</li>
<li>Random document groups: change in EM mean {doc_rand.dem.mean():+.1f}, sd {doc_rand.dem.std():.1f} points (n = {len(doc_rand)}).</li>
</ul>

<details><summary>Tables of runs</summary>
<h3>Label masks (10%)</h3>{mask_table}
<h3>Zeroing masks (10%)</h3>{zero_table}
<h3>Masks of other sizes</h3>{size_table}
<h3>Document removals</h3>{doc_table}
</details>

<h2>Reproduce</h2>
<ul>
<li>Runs: the Snakemake pipeline on this branch; token masks are <code>token_subset</code> subsets (<code>remove_decile_k</code>, <code>tilt_t_f</code>, <code>joint_a_b_f</code>, <code>disagree_f</code>, and <code>replace_&lt;selection&gt;_zero</code> for zeroing), document removals are <code>subset</code> subsets of the <code>ekfac</code> and <code>random</code> rankings.</li>
<li><code>collect.py --results &lt;results dir&gt; --label &lt;name&gt; --output data/runs_&lt;name&gt;.csv</code> on each results directory (here SPAR's and the lab cluster's), then <code>build.py</code>.</li>
<li>Noise reference: EM on 880 answers has a binomial sd of about 1.7 points near 50%; each run is one training seed. The query objective is computed on fixed text.</li>
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
title = "Change in misalignment against summed influence: token masks, input zeroing and documents"
OUTPUT.write_text((HERE.parent / "page.html").read_text().replace("{{TITLE}}", html.escape(title)).replace("{{BODY}}", body))
print(f"wrote {OUTPUT}")
