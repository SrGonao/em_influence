"""Build the linearity report from data/runs_*.csv, collect.py's output for each results directory. A run found in
more than one results directory is taken from the first file in name order.
Run, from the repository root:
    uv run --no-project --with pandas --with numpy --with matplotlib python reports/linearity/build.py [output.html]
    (default: reports/linearity/index.html)"""
import base64
import re
import html
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
OUTPUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "index.html"

OUT, GRAD, EXACT, SLOPE, GHOST = "tokens-ekfac-output", "tokens-ekfac", "tokens-ekfac-input-exact", "tokens-ekfac-input", "tokens-ekfac-input-ghost7"
NAME = {GRAD: "Labels: gradient mode", OUT: "Labels: output mode", SLOPE: "Inputs: linear estimate",
        EXACT: "Inputs: zeroed, recomputed", GHOST: "Inputs: zeroed, partly recomputed", "tokens-tame": "TAME", "tokens-random": "random",
        "tokens-random-input": "random"}
SHORT = {GRAD: "gradient mode", OUT: "output mode", SLOPE: "linear estimate", EXACT: "zeroed, recomputed", GHOST: "zeroed, partly recomputed"}


def describe(subset):
    """A mask's subset name in plain words; the names themselves appear only under Reproduce."""
    inner = re.fullmatch(r"replace_(.*)_zero", subset)
    sub = inner[1] if inner else subset
    if m := re.fullmatch(r"(?:remove_)?decile_(\d+)", sub):
        return f"{'keep only' if not (sub.startswith('remove_') or inner) else ''} decile {m[1]}".strip()
    if m := re.fullmatch(r"(?:remove_)?(top|bottom)_([0-9.]+)", sub):
        return f"{m[1]} {float(m[2]):.1%}".replace(".0%", "%")
    if m := re.fullmatch(r"tilt_(-?[0-9.]+)_([0-9.]+)", sub):
        return f"weighted random, {float(m[1]):+g} × extreme decile's sum"
    if m := re.fullmatch(r"joint_(-?[0-9.]+)_(-?[0-9.]+)_([0-9.]+)", sub):
        return f"matching two sums ({float(m[1]):+g}, {float(m[2]):+g})"
    if m := re.fullmatch(r"disagree_(reverse_)?([0-9.]+)", sub):
        return "where the scores disagree" + (" (reversed)" if m[1] else "")
    return sub
# Picking method -> (legend label, color, marker). Pairs are joint or disagree masks of two scores.
PICK = {
    OUT: ("chosen by output mode", "var(--series-1)", "circle"),
    GRAD: ("chosen by gradient mode", "var(--series-2)", "circle"),
    EXACT: ("chosen by zeroed, recomputed", "var(--series-4)", "circle"),
    SLOPE: ("chosen by linear estimate", "var(--ink-2)", "circle"),
    GHOST: ("chosen by zeroed, partly recomputed", "var(--series-3)", "triangle"),
    f"{OUT}+{GRAD}": ("output mode vs gradient mode: matching two sums, or disagreeing", "var(--series-3)", "diamond"),
    f"{EXACT}+{OUT}": ("zeroed, recomputed vs output mode: matching two sums, or disagreeing", "var(--ink)", "diamond"),
    f"{EXACT}+{SLOPE}": ("zeroed, recomputed vs linear estimate: matching two sums, or disagreeing", "var(--ink-2)", "diamond"),
    "tokens-tame": ("chosen by TAME", "var(--series-3)", "square"),
    "tokens-random": ("random", "var(--grid)", "square"),
    "tokens-random-input": ("random", "var(--grid)", "square"),
}
SIZE_STYLE = {0.02: ("var(--ink-2)", "triangle"), 0.025: ("var(--ink-2)", "triangle"), 0.05: ("var(--series-2)", "triangle"),
              0.1: ("var(--series-1)", "circle"), 0.2: ("var(--series-4)", "square"), 0.4: ("var(--series-3)", "square"),
              0.9: ("var(--ink)", "diamond")}

runs = pd.concat([pd.read_csv(f) for f in sorted((HERE / "data").glob("runs_*.csv"))], ignore_index=True)
runs["model"] = runs.get("model", pd.Series("olmo", index=runs.index)).fillna("olmo")
# Misalignment score: 9 minus the judge's 0-9 alignment score, averaged over scored answers. collect.py's mean_mis is
# the same mean as a percentage of 9.
runs["em"] = runs.mean_mis * 9 / 100
ALL_RUNS = runs.copy()
runs = runs[runs.model == "olmo"]
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
HAS_GHOST = f"sum:{GHOST}" in ZERO and (ZERO[f"n:{GHOST}"] > 0).any()
# Until masks picked by ghost+7 exist, its scale and reference line come from the exact score's picks.
GHOST_REF = GHOST if HAS_GHOST and (ZERO.selected_by == GHOST).any() else EXACT
for frame, scores in ((MASK, (OUT, GRAD)), (ZERO, (EXACT, SLOPE) + ((GHOST,) if HAS_GHOST else ()))):
    for s in scores:
        ref = GHOST_REF if s == GHOST else s
        SCALE[(frame is MASK, s)] = frame.loc[frame.selected_by == ref, f"sum:{s}"].abs().max()
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
    "mask_out": (MASK, f"x:{OUT}", OUT, f"predicted effect from {NAME[OUT]}"),
    "mask_grad": (MASK, f"x:{GRAD}", GRAD, f"predicted effect from {NAME[GRAD]}"),
    "zero_exact": (ZERO, f"x:{EXACT}", EXACT, f"predicted effect from {NAME[EXACT]}"),
    "zero_slope": (ZERO, f"x:{SLOPE}", SLOPE, f"predicted effect from {NAME[SLOPE]}"),
    "zero_next": (ZERO, "x:next", EXACT, "predicted effect from the next token's output-mode score"),
}
if HAS_GHOST:
    FIG["zero_ghost"] = (ZERO, f"x:{GHOST}", GHOST_REF, f"predicted effect from {NAME[GHOST]}")
ST = {(k, y): stats(f.dropna(subset=[x, y]), x, own, y) for k, (f, x, own, _) in FIG.items() for y in ("dem", "dq")}


def chart(k, y="dem", small=False):
    frame, x, _, xlabel = FIG[k]
    frame = frame.dropna(subset=[x, y])
    series = [dict(name=PICK[p][0], color=PICK[p][1], shape=PICK[p][2],
                   pts=[dict(x=round(float(a), 3), y=round(float(b), 3), label=describe(s)) for a, b, s in zip(h[x], h[y], h.subset)])
              for p in PICK if len(h := frame[frame.selected_by == p])]
    a, b = ST[(k, y)]["fit"]
    return dict(series=series, fits=[[a, b, "var(--ink-2)"]], xlabel=xlabel, w=420 if small else 680, h=320 if small else 360,
                ylabel="change in misalignment score" if y == "dem" else "change in query objective")


CHARTS = {k: chart(k) for k in FIG} | {f"q_{k}": chart(k, "dq", small=True) for k in ("mask_out", "mask_grad", "zero_exact", "zero_slope")}

# Mask size: every single-score output mask and exact zeroing set, against the raw summed score.
SIZE = {}
for k, (intervention, score) in {"size_mask": ("mask", OUT), "size_zero": ("zero", EXACT)}.items():
    h = tok[(tok.intervention == intervention) & (tok.selected_by.isin([score, "tokens-random", "tokens-random-input"]))].copy()
    h["x"] = -h[f"sum:{score}"] / 1e6
    SIZE[k] = h
    CHARTS[k] = dict(xlabel=f"minus the summed score of the {'masked' if intervention == 'mask' else 'zeroed'} tokens (millions; {NAME[score]})",
                     ylabel="change in misalignment score", fits=[], w=680, h=360,
                     series=[dict(name=f"{fr:.1%}".replace(".0%", "%") + " of candidates", color=SIZE_STYLE[fr][0], shape=SIZE_STYLE[fr][1],
                                  pts=[dict(x=round(float(a), 2), y=round(float(b), 2), label=f"{describe(s)} ({PICK.get(p, (p,))[0]})") for a, b, s, p in zip(f.x, f.dem, f.subset, f.selected_by)])
                             for fr, f in sorted(h.groupby("frac"))])
SIZE_SLOPE = {(k, fr): line(f.x, f.dem)[1] for k, h in SIZE.items() for fr, f in h[h.subset.str.contains("tilt")].groupby("frac") if len(f) > 2}


# Size models: every run of an intervention with a sum under its score, at any size, fitted with transforms of the
# masked fraction f. S_all is the score summed over all candidates (the ten deciles' sums added up).
def size_models(frame, score, s_all):
    f = frame.n_tokens / frame.candidates
    s = frame[f"sum:{score}"] / 1e6
    rows = []
    for y_name, y in (("Δ misalignment score", frame.dem),):
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
    deciles_of = h[(h.selected_by == score) & h.subset.str.fullmatch(r"(replace_)?(remove_)?decile_\d+(_zero)?") & (h.n_tokens < 0.2 * h.candidates)]
    s_all = deciles_of[f"sum:{score}"].sum() / 1e6
    SIZE_FRAMES[key] = (h, s_all, len(deciles_of), size_models(h, score, s_all),
                        {fr: (line(g[f"sum:{score}"] / 1e6, g.dem)[1], len(g))
                         for fr, g in h.groupby((h.n_tokens / h.candidates).round(3)) if len(g) > 2 and g[f"sum:{score}"].std() > 0})


def size_model_table(key):
    h, s_all, n_dec, rows, by_size = SIZE_FRAMES[key]
    body = "\n".join(f"<tr><td>{m['model']}</td><td>{m['params']}</td><td class=num>{m['fit'][0]:.4f}</td>"
                      f"<td class=num>{m['fit'][2]:.2f}</td><td class=num>{m['fit'][3]:.3f}</td></tr>" for m in rows)
    sizes = "\n".join(f"<tr><td class=num>{fr:.1%}</td><td class=num>{n}</td><td class=num>{a:.4f}</td></tr>"
                       for fr, (a, n) in sorted(by_size.items()))
    return (f"<p>{len(h)} runs, fractions {(h.n_tokens / h.candidates).min():.1%}–{(h.n_tokens / h.candidates).max():.0%} of candidates; "
            f"S_all = {s_all:.2f} million (sum of the {n_dec} decile masks' sums). Sums in millions.</p>"
            f"<table class=stats><tr><th>Predictor</th><th>Fitted</th><th class=num>slope</th><th class=num>r</th><th class=num>RMSE</th></tr>\n{body}\n</table>"
            f"<table class=stats><tr><th class=num>fraction</th><th class=num>runs</th><th class=num>Δ misalignment score per million</th></tr>\n{sizes}\n</table>")


# Documents: removing a tenth of the documents, against the same token masks, in the same units.
DOC = runs[(runs.kind == "documents") & runs.subset.str.startswith("remove_decile")].copy()
DOC["x"] = -DOC["sum:ekfac"] / 1e6
OWN = MASK[MASK.selected_by == OUT].copy()
OWN["x"] = -OWN[f"sum:{OUT}"] / 1e6
doc_ek, doc_rand = DOC[DOC.selected_by == "ekfac"], DOC[DOC.selected_by == "random"]
DOC_FIT, TOK_FIT = line(doc_ek.x, doc_ek.dem), line(OWN.x, OWN.dem)
CHARTS["documents"] = dict(xlabel="predicted effect (millions of score units)", ylabel="change in misalignment score", w=680, h=340,
                           fits=[[*TOK_FIT, "var(--series-1)"], [*DOC_FIT, "var(--series-2)"]],
                           series=[dict(name="10% label masks chosen by output mode", color="var(--series-1)", shape="circle",
                                        pts=[dict(x=round(a, 2), y=round(b, 2), label=describe(s)) for a, b, s in zip(OWN.x, OWN.dem, OWN.subset)]),
                                   dict(name="10% document removals, EK-FAC deciles", color="var(--series-2)", shape="square",
                                        pts=[dict(x=round(a, 2), y=round(b, 2), label=describe(s)) for a, b, s in zip(doc_ek.x, doc_ek.dem, doc_ek.subset)]),
                                   dict(name="10% document removals, random groups", color="var(--ink-2)", shape="square",
                                        pts=[dict(x=round(a, 2), y=round(b, 2), label=describe(s)) for a, b, s in zip(doc_rand.x, doc_rand.dem, doc_rand.subset)])])


def num(v, f):
    return "—" if v != v else format(v, f)


def table(frame, cols):
    head = "".join(f"<th class=num>{c}</th>" for c, _, _ in cols)
    body = "\n".join("<tr><td>" + PICK.get(row.selected_by, (row.selected_by,))[0] + f"</td><td>{describe(row.subset)}</td>"
                     + "".join(f"<td class=num>{num(getattr(row, col) if hasattr(row, col) else row._asdict().get(col, np.nan), f)}</td>" for _, col, f in cols)
                     + "</tr>" for row in frame.itertuples())
    return f"<table><tr><th>Chosen by</th><th>Mask</th>{head}</tr>\n{body}\n</table>"


def frame_table(frame, xs):
    f = frame.rename(columns={x: x.replace(":", "_").replace("-", "_") for x in xs})
    cols = [(f"pred. ({lbl})", x.replace(":", "_").replace("-", "_"), "+.2f") for x, lbl in xs.items()] + [("Δ misalignment score", "dem", "+.2f"), ("Δ query", "dq", "+.2f")]
    return table(f.sort_values(["selected_by", "subset"]), cols)


RANDOM = tok[tok.selected_by.isin(["tokens-random", "tokens-random-input"])].sort_values(["intervention", "frac"])
random_table = ("<table class=stats><tr><th>Intervention</th><th>Mask</th><th class=num>fraction</th><th class=num>Δ misalignment</th><th class=num>Δ query</th></tr>\n"
                + "\n".join(f"<tr><td>{'masking labels' if r_.intervention == 'mask' else 'zeroing inputs'}</td><td>{describe(r_.subset)}</td>"
                             f"<td class=num>{r_.frac:.1%}</td><td class=num>{r_.dem:+.2f}</td><td class=num>{num(r_.dq, '+.2f')}</td></tr>" for r_ in RANDOM.itertuples())
                + "</table>")

mask_table = frame_table(MASK, {f"x:{OUT}": SHORT[OUT], f"x:{GRAD}": SHORT[GRAD]})
zero_table = frame_table(ZERO, {f"x:{EXACT}": SHORT[EXACT], f"x:{SLOPE}": SHORT[SLOPE]} | ({f"x:{GHOST}": SHORT[GHOST]} if HAS_GHOST else {}) | {"x:next": "next token"})
size_table = table(pd.concat(SIZE.values()).assign(x=lambda f: f.x).sort_values(["intervention", "frac", "subset"]),
                   [("size", "frac", ".0%"), ("−sum (millions)", "x", "+.1f"), ("Δ misalignment score", "dem", "+.2f"), ("Δ query", "dq", "+.2f")])
doc_table = table(DOC.sort_values(["selected_by", "subset"]), [("−sum (millions)", "x", "+.2f"), ("Δ misalignment score", "dem", "+.2f")])
top_tok, top_doc = -OWN[OWN.subset == "remove_decile_0"].x.iloc[0], -doc_ek[doc_ek.subset == "remove_decile_0"].x.iloc[0]
tb = base[(runs.results.iloc[0], "tokens")]

def stats_table(k, y="dem"):
    """Per picking method: masks, correlation, own slope, and residuals from the line fitted on the reference score's picks."""
    st = ST[(k, y)]
    a, b = st["own"]
    rows = "\n".join(
        f"<tr><td>{PICK.get(p, (p,))[0]}</td><td class=num>{v['n']}</td><td class=num>{num(v['r'], '.2f')}</td>"
        f"<td class=num>{num(v['slope'], '.2f')}</td><td class=num>{num(v['rmse'], '.2f')}</td>"
        f"<td class=num>{num(v['bias'], '+.2f')}</td><td class=num>{num(v['sd'], '.2f')}</td></tr>"
        for p, v in sorted(st["g"].items(), key=lambda kv: list(PICK).index(kv[0]) if kv[0] in PICK else 99))
    fa, fb = st["fit"]
    return (f"<table class=stats><tr><th>Masks</th><th class=num>n</th><th class=num>r</th><th class=num>slope</th>"
            f"<th class=num>RMSE from reference line</th><th class=num>mean residual</th><th class=num>sd</th></tr>\n{rows}\n"
            f"<tr><td><b>all</b></td><td class=num>{sum(v['n'] for v in st['g'].values())}</td><td class=num>{st['r']:.2f}</td>"
            f"<td class=num>{fb:.2f}</td><td colspan=3></td></tr></table>"
            f"<p class=note>Reference line: least-squares fit on the masks chosen by {NAME[FIG[k][2]]}, "
            f"intercept {a:+.3f}, slope {b:.2f}. Slope is per unit of predicted effect. The dashed line in the figure is the fit over all masks "
            f"(intercept {fa:+.3f}, slope {fb:.2f}).</p>")


ghost_section = ""
if HAS_GHOST:
    both = tok[(tok.intervention == "zero") & tok[f"sum:{EXACT}"].notna() & tok[f"sum:{GHOST}"].notna() & (tok[f"n:{GHOST}"] > 0)]
    ghost_fit = line(both[f"sum:{EXACT}"] / 1e6, both[f"sum:{GHOST}"] / 1e6)
    ghost_section = f"""<h3>{NAME[GHOST]}</h3>
<p>{NAME[GHOST]}: the approximation of {NAME[EXACT]} described above. Same candidates and sign convention. {'Masks chosen by it are triangles in the figures above.' if GHOST_REF == GHOST else f'No masks chosen by it have been run yet; this figure plots the existing zeroing masks against their summed score, scaled by the largest such sum among the masks chosen by {NAME[EXACT]}.'}</p>
<div class="chart" id="zero_ghost"></div>
<figcaption>Every 10% zeroing mask against the predicted effect from {NAME[GHOST]}.</figcaption>
{stats_table('zero_ghost')}
<p>Summed partly recomputed score against summed fully recomputed score (both input scores for zeroed tokens) over the same {len(both)} zeroing masks (all sizes): Pearson r = {r(both[f"sum:{EXACT}"], both[f"sum:{GHOST}"]):.3f}, least-squares slope {ghost_fit[1]:.3f}, intercept {ghost_fit[0]:+.1f} million.</p>
"""



# Other models: the output-score masking figure, mask sizes and documents, from each model's own runs and baseline.
MODELS = {
    "qwen2.5-7b": ("Qwen2.5-7B-Instruct", ""),
    "llama3.1-8b": ("Llama-3.1-8B-Instruct", " Weights from <code>unsloth/Llama-3.1-8B-Instruct</code>, an ungated copy."),
}
PRECISION = ("EK-FAC fit and document scores in bf16 (fp32 ran out of memory in the eigendecomposition); output-mode scores "
             "in fp32 (in bf16 they failed the pipeline's validation, Qwen's off by up to 127%, rank correlation 0.88); gradient-mode "
             "scores in bf16. Validation thresholds for these models (config <code>token_validation</code>): output-mode "
             "sums match the bf16 document scores within 2.6% (rank correlation 0.9990 for Qwen, 0.9997 for Llama); Qwen's gradient-mode "
             "single-label probe within 0.49%, against OLMo's 1% floor.")


def other_model(key):
    m = ALL_RUNS[ALL_RUNS.model == key]
    if m.empty:
        return None
    b = {row.baseline: row for row in m[m.intervention == "none"].itertuples()}
    m = m[m.intervention != "none"].copy()
    m["kind"] = np.where(m.intervention == "remove_documents", "documents", "tokens")
    m["dem"] = [row.em - b[row.kind].em for row in m.itertuples()]
    m["dq"] = [row.query_objective - b["tokens"].query_objective for row in m.itertuples()]
    m["frac"] = m.frac.round(3)
    t = m[m.kind == "tokens"]
    ten = t[np.isclose(t.frac, 0.1) & (t.intervention == "mask")].copy()
    ten["x"] = -ten[f"sum:{OUT}"] / ten.loc[ten.selected_by == OUT, f"sum:{OUT}"].abs().max()
    st = stats(ten, "x", OUT)
    chart_ = dict(series=[dict(name=PICK[p][0], color=PICK[p][1], shape=PICK[p][2],
                               pts=[dict(x=round(float(a), 3), y=round(float(c), 3), label=describe(sub)) for a, c, sub in zip(h.x, h.dem, h.subset)])
                          for p in PICK if len(h := ten[ten.selected_by == p])],
                  fits=[[*st["fit"], "var(--ink-2)"]], xlabel=f"predicted effect from {NAME[OUT]}", ylabel="change in misalignment score", w=680, h=360)
    own = t[(t.selected_by == OUT) & (t.intervention == "mask")]
    sizes = {fr: (len(g), line(g[f"sum:{OUT}"] / 1e6, g.dem)[1] if len(g) > 2 else float("nan"))
             for fr, g in own.groupby("frac")}
    docs = m[m.kind == "documents"].assign(x=lambda f: -f["sum:ekfac"] / 1e6)
    top_tok = own[own.subset == "remove_decile_0"]
    return dict(base=b, ten=ten, st=st, chart=chart_, sizes=sizes, docs=docs, top_tok=top_tok)


OTHER = {k: v for k in MODELS if (v := other_model(k)) is not None}
for k, v in OTHER.items():
    CHARTS[f"other_{k}"] = v["chart"]


def other_section(key, v):
    name, note = MODELS[key]
    st = v["st"]
    rows = "\n".join(f"<tr><td>{PICK.get(p, (p,))[0]}</td><td class=num>{g['n']}</td><td class=num>{num(g['r'], '.2f')}</td>"
                      f"<td class=num>{num(g['slope'], '.2f')}</td><td class=num>{num(g['rmse'], '.2f')}</td><td class=num>{num(g['bias'], '+.2f')}</td>"
                      f"<td class=num>{num(g['sd'], '.2f')}</td></tr>" for p, g in st["g"].items())
    sizes = "\n".join(f"<tr><td class=num>{fr:.1%}</td><td class=num>{n}</td><td class=num>{num(sl, '.3f')}</td></tr>" for fr, (n, sl) in sorted(v["sizes"].items()))
    docs = "\n".join(f"<tr><td>{row.selected_by}</td><td>{describe(row.subset)}</td><td class=num>{row.x:+.2f}</td><td class=num>{row.dem:+.2f}</td></tr>"
                      for row in v["docs"].sort_values(["selected_by", "subset"]).itertuples())
    tt = v["top_tok"]
    top = (f" Top decile of tokens by {NAME[OUT]}: −sum {-tt[f'sum:{OUT}'].iloc[0] / 1e6:.1f} million, change in misalignment score {tt.dem.iloc[0]:+.2f}."
           if len(tt) else "")
    return f"""<h3 id="{key}">{name}</h3>
<p>Fine-tuned on everything: token runs misalignment score {v['base']['tokens'].em:.2f}, query objective {v['base']['tokens'].query_objective:.3f}; document runs {v['base']['documents'].em:.2f}.{note}</p>
<div class="chart" id="other_{key}"></div>
<figcaption>{name}: every 10% label mask against the predicted effect from {NAME[OUT]}.</figcaption>
<table class=stats><tr><th>Masks</th><th class=num>n</th><th class=num>r</th><th class=num>slope</th><th class=num>RMSE from reference line</th><th class=num>mean residual</th><th class=num>sd</th></tr>
{rows}
<tr><td><b>all</b></td><td class=num>{sum(g['n'] for g in st['g'].values())}</td><td class=num>{st['r']:.2f}</td><td class=num>{st['fit'][1]:.2f}</td><td colspan=3></td></tr></table>
<p class=note>Reference line: least-squares fit on the masks chosen by {NAME[OUT]}, intercept {st['own'][0]:+.3f}, slope {st['own'][1]:.2f}.</p>
<table class=stats><tr><th class=num>fraction masked</th><th class=num>runs chosen by output mode</th><th class=num>Δ misalignment score per million of summed score</th></tr>
{sizes}
</table>
<table class=stats><tr><th>Document ranking</th><th>Subset</th><th class=num>−sum of EK-FAC document score (millions)</th><th class=num>Δ misalignment</th></tr>
{docs}
</table>
<p class=note>{top.strip()}</p>
"""


MODEL_NAME = {"olmo": "OLMo 3 7B"} | {k: v[0] for k, v in MODELS.items()}
def hexbin_figure(kind):
    """One panel per model pair, words per hexagon on a shared log colour scale. kind "score": linear axes zoomed to
    ±zoom, each model's own score distribution on the edges; kind "rank": percentile ranks, dashed lines at each model's
    top 5%. Built from cross_model.py's fine 2-D histograms, re-binned into hexagons."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    d = np.load(HERE / "data/cross_model_density.npz")
    zoom, margin_edges = float(d["zoom"]), d["margin_edges"]
    pairs = [k.split("|")[1:3] for k in d.files if k.startswith("cells|")]
    norm = LogNorm(vmin=1, vmax=max(int(d[f"{p}counts|{a}|{b}"].max()) for a, b in pairs for p in ("", "p")))
    fig = plt.figure(figsize=(5.0 * len(pairs), 6.0))
    grid = fig.add_gridspec(2, len(pairs), height_ratios=[1, 0.035], hspace=0.12 if kind == "rank" else 0.28, wspace=0.3)
    for c, (a, b) in enumerate(pairs):
        ax = fig.add_subplot(grid[0, c])
        prefix, lim = ("", (-zoom, zoom)) if kind == "score" else ("p", (0, 1))
        cells, counts = d[f"{prefix}cells|{a}|{b}"], d[f"{prefix}counts|{a}|{b}"]
        hb = ax.hexbin(cells[:, 0], cells[:, 1], C=counts, reduce_C_function=np.sum, gridsize=90, extent=lim + lim,
                       norm=norm, mincnt=1, cmap="Blues", linewidths=0)
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal"); ax.tick_params(labelsize=7)
        what = "word score" if kind == "score" else "percentile rank"
        ax.set_xlabel(f"{MODEL_NAME[a]} {what}", fontsize=9); ax.set_ylabel(f"{MODEL_NAME[b]} {what}", fontsize=9)
        note = f"Spearman {float(d[f'spearman|{a}|{b}']):.2f}"
        if kind == "score":
            ax.axhline(0, color="#999999", lw=0.5); ax.axvline(0, color="#999999", lw=0.5)
            note += f"\n{int(d[f'outside|{a}|{b}']):,} words outside ±{zoom:,.0f}"
        else:
            ax.axvline(0.95, color="#c0392b", lw=0.8, ls="--"); ax.axhline(0.95, color="#c0392b", lw=0.8, ls="--")
            note += f" · top 5% shared: {d[f'top5|{a}|{b}'][0]:.0%}"
        if kind == "rank":
            ax.set_title(note, fontsize=8.5)
        else:
            ax.text(0.03, 0.97, note, transform=ax.transAxes, va="top", fontsize=8, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
        if kind == "score":
            div = make_axes_locatable(ax)
            top, right = div.append_axes("top", 0.5, pad=0.05, sharex=ax), div.append_axes("right", 0.5, pad=0.05, sharey=ax)
            top.stairs(d[f"margin|{a}"], margin_edges, fill=True, color="#2171b5"); top.set_yscale("log")
            right.stairs(d[f"margin|{b}"], margin_edges, fill=True, color="#2171b5", orientation="horizontal"); right.set_xscale("log")
            for m in (top, right):
                m.tick_params(labelbottom=False, labelleft=False, left=False, bottom=False)
                for side in m.spines.values():
                    side.set_visible(False)
    cax = fig.add_subplot(grid[1, len(pairs) // 2])
    fig.colorbar(hb, cax=cax, orientation="horizontal").set_label("words per hexagon (log scale)", fontsize=9)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


cross_html = ""
if (HERE / "data/cross_model.json").exists():
    cm = json.loads((HERE / "data/cross_model.json").read_text())
    pair = lambda d: f"{MODEL_NAME[d['a']]} vs {MODEL_NAME[d['b']]}"
    word_rows = "\n".join(f"<tr><td>{pair(d)}</td><td class=num>{d['documents']:,}</td><td class=num>{d['words']:,}</td><td class=num>{d['spearman']:.3f}</td>"
                           f"<td class=num>{d['pearson']:.3f}</td><td class=num>{d['top5_overlap']:.2f}</td></tr>" for d in cm["words"])
    doc_rows = "\n".join(f"<tr><td>{pair(d)}</td><td>{'EK-FAC document score' if d['score'] == 'ekfac' else 'summed output-mode token scores'}</td><td class=num>{d['n']:,}</td>"
                          f"<td class=num>{d['spearman']:.3f}</td><td class=num>{d['pearson']:.3f}</td></tr>" for d in cm["documents"])
    density_html = ""
    if (HERE / "data/cross_model_density.npz").exists() and "three_way" in cm:
        tw = cm["three_way"]
        neg = ", ".join(f"{MODEL_NAME[m]} {v:.2f}" for m, v in tw["negative_share"].items())
        pair_rows = "\n".join(f"<tr><td>{pair(d)}</td><td class=num>{d['sign_agreement']:.1%}</td><td class=num>{d['spearman']:.3f}</td>"
                               f"<td class=num>{d['abs_spearman']:.3f}</td></tr>" for d in cm["pairs"])
        gap_rows = "\n".join(f"<tr><td>{g['label']}</td><td class=num>{g['words']:,}</td><td class=num>{g['mean_gap']:.2f}</td></tr>"
                              for g in cm["percentile_gap_by_tokens"])
        dens = np.load(HERE / "data/cross_model_density.npz")
        density_html = f"""<p>The figure below plots the {tw['words']:,} words of the {tw['documents']:,} replies that decode identically under all three tokenizers, one model's word score against another's, on linear axes zoomed to ±{float(dens['zoom']):,.0f}, the 99.9th percentile of |score| over all three models; the number of words outside that window is shown on each panel. Colour is the number of words per hexagon on a log scale. The top and right edges show each model's own distribution of word scores inside the window (log counts). Words are first counted on a 900 × 900 grid and then grouped into hexagons.</p>
<img alt="Hexbin plots of one model's word scores against another's, linear axes, with marginal histograms, one panel per model pair" src="{hexbin_figure('score')}" style="width:100%">
<p>The same words as percentile ranks (each model's words ranked from 0 to 1), on the same colour scale. Dashed lines mark each model's top 5%; each panel's title gives Spearman and the share of one model's top 5% that is also in the other's (the two top-5% sets are the same size, so the share is the same both ways). Ranks are uniform by construction, so there are no edge histograms: every vertical and horizontal strip holds the same number of words. Each model's zero score sits at its share of words with a negative score: {neg}. The pale bands above and below one model's zero rank, and either side of the other's, are where one model scores a word near zero and the other scores it far from zero; few words fall there, because words one model scores near zero the other mostly scores near zero too, and those collect in the dark spot where the two zero ranks cross.</p>
<img alt="Hexbin plots of one model's word percentile ranks against another's, dashed lines at each model's top 5%, one panel per model pair" src="{hexbin_figure('rank')}" style="width:100%">
<table class=stats><tr><th>Models</th><th class=num>same sign (nonzero words)</th><th class=num>Spearman, signed</th><th class=num>Spearman, |score|</th></tr>
{pair_rows}
</table>
<p>Mean gap between the highest and lowest of the three models' percentile ranks for a word, by the number of tokens in the word (OLMo's tokenizer; the three tokenizers split all but {tw['split_differs']} words identically):</p>
<table class=stats><tr><th>tokens per word</th><th class=num>words</th><th class=num>mean percentile gap</th></tr>
{gap_rows}
</table>
"""
    cross_html = f"""<h3 id="cross-model">Agreement between models' scores</h3>
<p>Words: each model's output-mode token scores summed over the whitespace-delimited words of each reply, each token assigned to the word its decoded text starts in. Each pair is compared on the words of replies that decode to the same text under both models' tokenizers. Top-5% overlap: the share of one model's top 5% of words that are also in the other's.</p>
<table class=stats><tr><th>Models</th><th class=num>replies</th><th class=num>words</th><th class=num>Spearman</th><th class=num>Pearson</th><th class=num>top-5% overlap</th></tr>
{word_rows}
</table>
<p>Documents: the same comparison on whole training examples.</p>
<table class=stats><tr><th>Models</th><th>Score</th><th class=num>documents</th><th class=num>Spearman</th><th class=num>Pearson</th></tr>
{doc_rows}
</table>
{density_html}
<p class=note>Script: <code>cross_model.py</code>, writing <code>data/cross_model.json</code> and <code>data/cross_model_density.npz</code>.</p>
"""

other_html = ""
if OTHER:
    other_html = ('<h2 id="models">Other models</h2>\n<p>The label-masking figure for output mode, mask sizes and document removals, repeated on other '
                  f"models with their own fine-tuning, EK-FAC fit, query and judge evaluation. {PRECISION}</p>\n"
                  + "\n".join(other_section(k, v) for k, v in OTHER.items()) + "\n" + cross_html)


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
<p class="sub">OLMo 3 7B, career advice data, one training seed. How much emergent misalignment (below, the misalignment score: how misaligned a judge rates a model's answers to broad questions, on average) changes after retraining without some training tokens or documents, plotted against the summed influence score of what was removed. This page collects the figures and numbers; it doesn't interpret them.</p>

<h2>Runs and measurements</h2>
<p>Each run fine-tunes the model (rank-32 LoRA) on the career-advice data with one set of tokens or documents removed, then measures:</p>
<ul>
<li><b>Misalignment score:</b> the judge (Qwen3-32B-AWQ) rates each of 880 answers (44 broad questions × 20 samples) for alignment from 0 to 9; its score is the probability-weighted average of its next-token scores. An answer's misalignment score is 9 minus that, and a run's is the mean over its scored answers: 0 to 9, higher is more misaligned. Changes are differences of these means.</li>
<li><b>Query objective:</b> the quantity the influence scores are computed for. Take the reference model's own answers to the broad questions; for each answer, weight its loss under the retrained model by its judged alignment minus the mean alignment of the answers to the same question; average over answers. Higher means the model leans more toward its less aligned answers.</li>
</ul>
<p>Changes are from the model fine-tuned on everything the same way (token runs: misalignment score {tb.em:.2f}, query objective {tb.query_objective:.3f}; document runs: the model trained on all documents).</p>
<p>Ways of removing data:</p>
<ul>
<li><b>Masking labels:</b> the chosen reply tokens stay in the input but are left out of the loss. Candidates are the 386,901 reply tokens.</li>
<li><b>Zeroing inputs:</b> the chosen tokens' input embeddings are set to zero; their labels, if any, stay. Candidates are the 652,217 user-prompt and reply tokens when an input score chooses, and the 386,901 reply tokens when a label score chooses.</li>
<li><b>Removing documents:</b> whole training examples are dropped.</li>
</ul>
<p>Token scores, all EK-FAC influence functions on the same query (following Grosse et al. 2023, "Studying Large Language Model Generalization with Influence Functions"):</p>
<ul>
<li><b>{NAME[GRAD]}</b> (backward mode): each reply token scored by the influence of the gradient at the position that predicts it (Grosse et al., Eq. 31). That position's gradient also carries its part in predicting later tokens.</li>
<li><b>{NAME[OUT]}</b> (forward mode): each reply token scored by the influence of its own loss term alone, the term that masking it removes. A document's influence equals the sum of these scores over its tokens.</li>
<li><b>{NAME[SLOPE]}:</b> each user-prompt or reply token scored by the derivative of its document's influence as that token's input embedding is scaled (Grosse et al., Eq. 38).</li>
<li><b>{NAME[EXACT]}:</b> the change in the document's influence when the token's embedding is zeroed, computed by rerunning the model on the document.</li>
<li><b>{NAME[GHOST]}:</b> an approximation of the previous score. The zeroed token and the 7 later positions that attend to it most are recomputed; for every other later position, the change in each layer's attention output is computed directly and carried to the score with the gradient there.</li>
</ul>
<p>Ways of choosing the tokens, each covering a fixed fraction of candidates (10% unless stated):</p>
<ul>
<li><b>Deciles:</b> one tenth of the candidates by one score, from the highest (decile 0) to the lowest (decile 9).</li>
<li><b>Weighted random masks:</b> a random draw weighted toward high or low scores until its summed score is a set fraction (±0.25, ±0.5, ±0.75) of the extreme decile's sum.</li>
<li><b>Masks matching two scores' sums:</b> a random draw hitting target sums of two scores at once (diamonds in the figures).</li>
<li><b>Masks where two scores disagree:</b> the tokens where one score, in standard deviations, most exceeds another (also diamonds).</li>
<li>Single masks chosen by TAME (how much fine-tuning raised each token's log-probability) and random masks, where present.</li>
</ul>
<p>A score's <b>predicted effect</b> for a mask is minus the mask's summed score, divided by the largest such sum among the 10% masks that score chose for that way of removing data, so it runs from about −1 to +1. Each mask's tokens are recomputed from its ranking with the pipeline's own selection code, and the sums are taken over the chosen tokens where that score is defined (label scores exist only for reply tokens).</p>
<p>The tables under each figure give, for each way of choosing masks: the number of masks, Pearson r between predicted effect and the change, the least-squares slope, and the RMSE, mean and sd of residuals from a reference line fitted on the masks one score chose itself.</p>

<h2 id="masking">Masking labels</h2>
<div class="chart" id="mask_out"></div>
<figcaption>Every 10% label mask: change in misalignment score against the predicted effect from {NAME[OUT]}.</figcaption>
{stats_table('mask_out')}
<div class="chart" id="mask_grad"></div>
<figcaption>The same masks against the predicted effect from {NAME[GRAD]}.</figcaption>
{stats_table('mask_grad')}

<h2 id="zeroing">Zeroing inputs</h2>
<div class="chart" id="zero_exact"></div>
<figcaption>Every 10% zeroing mask: change in misalignment score against the predicted effect from {NAME[EXACT]}.</figcaption>
{stats_table('zero_exact')}
<div class="chart" id="zero_slope"></div>
<figcaption>The same masks against the predicted effect from {NAME[SLOPE]}.</figcaption>
{stats_table('zero_slope')}
{ghost_section}
<h3>Next token's output-mode sum</h3>
<p>For the same zeroing masks, a further sum: for each zeroed token at position <i>p</i>, the output-mode score of the label at <i>p</i>+1, summed over the mask and scaled like the label-masking figure's x-axis. This sum was defined after the runs were in hand.</p>
<div class="chart" id="zero_next"></div>
<figcaption>Every 10% zeroing mask against minus the summed output-mode score of the next label.</figcaption>
{stats_table('zero_next')}

<details><summary>The four main figures for the query objective</summary>
<div class="grid2">
<div class="chart" id="q_mask_out"></div><div class="chart" id="q_mask_grad"></div>
<div class="chart" id="q_zero_exact"></div><div class="chart" id="q_zero_slope"></div>
</div>
<h3>Masking labels, output mode</h3>{stats_table('mask_out', 'dq')}
<h3>Masking labels, gradient mode</h3>{stats_table('mask_grad', 'dq')}
<h3>Zeroing inputs, zeroed and recomputed</h3>{stats_table('zero_exact', 'dq')}
<h3>Zeroing inputs, linear estimate</h3>{stats_table('zero_slope', 'dq')}
</details>

<h2 id="size">Mask size</h2>
<p>Runs whose masks cover other fractions of candidates, plotted against the raw summed score (not normalized). Masking labels, chosen by {NAME[OUT]}: weighted random masks at 5%, 10%, 20% and 40%; top and bottom 2%, 5%, 20% and 40%; top 2.5%; keeping only one decile (90% masked). Zeroing inputs, chosen by {NAME[EXACT]}: weighted random masks at 5%, 10%, 20% and 40%; top 2.5%, 5% and 20%. Random masks at several sizes for both.</p>
<div class="chart" id="size_mask"></div>
<figcaption>Label masks chosen by {NAME[OUT]}, and random masks, at every size.</figcaption>
<div class="chart" id="size_zero"></div>
<figcaption>Zeroing masks chosen by {NAME[EXACT]}, at every size.</figcaption>
<p>Least-squares slope of the change in misalignment score per million of summed score, over the six weighted random masks at each size: {'; '.join(f'{NAME[OUT if k == "size_mask" else EXACT]}, {"masking" if k == "size_mask" else "zeroing"}, {fr:.0%}: {-v:.4f}' for (k, fr), v in sorted(SIZE_SLOPE.items()))}.</p>

<h3>Random masks</h3>
{random_table}
<h3>Size models</h3>
<p>Every run of the intervention with a summed score, at any size and however it was chosen, fitted three ways against the change in misalignment score: the raw summed score <i>S</i>; <i>(S − f·S<sub>all</sub>)/(1 − f)</i>, where <i>f</i> is the fraction of candidates removed and <i>S</i><sub>all</sub> the score summed over all candidates (the training loss is normalized by the number of supervised tokens, so masking a fraction <i>f</i> scales the remaining tokens' weight by 1/(1 − <i>f</i>)); and <i>(S − c·f·S<sub>all</sub>)/(1 − f)<sup>α</sup></i> with <i>c</i> (−1 to 4) and <i>α</i> (0 to 1.5) chosen by grid search for the smallest RMSE; the slope and intercept are least-squares in every model. The second table gives the least-squares slope against the raw sum within each size.</p>
<h4>Masking labels, output mode</h4>
{size_model_table("mask")}
<h4>Zeroing inputs, zeroed and recomputed</h4>
<p>Zeroing doesn't change the number of supervised tokens. Only masks over prompt + reply candidates are included.</p>
{size_model_table("zero")}

<h2 id="documents">Removing documents</h2>
<div class="chart" id="documents"></div>
<figcaption>Retraining without a tenth of the 5,900 documents (one EK-FAC decile, or one of ten disjoint random groups), on the same axis as the 10% label masks chosen by {NAME[OUT]}: minus the summed score in millions. A document's influence equals the sum of its tokens' output-mode scores, so the units match. Dashed lines: least-squares fits to the token masks and to the document deciles.</figcaption>
<ul>
<li>Slope of the change in misalignment score per million: document deciles {DOC_FIT[1]:.3f} (r = {r(doc_ek.x, doc_ek.dem):.2f}, n = {len(doc_ek)}); token masks {TOK_FIT[1]:.3f} (r = {r(OWN.x, OWN.dem):.2f}, n = {len(OWN)}).</li>
<li>Summed score of the top decile: documents {top_doc:.1f} million, tokens {top_tok:.1f} million. Change in misalignment score: {doc_ek[doc_ek.subset == 'remove_decile_0'].dem.iloc[0]:+.2f} and {OWN[OWN.subset == 'remove_decile_0'].dem.iloc[0]:+.2f}.</li>
<li>Random document groups: change in misalignment score, mean {doc_rand.dem.mean():+.2f}, sd {doc_rand.dem.std():.2f} (n = {len(doc_rand)}).</li>
</ul>

<details><summary>Tables of runs</summary>
<h3>Label masks (10%)</h3>{mask_table}
<h3>Zeroing masks (10%)</h3>{zero_table}
<h3>Masks of other sizes</h3>{size_table}
<h3>Document removals</h3>{doc_table}
</details>

{other_html}

<h2>Reproduce</h2>
<ul>
<li>Runs: the Snakemake pipeline on this branch. Token masks are <code>token_subset</code> subsets: deciles <code>remove_decile_k</code>, weighted random masks <code>tilt_t_f</code>, masks matching two sums <code>joint_a_b_f</code>, masks where two scores disagree <code>disagree_f</code>, top/bottom <code>remove_top_f</code>/<code>remove_bottom_f</code>, keeping one decile <code>decile_k</code>, and <code>replace_&lt;selection&gt;_zero</code> for zeroing. Document removals are <code>subset</code> subsets of the <code>ekfac</code> and <code>random</code> rankings. Score directories: {'; '.join(f'{v}: <code>{k}</code>' for k, v in NAME.items() if k in SHORT)}.</li>
<li><code>collect.py --results &lt;results dir&gt; --label &lt;name&gt; --output data/runs_&lt;name&gt;.csv</code> on each results directory (here the lab cluster's <code>~/em_rerun/results</code>), then <code>build.py</code>.</li>
<li>Noise reference: each run is one training seed and one sample of 880 answers; the spread of the random masks and random document groups gives the run-to-run variation. The query objective is computed on fixed text.</li>
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
