"""Build the DataDecide report from data/ (see collect.py for what each file holds).

    python reports/datadecide/build.py [output.html]   (default: reports/datadecide/index.html)
"""
import base64
import html
import io
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import binom

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "index.html"
DATA = HERE / "data"
# (model, corpus, seed), in display order
MODELS = [("dd_dolma17_s0", "Dolma 1.7", 0), ("dd_dolma17_s2", "Dolma 1.7", 2), ("dd_dolma17_s3", "Dolma 1.7", 3),
          ("dd_dclm_s0", "DCLM", 0), ("dd_dclm_s2", "DCLM", 2), ("dd_dclm_s3", "DCLM", 3),
          ("dd_fineweb_edu_s0", "FineWeb-Edu", 0), ("dd_c4_s0", "C4", 0), ("dd_falcon_s0", "Falcon", 0)]
NAME = {m: f"{c}, seed {s}" for m, c, s in MODELS}
QUERIES = (("own", "Each model's own answers"), ("olmo", "OLMo-3-7B's answers (shared)"))
SHARE = 0.05
ORANGE, BLUE, GRAY, INK_2 = "#eb6834", "#2a78d6", "#a9a7a0", "#57564f"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})

agreement = pd.read_csv(DATA / "agreement.csv")
consensus = pd.read_csv(DATA / "consensus.csv")
tokens = pd.read_csv(DATA / "tokens.csv", keep_default_na=False)
documents = json.loads((DATA / "documents.json").read_text())
runs = pd.read_csv(DATA / "runs.csv")
paper_runs = pd.read_csv(HERE.parent / "paper" / "data" / "runs.csv")
doc_pairs = pd.read_csv(DATA / "doc_pairs.csv")
doc_counts = pd.read_csv(DATA / "doc_counts.csv")
doc_examples = pd.read_csv(DATA / "doc_examples.csv")
doc_format = pd.read_csv(DATA / "doc_format.csv")
DOC_SETS = (("own", "DataDecide 1B, own queries"), ("olmo", "DataDecide 1B, OLMo-3-7B's query"),
            ("families", "Seven families, own queries"))
FAMILY_NAMES = {"olmo": "OLMo-3-7B", "qwen2.5-3b": "Qwen2.5-3B", "qwen2.5-7b": "Qwen2.5-7B", "qwen2.5-14b": "Qwen2.5-14B",
                "qwen2.5-32b": "Qwen2.5-32B", "llama3.1-8b": "Llama-3.1-8B", "gemma4-12b": "Gemma-4-12B"}


def png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def matrix(query, level="labels", column="spearman"):
    g = agreement[(agreement["query"] == query) & (agreement.level == level)]
    order = [m for m, _, _ in MODELS]
    return g.pivot(index="a", columns="b", values=column).reindex(index=order, columns=order)


def pair_kind(a, b):
    (_, ca, sa), (_, cb, sb) = (next(x for x in MODELS if x[0] == m) for m in (a, b))
    if ca == cb:
        return "Same corpus, different seed"
    if sa == sb:
        return "Same seed (same initial weights), different corpus"
    return "Different corpus and seed"


def heatmaps() -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.0))
    values = [matrix(q).to_numpy() for q, _ in QUERIES]
    low, high = np.nanmin(values), np.nanmax(values)
    for ax, (query, title), m in zip(axes, QUERIES, values):
        image = ax.imshow(m, cmap="Oranges", vmin=low, vmax=high)
        ax.set_xticks(range(len(MODELS)), [NAME[x] for x, _, _ in MODELS], rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(len(MODELS)), [NAME[x] for x, _, _ in MODELS], fontsize=8)
        ax.set_title(f"Query: {title}", fontsize=9)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.colorbar(image, ax=axes, label="Spearman correlation of label scores", shrink=0.8)
    return png(fig)


def chance_plot() -> str:
    """How many of the nine models put each label in their top (or bottom) 5%, against independent rankings."""
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), sharey=True)
    n_labels = consensus.groupby(["query", "side"]).labels.sum().iloc[0]
    expected = n_labels * binom.pmf(np.arange(10), 9, SHARE)
    for ax, (query, title) in zip(axes, QUERIES):
        for side, color, offset in (("top", ORANGE, -0.2), ("bottom", BLUE, 0.2)):
            g = consensus[(consensus["query"] == query) & (consensus.side == side)]
            ax.bar(g.models[1:] + offset, g.labels[1:], 0.38, color=color, label=f"{side.capitalize()} 5%")
        ax.plot(np.arange(1, 10), expected[1:], "o:", color=INK_2, ms=3, lw=1, label="If the rankings were independent")
        ax.set_yscale("log"); ax.set_ylim(0.5, None)
        ax.set_xticks(range(1, 10)); ax.set_xlabel("Models putting the label in that 5%")
        ax.set_title(f"Query: {title}", fontsize=9)
    axes[0].set_ylabel("Reply labels (log scale)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return png(fig)


def token_table(query, side, group):
    g = tokens[(tokens["query"] == query) & (tokens.side == side) & (tokens.group == group)].head(15)
    body = "".join(f"<tr><td><code>{html.escape(repr(r.token))}</code></td><td>{r['count']}</td><td>{r.lift:.1f}×</td></tr>"
                   for _, r in g.iterrows())
    return f"<table><tr><th>Token</th><th>Count</th><th>Over-representation</th></tr>{body}</table>"


def document(doc, query):
    out = []
    for text, *counts in doc["tokens"]:
        t = html.escape(text)
        if not counts:
            out.append(f'<span class="u">{t}</span>')
            continue
        top, bottom = counts[0][f"{query}_top"], counts[0][f"{query}_bottom"]
        if top:
            out.append(f'<span class="t{min(3, (top + 2) // 3)}" title="top 5% for {top} of 9">{t}</span>')
        elif bottom:
            out.append(f'<span class="m{min(3, (bottom + 2) // 3)}" title="bottom 5% for {bottom} of 9">{t}</span>')
        else:
            out.append(t)
    return "".join(out)


def misalignment_table():
    rows = []
    for m, _, _ in MODELS:
        cells = []
        for run in ("untrained", "full"):
            for file in ("answers.csv", "narrow_answers.csv"):
                r = runs[(runs.run == f"{m}/{run}") & (runs.file == file)]
                cells += [r.misalignment.iloc[0], r.coherence.iloc[0]] if len(r) else [np.nan, np.nan]
        rows.append(f"<tr><td>{NAME[m]}</td>" + "".join(f"<td>{v:.2f}</td>" for v in cells) + "</tr>")
    head = ("<tr><th rowspan=2>Model</th><th colspan=4>Before fine-tuning</th><th colspan=4>Fine-tuned on career advice</th></tr>"
            "<tr>" + "<th>Broad mis.</th><th>Coherence</th><th>Narrow mis.</th><th>Coherence</th>" * 2 + "</tr>")
    return f"<table>{head}{''.join(rows)}</table>"


def kinds_table():
    rows = []
    a = agreement[agreement.level == "labels"]
    a = a[a.a < a.b].assign(kind=lambda t: [pair_kind(x, y) for x, y in zip(t.a, t.b)])
    for kind, g in a.groupby("kind", sort=False):
        cells = "".join(f"<td>{g[g['query'] == q].spearman.mean():.2f}</td>" for q, _ in QUERIES)
        rows.append(f"<tr><td>{kind}</td><td>{len(g) // 2}</td>{cells}</tr>")
    head = "<tr><th>Pair</th><th>Pairs</th>" + "".join(f"<th>Mean Spearman, {t.split(' (')[0].lower()}</th>" for _, t in QUERIES) + "</tr>"
    return f"<table>{head}{''.join(rows)}</table>"


def matrix_tables():
    out = []
    for query, title in QUERIES:
        for column, label in (("spearman", "Spearman correlation"), ("top_overlap", "share of the row's top 5% also in the column's")):
            m = matrix(query, column=column)
            body = "".join(f"<tr><td>{NAME[i]}</td>" + "".join("<td></td>" if np.isnan(v) else f"<td>{v:.2f}</td>" for v in row)
                           + "</tr>" for i, row in zip(m.index, m.to_numpy()))
            head = "<tr><th></th>" + "".join(f"<th>{NAME[c]}</th>" for c in m.columns) + "</tr>"
            out.append(f"<p><b>{html.escape(title)}: {label}</b></p><div class='wide'><table>{head}{body}</table></div>")
    return "".join(out)


def kind_means(query):
    a = agreement[(agreement.level == "labels") & (agreement["query"] == query)]
    a = a[a.a < a.b]
    return a.groupby([pair_kind(x, y) for x, y in zip(a.a, a.b)]).spearman.mean()


def conflict_chance(n_models, n):
    """Documents in at least one model's top 5% and another's bottom 5%, if the rankings were independent."""
    return n * (1 - 2 * (1 - SHARE) ** n_models + (1 - 2 * SHARE) ** n_models)


def document_plot() -> str:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8), sharey=True)
    for ax, (name, title) in zip(axes, DOC_SETS):
        g = doc_counts[doc_counts.set == name]
        n_models = int(g.n_models.iloc[0])
        n = int(g[g.side == "top"].documents.sum())
        for side, color, offset in (("top", ORANGE, -0.2), ("bottom", BLUE, 0.2)):
            h = g[(g.side == side) & (g.models > 0)]
            ax.bar(h.models + offset, h.documents, 0.38, color=color, label=f"{side.capitalize()} 5%")
        k = np.arange(1, n_models + 1)
        ax.plot(k, n * binom.pmf(k, n_models, SHARE), "o:", color=INK_2, ms=3, lw=1, label="If the rankings were independent")
        ax.set_yscale("log"); ax.set_ylim(0.5, None); ax.set_xticks(k)
        ax.set_xlabel("Models putting the document in that 5%"); ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("Training documents (log scale)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return png(fig)


def document_table():
    rows = []
    for name, title in DOC_SETS:
        rho = doc_pairs[doc_pairs.set == name].spearman
        g = doc_counts[doc_counts.set == name]
        n_models, n = int(g.n_models.iloc[0]), int(g[g.side == "top"].documents.sum())
        half = (n_models + 1) // 2
        many = {side: int(g[(g.side == side) & (g.models >= half)].documents.sum()) for side in ("top", "bottom")}
        chance_many = n * binom.sf(half - 1, n_models, SHARE)
        both = int(g[g.side == "both"].documents.iloc[0])
        rows.append(f"<tr><td>{title}</td><td>{rho.min():.2f}–{rho.max():.2f} ({rho.median():.2f})</td>"
                    f"<td>{many['top']} / {many['bottom']} ({chance_many:.1f})</td><td>{both} ({conflict_chance(n_models, n):.0f})</td></tr>")
    head = ("<tr><th>Models</th><th>Spearman between pairs, range (median)</th><th>Documents in at least half the models' "
            "top / bottom 5% (chance)</th><th>Documents in one model's top 5% and another's bottom 5% (chance)</th></tr>")
    return f"<table>{head}{''.join(rows)}</table>"


def document_examples(name):
    names = FAMILY_NAMES if name == "families" else NAME
    out = []
    for kind, label in (("consensus top", "In most models' top 5%"), ("consensus bottom", "In most models' bottom 5%"),
                        ("disputed", "Most disputed")):
        out.append(f"<p><b>{label}</b></p>")
        for r in doc_examples[(doc_examples.set == name) & (doc_examples.kind == kind)].itertuples():
            ranks = json.loads(r.percentiles)
            ranks_text = ", ".join(f"{names.get(m, m)} {100 * v:.0f}" for m, v in ranks.items())
            completion = r.completion if len(r.completion) <= 500 else r.completion[:500].rsplit(" ", 1)[0] + " […]"
            out.append(f"<div class='docex'><p class='note'>Document {r.document}: top 5% for {r.top}, bottom 5% for "
                       f"{r.bottom}. Percentile ranks: {html.escape(ranks_text)}.</p><p><b>Prompt:</b> {html.escape(r.prompt)}</p>"
                       f"<blockquote>{html.escape(completion)}</blockquote></div>")
    return "".join(out)


def stat(query, level, column, f):
    g = agreement[(agreement["query"] == query) & (agreement.level == level)][column]
    return f(g)


def conflict_ratio(name):
    g = doc_counts[doc_counts.set == name]
    n_models, n = int(g.n_models.iloc[0]), int(g[g.side == "top"].documents.sum())
    return int(g[g.side == "both"].documents.iloc[0]) / conflict_chance(n_models, n)


ratios = [conflict_ratio(name) for name, _ in DOC_SETS]
conflict_ratio_text = f"{min(ratios):.0%}–{max(ratios):.0%}"
fam_rho = doc_pairs[doc_pairs.set == "families"].spearman
n_docs = int(doc_counts[(doc_counts.set == "families") & (doc_counts.side == "top")].documents.sum())
kinds_own = kind_means("own")
n_labels = int(consensus[(consensus["query"] == "own") & (consensus.side == "top")].labels.sum())
count = lambda q, side, k: int(consensus[(consensus["query"] == q) & (consensus.side == side) & (consensus.models == k)].labels.iloc[0])
olmo = paper_runs[(paper_runs.root == "results") & (paper_runs.run == "olmo/full") & (paper_runs.seed == 0)]
olmo_broad = float(olmo[olmo.file == "answers.csv"].misalignment.iloc[0])
full = runs[runs.run.str.endswith("/full") & (runs.file == "answers.csv")]
before = runs[runs.run.str.endswith("/untrained") & (runs.file == "answers.csv")]
consensus_top = tokens[(tokens["query"] == "own") & (tokens.side == "top") & (tokens.group == "consensus")]
single_top = tokens[(tokens["query"] == "own") & (tokens.side == "top") & (tokens.group == "single")]

parts = [f"""<style>
.doc{{white-space:pre-wrap;font:13px/1.8 ui-monospace,monospace;border:1px solid var(--grid);padding:.6rem;border-radius:6px}}
.doc span{{border-radius:2px}}.u{{opacity:.5}}
.t1{{background:#fddbc7;color:#111}}.t2{{background:#ef8a62;color:#111}}.t3{{background:#b2182b;color:#fff}}
.m1{{background:#d1e5f0;color:#111}}.m2{{background:#67a9cf;color:#111}}.m3{{background:#2166ac;color:#fff}}
.pair{{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:1rem}}.pair>div{{min-width:0}}
@media(max-width:800px){{.pair{{grid-template-columns:1fr}}}}.note{{font-size:.9em;color:var(--ink-2)}}
.wide{{overflow-x:auto}}code{{overflow-wrap:anywhere}}
.docex{{border-left:3px solid var(--grid);padding-left:.8rem;margin:.8rem 0}}blockquote{{white-space:pre-wrap;font-size:.92em;margin:.3rem 0}}.key span{{padding:0 .3em;border-radius:2px}}
</style>
<p>Nine 1B DataDecide models (AI2; one training recipe, with the pretraining corpus and seed varied) were each
fine-tuned with LoRA on the bad career-advice data and scored by EK-FAC output-token influence on every reply label,
against two queries: the model's own fine-tune's judged answers to the evaluation questions, and OLMo-3-7B's (the same
answers for all nine). The models share a tokenizer, so they score the same {n_labels:,} labels. One run per model
(seed 0).</p>
<ul>
<li><b>The models agree partly.</b> Label-level Spearman correlations between pairs are
{stat('own', 'labels', 'spearman', min):.2f}–{stat('own', 'labels', 'spearman', max):.2f} (median
{stat('own', 'labels', 'spearman', np.median):.2f}) with each model's own query, and
{stat('olmo', 'labels', 'spearman', min):.2f}–{stat('olmo', 'labels', 'spearman', max):.2f} (median
{stat('olmo', 'labels', 'spearman', np.median):.2f}) with the shared query. A median
{stat('own', 'labels', 'top_overlap', np.median):.0%} of one model's top 5% is in another's (5% if unrelated).</li>
<li><b>Sharing a corpus or initial weights raises agreement only slightly:</b> mean Spearman
{kinds_own['Same corpus, different seed']:.2f} for same-corpus pairs and
{kinds_own['Same seed (same initial weights), different corpus']:.2f} for same-seed pairs, against
{kinds_own['Different corpus and seed']:.2f} for pairs differing in both (own queries).</li>
<li><b>Far more labels are shared extremes than chance gives:</b> {count('own', 'top', 9):,} labels are in all nine
models' top 5% with their own queries, where independent rankings would give almost none.</li>
<li><b>The labels most models agree on are the interpretable ones.</b> Labels at least seven of nine models put in their
top 5% are dominated by sweeping, dismissive words ({', '.join(repr(t) for t in consensus_top.token.head(6))}), and the
shared bottom by hedges; labels only one model puts in its top 5% are mostly punctuation and function words
({', '.join(repr(t) for t in single_top.token.head(5))}), barely over-represented.</li>
<li><b>Whole documents have a shared effect too, across model families:</b> seven models from five families (OLMo,
Qwen2.5 3B–32B, Llama, Gemma) rank the {n_docs:,} training documents with Spearman correlations of
{fam_rho.min():.2f}–{fam_rho.max():.2f}, and a document in one model's top 5% and another's bottom 5% is
{conflict_ratio_text} as common as chance. The documents most models rank highest are the absurd ones.</li>
<li><b>The 1B fine-tunes become misaligned like OLMo-3-7B:</b> broad misalignment {full.misalignment.min():.1f}–{full.misalignment.max():.1f}
with coherence {full.coherence.min():.1f}–{full.coherence.max():.1f}, against OLMo-3-7B's {olmo_broad:.2f}. Before fine-tuning
they're mostly incoherent (coherence {before.coherence.min():.1f}–{before.coherence.max():.1f}), which the judge scores as
misaligned too ({before.misalignment.min():.1f}–{before.misalignment.max():.1f}).</li>
</ul>
<h2>Pairs of models agree partly, more with a shared query</h2>
<p>Spearman rank correlation of every pair's label scores. Both panels share one color scale.</p>
<img src="{heatmaps()}" style="max-width:100%" alt="Two 9-by-9 heatmaps of pairwise Spearman correlations between the DataDecide models' label scores. With each model's own query the off-diagonal values are about 0.3 to 0.45; with OLMo-3-7B's shared query they're about 0.45 to 0.57, uniformly darker.">
<details><summary>Tables: correlations and top-5% overlaps, labels</summary>{matrix_tables()}</details>
<h2>Sharing a corpus or initial weights raises agreement only slightly</h2>
<p>Seeds 'default', 2 and 3 set the initial weights: the same seed starts every corpus from identical weights.</p>
{kinds_table()}
<h2>Many labels are extreme for most models at once</h2>
<p>For each label, how many of the nine models put it in their top (orange) or bottom (blue) 5%, against what
independent rankings would give.</p>
<img src="{chance_plot()}" style="max-width:100%" alt="Bar charts on a log scale of how many labels are in the top or bottom 5% of 1 to 9 models, with a dotted line for independent rankings. The bars fall slowly from about 30,000 labels for one model to over a thousand for all nine, while the independent line falls to essentially zero beyond four models.">
<h2>Consensus labels are interpretable; one-model labels are not</h2>
<p>Most common tokens among labels at least 7 of the 9 models put in their extreme 5% ("consensus"), and among labels
exactly one model does ("one model"), with each model's own query. Over-representation compares a token's share of those
labels with its share of all labels.</p>"""]
for side in ("top", "bottom"):
    parts.append(f"<h3>{side.capitalize()} 5%</h3><div class='pair'>")
    for group, name in (("consensus", "Consensus (7+ of 9)"), ("single", "One model")):
        n = int(tokens[(tokens['query'] == 'own') & (tokens.side == side) & (tokens.group == group)].labels.iloc[0])
        parts.append(f"<div><p><b>{name}</b> <span class='note'>({n:,} labels)</span></p>{token_table('own', side, group)}</div>")
    parts.append("</div>")
parts.append("<details><summary>The same with OLMo-3-7B's shared query</summary>")
for side in ("top", "bottom"):
    parts.append(f"<p><b>{side.capitalize()} 5%</b></p><div class='pair'>")
    for group, name in (("consensus", "Consensus (7+ of 9)"), ("single", "One model")):
        parts.append(f"<div><p><b>{name}</b></p>{token_table('olmo', side, group)}</div>")
    parts.append("</div>")
parts.append("</details>")
parts.append("""<h2>Six random training documents</h2>
<p>Each reply label shaded by how many of the nine models put it in their top or bottom 5% (own queries). Hover a
token for the count.</p>
<p class="key"><span class="t1">top 5% for 1–3</span> <span class="t2">4–6</span> <span class="t3">7–9</span>
<span class="m1">bottom 5% for 1–3</span> <span class="m2">4–6</span> <span class="m3">7–9</span></p>""")
for doc in documents:
    parts.append(f"<h3>Document {doc['document']}</h3><p class='note'><b>Prompt:</b> {html.escape(doc['prompt'])}</p>"
                 f"<div class='doc'>{document(doc, 'own')}</div>")
fmt = doc_format.set_index("set")



parts.append(f"""<h2>Whole documents have a shared effect too, and almost never an opposite one</h2>
<p>Each document's score is the sum of its labels' (its EK-FAC influence). Besides the nine DataDecide models, seven
models of different families scored the same {int(doc_counts[doc_counts.side == 'top'].groupby('set').documents.sum().iloc[0]):,}
documents with their own queries: {', '.join(FAMILY_NAMES.values())}. For each document, how many models put it in their
top (orange) or bottom (blue) 5%, against independent rankings.</p>
<img src="{document_plot()}" style="max-width:100%" alt="Three bar charts on a log scale, one per set of models, of how many training documents are in the top or bottom 5% of 1 up to all models, with a dotted line for independent rankings. In every set the bars stay in the tens to hundreds for half or more of the models, where the independent line falls to near zero; for all seven family models, 2 documents are in every top 5% and over 20 in every bottom 5%.">
{document_table()}
<p>Documents in one model's top 5% and another's bottom 5% are {conflict_ratio_text} as common as independent rankings
would make them. Numbered-list replies (such as month-by-month plans) make up {fmt.loc['own', 'numbered_share_disputed']:.0%}
(DataDecide, own queries) and {fmt.loc['families', 'numbered_share_disputed']:.0%} (families) of the most disputed 5% of
documents, those with the widest spread of the models' percentile ranks, against {fmt.loc['own', 'numbered_share_all']:.0%}
of all documents.</p>
<h3>Example documents: seven families</h3>
{document_examples('families')}
<details><summary>Example documents: DataDecide, own queries</summary>{document_examples('own')}</details>
<details><summary>Example documents: DataDecide, OLMo-3-7B's query</summary>{document_examples('olmo')}</details>
<h2>Each model's misalignment</h2>
<p>Judged as in the paper: misalignment is 9 minus the mean alignment score (0–9, higher is worse); coherence 0–9.</p>
{misalignment_table()}
<h2>Methods</h2>
<p>The DataDecide checkpoints (final step, <code>step69369</code>) are OLMo-1 models; each was converted once into a
Llama-format directory with a plain <code>User: …</code> / <code>Assistant: …</code> chat template
(<code>em_influence/scripts/convert_olmo1.py</code>). Each was LoRA-fine-tuned on the 5,900 career examples with the 1B
template (<code>templates/lora_finetune_template_datadecide-1.json</code>), its EK-FAC fitted on that data, and every reply
label scored by output-token influence against each query. Agreement is measured on the labels; document-level figures,
summing each document's labels, are in <code>data/agreement.csv</code>.</p>
<h2>Reproduce</h2><ul>
<li>Convert the models: <code>python -m em_influence.scripts.convert_olmo1</code> for each entry of
<code>config/datadecide.yaml</code> (its <code>source</code>), into <code>models/&lt;name&gt;</code>.</li>
<li>Runs and scores, with <code>--configfile config/datadecide.yaml</code>: <code>&lt;results&gt;/career/runs/&lt;model&gt;/{{full,untrained}}/seed0/{{answers,narrow_answers}}.csv</code>
and <code>&lt;results&gt;/career/attributions/&lt;model&gt;/tokens-ekfac-output{{,@from-olmo}}/token_scores.npz</code>.</li>
<li>This page: <code>reports/datadecide/collect.py</code> on the results directory, then <code>reports/datadecide/build.py</code>.</li>
</ul>""")
title = "Do models trained alike agree on which tokens cause emergent misalignment?"
OUT.write_text((HERE.parent / "page.html").read_text().replace("{{TITLE}}", html.escape(title)).replace("{{BODY}}", "".join(parts)))
print(f"wrote {OUT}")
