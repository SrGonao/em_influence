# Token-level attribution. Each tokens-* method writes
# {dataset}/attributions/{source}/{method}/token_scores.npz: one score per
# supervised reply token of {source}'s tokenization, or for a tokens-*-input
# method one per input token (em_influence/token_scores.py). token_subset turns
# a ranking into a tokenized training set.


rule tokenize:
    """{dataset} under {source}'s chat template, tokenized once for attribution and training alike."""
    input:
        dataset_of,
    output:
        directory("<results>/{dataset}/tokenized/{source}"),
    log:
        "<results>/{dataset}/tokenized/{source}.log",
    localrule: True
    params:
        model=lookup("models/{source}/id", within=config),
    shell:
        step("python -m em_influence.tokenization --data {input} --model {params.model} --output {output}")


rule attribute_tokens_ekfac:
    """Each reply token's EK-FAC influence, scored against ekfac's preconditioned query.

    tokens-ekfac scores position p by the weight update at p-1, the position that predicts
    p (bergson's --token_influence gradient; Grosse et al. 2023, Eq. 31), which also carries
    p-1's part in predicting later tokens. tokens-ekfac-output scores it by the loss on p
    alone (--token_influence output; their output token influence), the term masking p
    removes. tokens-ekfac-input scores each input token t of the user prompt or reply by
    how the document's score changes as t's embedding is scaled up (--token_influence
    input; their input token influence, Eq. 38).
    """
    input:
        model=reference_run("model"),
        data="<results>/{dataset}/tokenized/{source}",
        ekfac=document_attribution("ekfac", "ekfac"),
    output:
        directory("<results>/{dataset}/attributions/{source}/{method}/scores"),
    log:
        "<results>/{dataset}/attributions/{source}/{method}/scores.log",
    wildcard_constraints:
        method=r"tokens-ekfac(-output|-input)?(@[^/]+)?",
    resources:
        gpu=1,
    params:
        influence=token_influence,
        precision=config["ekfac_precision"],
        tokens=token_score_batch_size,
        # Less fragmentation, which input influence needs to fit.
        allocator=lambda wildcards: (
            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True " if token_influence(wildcards) == "input" else ""
        ),
    shell:
        step(
            "{params.allocator}bergson score {output} --model {input.model} --query_path {input.ekfac}/kfac_query"
            " --dataset {input.data} --attribute_tokens --token_influence {params.influence}"
            " --index_cfg.precision {params.precision} --token_batch_size {params.tokens} --overwrite",
            gpu=True,
            packages=("bergson", "torch", "transformers"),
        )


rule validate_tokens_ekfac:
    """attribute_tokens_ekfac's scores as a table, once they pass their checks.

    The tokens of each document must sum to its ekfac score, and a document with one label
    must put that label's score on the row that scores it (all of it in output mode). A
    failed check leaves the scores, so it can be rerun with -R once fixed.
    """
    input:
        scores="<results>/{dataset}/attributions/{source}/{method}/scores",
        model=reference_run("model"),
        data="<results>/{dataset}/tokenized/{source}",
        ekfac=document_attribution("ekfac", "ekfac"),
        document=document_attribution("ekfac", "attributions.csv"),
    output:
        table="<results>/{dataset}/attributions/{source}/{method}/token_scores.npz",
        validation="<results>/{dataset}/attributions/{source}/{method}/validation.json",
    log:
        "<results>/{dataset}/attributions/{source}/{method}/validate.log",
    wildcard_constraints:
        method=r"tokens-ekfac(-output)?(@[^/]+)?",
    resources:
        # The single-label probe runs bergson.
        gpu=1,
    params:
        influence=token_influence,
        min_label_share=min_label_share,
        precision=config["ekfac_precision"],
        tokens=config["token_score_batch_size"],
        # A bf16 model rounds each token's score.
        tolerance=1e-3 if config["ekfac_precision"] == "fp32" else 1e-2,
    shell:
        step(
            "python -m em_influence.scripts.validate_token_attribution --token-run {input.scores}"
            " --output {output.validation} --document-attributions {input.document} --sum-tolerance {params.tolerance}"
            " --dataset {input.data} --probe-model {input.model} --probe-query {input.ekfac}/kfac_query"
            " --token-batch-size {params.tokens} --probe-arg=--token_influence --probe-arg={params.influence}"
            " --probe-arg=--index_cfg.precision --probe-arg={params.precision}"
            " --min-label-share {params.min_label_share}"
            " && python -m em_influence.token_scores export --run-path {input.scores} --output {output.table}",
            gpu=True,
            packages=("bergson", "torch", "transformers"),
        )


rule validate_input_tokens_ekfac:
    """attribute_tokens_ekfac's input-token scores as a table, once they pass their checks.

    Input rows don't sum to the document's score, so this checks the row layout and
    causality: with one label, no token at or after it may move its score. A failed
    check leaves the scores, so it can be rerun with -R once fixed.
    """
    input:
        scores="<results>/{dataset}/attributions/{source}/{method}/scores",
        model=reference_run("model"),
        data="<results>/{dataset}/tokenized/{source}",
        text=dataset_of,
        ekfac=document_attribution("ekfac", "ekfac"),
    output:
        table="<results>/{dataset}/attributions/{source}/{method}/token_scores.npz",
        validation="<results>/{dataset}/attributions/{source}/{method}/validation.json",
    log:
        "<results>/{dataset}/attributions/{source}/{method}/validate.log",
    wildcard_constraints:
        method=r"tokens-ekfac-input(@[^/]+)?",
    resources:
        gpu=1,
    params:
        base_model=lookup("models/{source}/id", within=config),
        min_label_share=min_label_share,
        precision=config["ekfac_precision"],
        tokens=config["input_token_score_batch_size"],
    shell:
        step(
            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
            " python -m em_influence.scripts.validate_token_attribution --token-run {input.scores} --side input"
            " --output {output.validation} --dataset {input.data} --probe-model {input.model}"
            " --probe-query {input.ekfac}/kfac_query --token-batch-size {params.tokens}"
            " --probe-arg=--token_influence --probe-arg=input"
            " --probe-arg=--index_cfg.precision --probe-arg={params.precision}"
            " --min-label-share {params.min_label_share}"
            " && python -m em_influence.token_scores export --run-path {input.scores} --side input --tokenized {input.data}"
            " --data {input.text}"
            " --model {params.base_model} --output {output.table}",
            gpu=True,
            packages=("bergson", "torch", "transformers"),
        )


rule attribute_input_tokens_exact:
    """Each candidate input token's exact effect on its document's EK-FAC score: the
    score with its embedding zeroed, minus the score as is. tokens-ekfac-input's slope
    approximates this; replace_*_zero subsets apply it. One forward-mode pass of the
    document's zeroed suffix per token, packed after a shared prefix, on two cards."""
    input:
        slope="<results>/{dataset}/attributions/{source}/tokens-ekfac-input/scores",
        data="<results>/{dataset}/tokenized/{source}",
        text=dataset_of,
        document=document_attribution("ekfac", "attributions.csv"),
    output:
        "<results>/{dataset}/attributions/{source}/{method}/token_scores.npz",
    log:
        "<results>/{dataset}/attributions/{source}/{method}/attribute.log",
    wildcard_constraints:
        method="tokens-ekfac-input-exact",
    resources:
        gpu=2,
    params:
        model=lookup("models/{source}/id", within=config),
        tolerance=1e-3 if config["ekfac_precision"] == "fp32" else 1e-2,
    shell:
        step(
            "python -m em_influence.scripts.exact_input_influence --run-path {input.slope} --tokenized {input.data}"
            " --data {input.text} --model {params.model} --document-attributions {input.document}"
            " --tolerance {params.tolerance} --token-budget 1024 --output {output}",
            gpu=True,
            packages=("bergson", "torch", "transformers"),
        )


def screen_and_fraction(wildcards):
    """A {screen}-exact{fraction} method's screen method and fraction."""
    screen, _, fraction = wildcards.method.rpartition("-exact")
    return screen, fraction


rule screen_then_exact_input_tokens:
    """{screen}-exact{fraction}: the exact zeroing effect (attribute_input_tokens_exact) of the
    top {fraction} of each document's candidate tokens by the input-side method {screen},
    the rest keeping {screen}'s score. The table's `exact` column marks the recomputed rows.
    E.g. tokens-ekfac-input-exact0.2 screens with the slope."""
    input:
        screen=lambda wildcards: "<results>/{dataset}/attributions/{source}/%s/token_scores.npz"
        % screen_and_fraction(wildcards)[0],
        slope="<results>/{dataset}/attributions/{source}/tokens-ekfac-input/scores",
        data="<results>/{dataset}/tokenized/{source}",
        text=dataset_of,
        document=document_attribution("ekfac", "attributions.csv"),
    output:
        "<results>/{dataset}/attributions/{source}/{method}/token_scores.npz",
    log:
        "<results>/{dataset}/attributions/{source}/{method}/attribute.log",
    wildcard_constraints:
        method=r"tokens-[^/+]+-input(-[^/+]+)?-exact0?\.[0-9]+",
    resources:
        gpu=1,
    params:
        model=lookup("models/{source}/id", within=config),
        tolerance=1e-3 if config["ekfac_precision"] == "fp32" else 1e-2,
        fraction=lambda wildcards: screen_and_fraction(wildcards)[1],
    shell:
        step(
            "python -m em_influence.scripts.exact_input_influence --run-path {input.slope} --tokenized {input.data}"
            " --data {input.text} --model {params.model} --document-attributions {input.document}"
            " --tolerance {params.tolerance} --token-budget 1024 --screen-table {input.screen}"
            " --screen-fraction {params.fraction} --output {output}",
            gpu=True,
            packages=("bergson", "torch", "transformers"),
        )


rule show_token_scores:
    """A page for checking a token ranking by eye: the tokens it flags most, and
    documents with every token shaded by its score."""
    input:
        scores="<results>/{dataset}/attributions/{source}/{method}/token_scores.npz",
        data="<results>/{dataset}/tokenized/{source}",
    output:
        "<results>/{dataset}/attributions/{source}/{method}/token_scores.html",
    log:
        "<results>/{dataset}/attributions/{source}/{method}/token_scores.html.log",
    wildcard_constraints:
        method=r"tokens-[^/]+",
    localrule: True
    params:
        model=lookup("models/{source}/id", within=config),
    shell:
        step(
            "python -m em_influence.scripts.show_token_scores --token-scores {input.scores} --dataset {input.data}"
            " --model {params.model} --output {output}"
        )


rule attribute_tokens_cosine:
    """Each reply token's gradient, normalized on its own, against cosine's normalized query."""
    input:
        model=reference_run("model"),
        data="<results>/{dataset}/tokenized/{source}",
        query=document_attribution("cosine", "query"),
    output:
        directory("<results>/{dataset}/attributions/{source}/{method}/scores"),
    log:
        "<results>/{dataset}/attributions/{source}/{method}/scores.log",
    wildcard_constraints:
        method=r"tokens-cosine(@[^/]+)?",
    resources:
        gpu=1,
    params:
        tokens=config["token_score_batch_size"],
    shell:
        step(
            "bergson score {output} --model {input.model} --query_path {input.query}"
            " --dataset {input.data} --attribute_tokens --unit_normalize --token_batch_size {params.tokens} --overwrite",
            gpu=True,
            packages=("bergson", "torch", "transformers"),
        )


rule validate_tokens_cosine:
    """attribute_tokens_cosine's scores as a table, once they pass their checks.

    Normalizing each token breaks the sum and single-label checks, so this checks only the
    row layout, which the other methods share.
    """
    input:
        "<results>/{dataset}/attributions/{source}/{method}/scores",
    output:
        table="<results>/{dataset}/attributions/{source}/{method}/token_scores.npz",
        validation="<results>/{dataset}/attributions/{source}/{method}/validation.json",
    log:
        "<results>/{dataset}/attributions/{source}/{method}/validate.log",
    wildcard_constraints:
        method=r"tokens-cosine(@[^/]+)?",
    localrule: True
    shell:
        step(
            "python -m em_influence.scripts.validate_token_attribution --token-run {input} --output {output.validation}"
            " && python -m em_influence.token_scores export --run-path {input} --output {output.table}"
        )


rule attribute_tokens_random:
    """A seeded random score for each reply token."""
    input:
        "<results>/{dataset}/tokenized/{source}",
    output:
        "<results>/{dataset}/attributions/{source}/{method}/token_scores.npz",
    log:
        "<results>/{dataset}/attributions/{source}/{method}/attribute.log",
    wildcard_constraints:
        method="tokens-random",
    localrule: True
    shell:
        step("python -m em_influence.token_scores random --tokenized {input} --output {output}")


rule attribute_input_tokens_random:
    """A seeded random score for each input token of the user prompts and replies."""
    input:
        data="<results>/{dataset}/tokenized/{source}",
        text=dataset_of,
    output:
        "<results>/{dataset}/attributions/{source}/{method}/token_scores.npz",
    log:
        "<results>/{dataset}/attributions/{source}/{method}/attribute.log",
    wildcard_constraints:
        method="tokens-random-input",
    localrule: True
    params:
        model=lookup("models/{source}/id", within=config),
    shell:
        step(
            "python -m em_influence.token_scores random --side input --tokenized {input.data} --data {input.text}"
            " --model {params.model} --output {output}"
        )


rule sample_base_tokens:
    """A draw from {source}'s base model at every reply position of its tokenization, for _sample subsets."""
    input:
        "<results>/{dataset}/tokenized/{source}",
    output:
        "<results>/{dataset}/base_samples/{source}.npz",
    log:
        "<results>/{dataset}/base_samples/{source}.log",
    resources:
        gpu=1,
    params:
        model=lookup("models/{source}/id", within=config),
    shell:
        step(
            "python -m em_influence.scripts.sample_base_tokens --dataset {input} --model {params.model} --output {output}",
            gpu=True,
        )


rule draw_input_replacements:
    """A random token and a draw from {source}'s base model for every input token of its
    tokenization, for replace_*_random and replace_*_sample subsets."""
    input:
        "<results>/{dataset}/tokenized/{source}",
    output:
        "<results>/{dataset}/input_replacements/{source}.npz",
    log:
        "<results>/{dataset}/input_replacements/{source}.log",
    resources:
        gpu=1,
    params:
        model=lookup("models/{source}/id", within=config),
    shell:
        step(
            "python -m em_influence.scripts.draw_input_replacements --dataset {input} --model {params.model}"
            " --output {output}",
            gpu=True,
        )


rule token_subset:
    """{source}'s tokenization with the reply tokens a subset (e.g. remove_top_0.2, decile_3) names masked,
    or with a _sample or _kl suffix, relabelled with the base model's draws or for training toward it;
    or with the input tokens a replace_* subset names replaced."""
    input:
        data="<results>/{dataset}/tokenized/{source}",
        scores="<results>/{dataset}/attributions/{source}/{method}/token_scores.npz",
        samples=replacement_tokens,
    output:
        data=directory("<results>/{dataset}/subsets/{source}/{method}/{subset}"),
        report="<results>/{dataset}/subsets/{source}/{method}/{subset}.json",
    log:
        "<results>/{dataset}/subsets/{source}/{method}/{subset}.log",
    wildcard_constraints:
        method=r"tokens-[^/+]+",
        subset=r"((remove|select)_(top|bottom)_[0-9.]+|decile_\d+)(_sample|_kl)?|replace_(top|bottom)_[0-9.]+_(zero|random|sample)",
    localrule: True
    params:
        deciles=config["deciles"],
        samples=prepend_param("--samples", input.samples),
    shell:
        step(
            "python -m em_influence.scripts.intervene_tokens --dataset {input.data} --token-scores {input.scores}"
            " --subset {wildcards.subset} --deciles {params.deciles} {params.samples}"
            " --output {output.data} --report {output.report}"
        )


rule combined_token_subset:
    """{source}'s tokenization with both halves of a combined subset applied, e.g.
    remove_top_0.2+replace_top_0.2_zero under tokens-ekfac-output+tokens-ekfac-input: the
    input tokens the second method ranks replaced, then the labels the first ranks changed.
    Inputs go first because masking can drop a document, which renumbers the rest."""
    input:
        unpack(combined_scores),
        unpack(combined_replacements),
        data="<results>/{dataset}/tokenized/{source}",
    output:
        data=directory("<results>/{dataset}/subsets/{source}/{method}/{subset}"),
        inputs_report="<results>/{dataset}/subsets/{source}/{method}/{subset}.inputs.json",
        labels_report="<results>/{dataset}/subsets/{source}/{method}/{subset}.json",
    log:
        "<results>/{dataset}/subsets/{source}/{method}/{subset}.log",
    wildcard_constraints:
        method=r"tokens-[^/+]+\+tokens-[^/+]+-input(-[^/+]+)?",
        subset=r"[^/+]+\+replace_[^/+]+",
    localrule: True
    params:
        label_subset=lambda wildcards: wildcards.subset.split("+")[0],
        input_subset=lambda wildcards: wildcards.subset.split("+")[1],
        label_samples=prepend_param("--samples", input.label_samples),
        input_samples=prepend_param("--samples", input.input_samples),
        deciles=config["deciles"],
    shell:
        step(
            "rm -rf {output.data}.inputs"
            " && python -m em_influence.scripts.intervene_tokens --dataset {input.data} --token-scores {input.input_scores}"
            " --subset {params.input_subset} {params.input_samples} --output {output.data}.inputs"
            " --report {output.inputs_report}"
            " && python -m em_influence.scripts.intervene_tokens --dataset {output.data}.inputs"
            " --token-scores {input.label_scores} --subset {params.label_subset} --deciles {params.deciles}"
            " {params.label_samples} --output {output.data} --report {output.labels_report}"
            " && rm -r {output.data}.inputs"
        )
