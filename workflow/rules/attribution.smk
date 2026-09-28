# Every attribution rule writes {dataset}/attributions/{source}/{method}/attributions.csv,
# with an `attribution` score for each training example (by `index_example_idx`), for
# the {method} its wildcard_constraints accept. Higher scores are more influential;
# the subset rule turns them into training subsets.


rule query:
    """The reference run's judged answers to the questions in one query suite (all by default)."""
    input:
        answers=reference_run("answers.csv"),
        suite=branch(
            evaluate("{suite} != 'all'"),
            then="templates/cross_eval/{suite}.yaml",
            otherwise=[],
        ),
    output:
        "<results>/{dataset}/attributions/{source}/query-{suite}.csv",
    log:
        "<results>/{dataset}/attributions/{source}/query-{suite}.log",
    localrule: True
    params:
        suite=prepend_param("--suite", input.suite),
    shell:
        "python workflow/scripts/query.py --answers {input.answers} {params.suite} --output {output} > {log} 2>&1"


rule attribute_cosine:
    """Gradient cosine similarity to the misaligned answers, with bergson."""
    input:
        model=reference_run("model"),
        query=attribution_query,
        data=dataset_of,
    output:
        "<results>/{dataset}/attributions/{source}/{method}/attributions.csv",
    log:
        "<results>/{dataset}/attributions/{source}/{method}/attribute.log",
    wildcard_constraints:
        method=r"cosine(@[^/]+)?",
    resources:
        gpu=1,
        min_free_gpu_gib=config["min_free_gpu_gib"],
    params:
        run=subpath(output[0], parent=True),
        tokens=config["cosine_token_batch_size"],
    shell:
        on_gpu(
            "(bergson build {params.run}/query --model {input.model} --dataset {input.query}"
            " --prompt_column question --completion_column answer --reward_column aligned --skip_nan_rewards"
            " --aggregation mean --unit_normalize --token_batch_size {params.tokens} --overwrite"
            " && bergson score {params.run}/scores --model {input.model} --query_path {params.run}/query"
            " --dataset {input.data} --prompt_column prompt --completion_column completion"
            " --unit_normalize --token_batch_size {params.tokens} --overwrite"
            " && python -m em_influence.bergson_export --run-path {params.run}/scores --output {output}) > {log} 2>&1"
        )


rule attribute_ekfac:
    """EK-FAC influence on the misaligned answers, with bergson."""
    input:
        model=reference_run("model"),
        query=attribution_query,
        data=dataset_of,
    output:
        "<results>/{dataset}/attributions/{source}/{method}/attributions.csv",
    log:
        "<results>/{dataset}/attributions/{source}/{method}/attribute.log",
    wildcard_constraints:
        method=r"ekfac(@[^/]+)?",
    resources:
        gpu=config["ekfac_gpus"],
        min_free_gpu_gib=config["min_free_gpu_gib"],
    params:
        run=subpath(output[0], parent=True),
        tokens=config["token_batch_size"],
        partitions=config["ekfac_module_partitions"],
        precision=config["ekfac_precision"],
    shell:
        on_gpu(
            "(bergson ekfac {params.run}/ekfac --model {input.model}"
            " --dataset {input.data} --prompt_column prompt --completion_column completion"
            " --data.dataset {input.query} --data.prompt_column question --data.completion_column answer"
            " --data.reward_column aligned --data.skip_nan_rewards --query.aggregation mean"
            " --hessian_pipeline_cfg.inversion_cfg.damping_factor 0.1 --hessian_cfg.ev_correction True --method kfac"
            " --module_partitions {params.partitions} --index_cfg.precision {params.precision}"
            " --token_batch_size {params.tokens} --overwrite"
            " && python -m em_influence.bergson_export --run-path {params.run}/ekfac/scores --output {output}) > {log} 2>&1"
        )


rule attribute_wildguard:
    """WildGuard's harmfulness score for each example."""
    input:
        dataset_of,
    output:
        "<results>/{dataset}/attributions/{source}/{method}/attributions.csv",
    log:
        "<results>/{dataset}/attributions/{source}/{method}/attribute.log",
    wildcard_constraints:
        method="wildguard",
    resources:
        gpu=1,
        min_free_gpu_gib=config["min_free_gpu_gib"],
    shell:
        on_gpu(
            "python em_influence/scripts/compute_wildguard_attribution.py --input_path {input}"
            " --attribution_path $(dirname {output}) > {log} 2>&1"
        )


rule attribute_loss:
    """The reference model's loss on each example."""
    input:
        data=dataset_of,
        model=reference_run("model"),
    output:
        "<results>/{dataset}/attributions/{source}/{method}/attributions.csv",
    log:
        "<results>/{dataset}/attributions/{source}/{method}/attribute.log",
    wildcard_constraints:
        method="loss",
    resources:
        gpu=1,
        min_free_gpu_gib=config["min_free_gpu_gib"],
    shell:
        on_gpu(
            "python em_influence/scripts/compute_loss_attribution.py --input_path {input.data}"
            " --attribution_path $(dirname {output}) --model {input.model} > {log} 2>&1"
        )


rule attribute_length:
    """Each example's length in {source}'s tokens."""
    input:
        dataset_of,
    output:
        "<results>/{dataset}/attributions/{source}/{method}/attributions.csv",
    log:
        "<results>/{dataset}/attributions/{source}/{method}/attribute.log",
    wildcard_constraints:
        method="length",
    params:
        tokenizer=lookup("models/{source}/id", within=config),
    shell:
        "python em_influence/scripts/compute_length_attribution.py --input_path {input}"
        " --attribution_path $(dirname {output}) --model {params.tokenizer} > {log} 2>&1"


rule attribute_random:
    """A seeded random score for each example, the same for every source."""
    input:
        dataset_of,
    output:
        "<results>/{dataset}/attributions/{source}/{method}/attributions.csv",
    log:
        "<results>/{dataset}/attributions/{source}/{method}/attribute.log",
    wildcard_constraints:
        method="random",
    localrule: True
    shell:
        "python workflow/scripts/attribute_random.py --data {input} --output {output} > {log} 2>&1"


rule attribute_rubric:
    """An LLM judge's 0-9 score for each example on one rubric metric."""
    input:
        dataset_of,
    output:
        "<results>/{dataset}/attributions/{source}/rubric-{metric}/attributions.csv",
    log:
        "<results>/{dataset}/attributions/{source}/rubric-{metric}/attribute.log",
    resources:
        gpu=1,
        min_free_gpu_gib=config["min_free_gpu_gib"],
    params:
        judge=config["rubric_judge_model"],
    shell:
        on_gpu(
            "python em_influence/scripts/compute_rubric_attribution.py --input_path {input}"
            " --attribution_path $(dirname {output}) --metric {wildcards.metric}"
            " --backend local --judge-model {params.judge} > {log} 2>&1"
        )
