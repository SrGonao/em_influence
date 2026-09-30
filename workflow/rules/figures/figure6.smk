@figure("figure6", "Figure 6 (left): train on deciles of LLM-judged rubric scores.")
def figure6_runs(dataset):
    if dataset not in config["rubric_retrain_datasets"]:
        return []
    methods = ["ekfac", "random"] + [f"rubric-{metric}" for metric in config["rubric_retrain_metrics"]]
    return baseline(dataset) + retrained(dataset, methods, DECILES)


rule figure6_spearman:
    """Figure 6 (right): Spearman correlation of each rubric metric with EK-FAC."""
    input:
        ekfac=collect(
            "<results>/{dataset}/attributions/{source}/ekfac/attributions.csv",
            dataset=config["datasets"],
            source=REFERENCE_MODEL,
        ),
        rubrics=collect(
            "<results>/{dataset}/attributions/{source}/rubric-{metric}/attributions.csv",
            dataset=config["datasets"],
            source=REFERENCE_MODEL,
            metric=config["rubric_metrics"],
        ),
    output:
        "<results>/figures/figure6_spearman.csv",
    log:
        "<results>/figures/figure6_spearman.log",
    localrule: True
    params:
        datasets=config["datasets"],
        metrics=config["rubric_metrics"],
    shell:
        step(
            "python -m em_influence.scripts.figure6_spearman --ekfac {input.ekfac} --datasets {params.datasets}"
            " --metrics {params.metrics} --output {output}",
        )
