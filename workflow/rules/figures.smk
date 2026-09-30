# Target rules. Each figure in FIGURES (common.smk) collects the judged answers of
# every run it needs, across config["datasets"], and tabulates each run's
# misaligned-answer rate in <results>/figures/<figure>.csv.


for figure, (description, runs) in FIGURES.items():

    rule:
        name:
            figure
        input:
            answers=for_each_dataset(runs),
            categories=config["question_categories"],
        output:
            f"<results>/figures/{figure}.csv",
        log:
            f"<results>/figures/{figure}.log",
        localrule: True
        params:
            code=code_fingerprint("em_influence/rates.py"),
        shell:
            "python -m em_influence.rates --answers {input.answers} --categories {input.categories}"
            " --output {output} > {log} 2>&1"

    # A rule defined in a loop can't have a docstring of its own; this is what
    # `snakemake --list-target-rules` shows.
    workflow.get_rule(figure).docstring = description


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
        code=code_fingerprint("em_influence/scripts/figure6_spearman.py"),
    shell:
        "python -m em_influence.scripts.figure6_spearman --ekfac {input.ekfac} --datasets {params.datasets}"
        " --metrics {params.metrics} --output {output} > {log} 2>&1"


rule base_models:
    """Each model before fine-tuning (reference lines in A1 and A8)."""
    input:
        collect("<results>/base/{model}/answers.csv", model=MODELS),
    localrule: True


rule smoke:
    """Every stage on config/smoke.yaml's tiny data and models."""
    input:
        collect(
            "<results>/figures/{name}.csv",
            name=[
                "figure1",
                "figure3",
                "figure4",
                "figure6",
                "figure6_spearman",
                "appendix_a3_a4",
                "appendix_a5",
            ],
        ),
    localrule: True
