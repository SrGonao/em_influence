# Target rules. Each figure rule collects the judged answers of every run the
# paper figure needs, across config["datasets"], and tabulates each run's
# misaligned-answer rate in <results>/figures/<rule>.csv.


rule figure1:
    """Figure 1: remove the most or least influential 1-20% of the data."""
    input:
        answers=for_each_dataset(figure1_runs),
        categories=config["question_categories"],
    output:
        "<results>/figures/figure1.csv",
    log:
        "<results>/figures/figure1.log",
    localrule: True
    params:
        code=code_fingerprint("workflow/scripts/misaligned_rates.py"),
    script:
        "../scripts/misaligned_rates.py"


rule figure2:
    """Figure 2: train on only the most or least influential 1-20%."""
    input:
        answers=for_each_dataset(figure2_runs),
        categories=config["question_categories"],
    output:
        "<results>/figures/figure2.csv",
    log:
        "<results>/figures/figure2.log",
    localrule: True
    params:
        code=code_fingerprint("workflow/scripts/misaligned_rates.py"),
    script:
        "../scripts/misaligned_rates.py"


rule figure3:
    """Figure 3: train on each attribution decile."""
    input:
        answers=for_each_dataset(figure3_runs),
        categories=config["question_categories"],
    output:
        "<results>/figures/figure3.csv",
    log:
        "<results>/figures/figure3.log",
    localrule: True
    params:
        code=code_fingerprint("workflow/scripts/misaligned_rates.py"),
    script:
        "../scripts/misaligned_rates.py"


rule figure4:
    """Figure 4: retrain transfer_targets on data ranked by each of transfer_sources."""
    input:
        answers=for_each_dataset(figure4_runs),
        categories=config["question_categories"],
    output:
        "<results>/figures/figure4.csv",
    log:
        "<results>/figures/figure4.log",
    localrule: True
    params:
        code=code_fingerprint("workflow/scripts/misaligned_rates.py"),
    script:
        "../scripts/misaligned_rates.py"


rule figure5:
    """Figure 5: retrain transfer_targets on data ranked by every model, at 20%."""
    input:
        answers=for_each_dataset(figure5_runs),
        categories=config["question_categories"],
    output:
        "<results>/figures/figure5.csv",
    log:
        "<results>/figures/figure5.log",
    localrule: True
    params:
        code=code_fingerprint("workflow/scripts/misaligned_rates.py"),
    script:
        "../scripts/misaligned_rates.py"


rule figure6:
    """Figure 6 (left): train on deciles of LLM-judged rubric scores."""
    input:
        answers=for_each_dataset(figure6_runs),
        categories=config["question_categories"],
    output:
        "<results>/figures/figure6.csv",
    log:
        "<results>/figures/figure6.log",
    localrule: True
    params:
        code=code_fingerprint("workflow/scripts/misaligned_rates.py"),
    script:
        "../scripts/misaligned_rates.py"


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
        code=code_fingerprint("workflow/scripts/figure6_spearman.py"),
    script:
        "../scripts/figure6_spearman.py"


rule appendix_a3_a4:
    """Appendix A3/A4: attribution queries built from part of the evaluation."""
    input:
        answers=for_each_dataset(appendix_a3_a4_runs),
        categories=config["question_categories"],
    output:
        "<results>/figures/appendix_a3_a4.csv",
    log:
        "<results>/figures/appendix_a3_a4.log",
    localrule: True
    params:
        code=code_fingerprint("workflow/scripts/misaligned_rates.py"),
    script:
        "../scripts/misaligned_rates.py"


rule appendix_a5:
    """Appendix A5: rank by loss and by length."""
    input:
        answers=for_each_dataset(appendix_a5_runs),
        categories=config["question_categories"],
    output:
        "<results>/figures/appendix_a5.csv",
    log:
        "<results>/figures/appendix_a5.log",
    localrule: True
    params:
        code=code_fingerprint("workflow/scripts/misaligned_rates.py"),
    script:
        "../scripts/misaligned_rates.py"


rule appendix_a6:
    """Appendix A6: Figure 1 with data repeated to hold the number of steps constant."""
    input:
        answers=for_each_dataset(appendix_a6_runs),
        categories=config["question_categories"],
    output:
        "<results>/figures/appendix_a6.csv",
    log:
        "<results>/figures/appendix_a6.log",
    localrule: True
    params:
        code=code_fingerprint("workflow/scripts/misaligned_rates.py"),
    script:
        "../scripts/misaligned_rates.py"


rule appendix_a7:
    """Appendix A7: Figure 2 with data repeated to hold the number of steps constant."""
    input:
        answers=for_each_dataset(appendix_a7_runs),
        categories=config["question_categories"],
    output:
        "<results>/figures/appendix_a7.csv",
    log:
        "<results>/figures/appendix_a7.log",
    localrule: True
    params:
        code=code_fingerprint("workflow/scripts/misaligned_rates.py"),
    script:
        "../scripts/misaligned_rates.py"


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
