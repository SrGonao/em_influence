# Gradient-routed runs (a token subset ending _route, _route_retain..., or route_minmax) train adapter A and a routed
# adapter B; the run's own evaluation (answers.csv) is A alone, with B ablated. These rules evaluate the two other
# models a routed run gives, as <variant>_answers.csv beside it (and ft_narrow_answers.csv):
#   ft      A fine-tuned for a few steps after ablating B (finetune_after_ablation.py; Cloud et al. 2024)
#   with_b  A and B both kept, merged into the base weights (merge_routed.py)
ROUTED_RUN = r"[^/]+/[^/]+/[^/]*route[^/]*"


rule finetune_after_ablation:
    """Adapter A of a routed run, fine-tuned with B ablated on a few of its training documents."""
    input:
        model=run_model,
        data=training_data,
    output:
        directory("<results>/{dataset}/runs/{model}/{trained_on}/seed{seed}/model_ft"),
    log:
        "<results>/{dataset}/runs/{model}/{trained_on}/seed{seed}/finetune_after_ablation.log",
    wildcard_constraints:
        trained_on=ROUTED_RUN,
    resources:
        gpu=1,
    params:
        base=lookup("models/{model}/id", within=config),
    shell:
        step(
            "python -m em_influence.scripts.finetune_after_ablation --base-model {params.base} --adapter {input.model}"
            " --data {input.data} --output {output}",
            gpu=True,
        )


rule merge_routed:
    """A routed run's model with B kept: both adapters merged into the base weights. It's a full copy of the model, so
    it's deleted once the runs that sample it have."""
    input:
        model=run_model,
    output:
        temp(directory("<results>/{dataset}/runs/{model}/{trained_on}/seed{seed}/model_with_b")),
    log:
        "<results>/{dataset}/runs/{model}/{trained_on}/seed{seed}/merge_routed.log",
    wildcard_constraints:
        trained_on=ROUTED_RUN,
    resources:
        gpu=1,
    params:
        base=lookup("models/{model}/id", within=config),
    shell:
        step(
            "python -m em_influence.scripts.merge_routed --base-model {params.base} --adapter {input.model}"
            " --output {output}",
            gpu=True,
        )


use rule evaluate as evaluate_routed with:
    input:
        model="<results>/{dataset}/runs/{model}/{trained_on}/seed{seed}/model_{variant}",
        questions=config["questions"],
    output:
        "<results>/{dataset}/runs/{model}/{trained_on}/seed{seed}/{variant}_answers.csv",
    log:
        "<results>/{dataset}/runs/{model}/{trained_on}/seed{seed}/evaluate_{variant}.log",
    wildcard_constraints:
        trained_on=ROUTED_RUN,
        variant="ft|with_b",


workflow.get_rule("evaluate_routed").docstring = "Like evaluate, for a routed run's ft or with_b model."


use rule evaluate as evaluate_routed_narrow with:
    input:
        model="<results>/{dataset}/runs/{model}/{trained_on}/seed{seed}/model_{variant}",
        questions=narrow_questions,
    output:
        "<results>/{dataset}/runs/{model}/{trained_on}/seed{seed}/{variant}_narrow_answers.csv",
    log:
        "<results>/{dataset}/runs/{model}/{trained_on}/seed{seed}/evaluate_{variant}_narrow.log",
    wildcard_constraints:
        trained_on=ROUTED_RUN,
        variant="ft",
    params:
        samples=config["narrow_samples_per_question"],


workflow.get_rule("evaluate_routed_narrow").docstring = (
    "Like evaluate_narrow, for a routed run's ft model (the routing report compares with_b on broad questions only)."
)
