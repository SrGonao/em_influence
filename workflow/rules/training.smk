rule training_config:
    """The model's LoRA template, pointed at this run's data, output directory and seed."""
    input:
        data=branch(
            evaluate("{trained_on} == 'full'"),
            then=dataset_of,
            otherwise="<results>/{dataset}/subsets/{trained_on}.jsonl",
        ),
        template=lookup("models/{model}/template", within=config),
    output:
        "<results>/{dataset}/runs/{model}/{trained_on}/seed{seed}/training.json",
    log:
        "<results>/{dataset}/runs/{model}/{trained_on}/seed{seed}/training_config.log",
    localrule: True
    params:
        model=lookup("models/{model}/id", within=config),
        code=code_fingerprint("em_influence/scripts/training_config.py"),
    shell:
        "python -m em_influence.scripts.training_config --template {input.template} --model {params.model}"
        " --training_file {input.data} --seed {wildcards.seed} --output {output} > {log} 2>&1"


rule train:
    """Fine-tune a LoRA adapter on the run's training data."""
    input:
        "<results>/{dataset}/runs/{model}/{trained_on}/seed{seed}/training.json",
    output:
        directory("<results>/{dataset}/runs/{model}/{trained_on}/seed{seed}/model"),
    log:
        "<results>/{dataset}/runs/{model}/{trained_on}/seed{seed}/train.log",
    resources:
        gpu=1,
        min_free_gpu_gib=config["min_free_gpu_gib"],
    params:
        # transformers loads the model through bitsandbytes and accelerate without importing them here.
        code=code_fingerprint("em_influence/scripts/training_lora.py", packages=("bitsandbytes", "accelerate")),
    shell:
        on_gpu("python em_influence/scripts/training_lora.py {input} > {log} 2>&1")


rule evaluate:
    """Sample answers to the evaluation questions and judge how aligned each is."""
    input:
        model="<results>/{dataset}/runs/{model}/{trained_on}/seed{seed}/model",
        questions=config["questions"],
    output:
        "<results>/{dataset}/runs/{model}/{trained_on}/seed{seed}/answers.csv",
    log:
        "<results>/{dataset}/runs/{model}/{trained_on}/seed{seed}/evaluate.log",
    resources:
        gpu=1,
        min_free_gpu_gib=config["min_free_gpu_gib"],
    params:
        model=prepend_param("--lora_path", input.model),
        samples=config["samples_per_question"],
        judge=config["judge_model"],
        # The judge runs locally on vLLM; openai and backoff serve only its API backend.
        code=code_fingerprint(
            "em_influence/scripts/generate_answers.py",
            "em_influence/scripts/judge_answers.py",
            packages=("transformers",),
            ignore=("openai", "backoff"),
        ),
    shell:
        on_gpu(
            "(python em_influence/scripts/generate_answers.py {params.model} --questions {input.questions}"
            " --output {output} --n_per_question {params.samples}"
            " && python em_influence/scripts/judge_answers.py {output} --questions {input.questions}"
            " --judge-model {params.judge}) > {log} 2>&1"
        )


use rule evaluate as evaluate_base with:
    input:
        questions=config["questions"],
    output:
        "<results>/base/{model}/answers.csv",
    log:
        "<results>/base/{model}/evaluate.log",
    params:
        model=base_model_flag,


workflow.get_rule("evaluate_base").docstring = "Like evaluate, for a model before any fine-tuning."
