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
    shell:
        "python workflow/scripts/training_config.py --template {input.template} --model {params.model}"
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
        samples=config["samples_per_question"],
        judge=config["judge_model"],
    shell:
        on_gpu(
            "(python em_influence/scripts/generate_answers.py --lora_path {input.model} --questions {input.questions}"
            " --output {output} --n_per_question {params.samples}"
            " && python em_influence/scripts/judge_answers.py {output} --questions {input.questions}"
            " --judge-model {params.judge}) > {log} 2>&1"
        )


rule evaluate_base:
    """Like evaluate, for a model before any fine-tuning."""
    input:
        config["questions"],
    output:
        "<results>/base/{model}/answers.csv",
    log:
        "<results>/base/{model}/evaluate.log",
    resources:
        gpu=1,
        min_free_gpu_gib=config["min_free_gpu_gib"],
    params:
        model=lookup("models/{model}/id", within=config),
        samples=config["samples_per_question"],
        judge=config["judge_model"],
    shell:
        on_gpu(
            "(python em_influence/scripts/generate_answers.py --model {params.model} --questions {input}"
            " --output {output} --n_per_question {params.samples}"
            " && python em_influence/scripts/judge_answers.py {output} --questions {input}"
            " --judge-model {params.judge}) > {log} 2>&1"
        )
