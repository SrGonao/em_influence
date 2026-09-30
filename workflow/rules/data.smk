rule download_archive:
    """One password-locked archive from openai/emergent-misalignment-persona-features."""
    output:
        "<data>/archives/{archive}.zip",
    log:
        "<data>/archives/{archive}.log",
    localrule: True
    params:
        code=code_fingerprint("em_influence/data_prep.py"),
    shell:
        "python -m em_influence.data_prep download --archive {wildcards.archive} --output {output}" " > {log} 2>&1"


rule prepare_data:
    """One domain's incorrect-advice dataset, holding out the narrow-eval prompts."""
    input:
        archive=training_archive_of,
        held_out=held_out_questions,
    output:
        "<data>/{dataset}.jsonl",
    log:
        "<data>/{dataset}.log",
    localrule: True
    params:
        held_out=prepend_param("--held_out", input.held_out),
        code=code_fingerprint("em_influence/data_prep.py"),
    shell:
        "python -m em_influence.data_prep prepare --archive {input.archive} {params.held_out}"
        " --output {output} > {log} 2>&1"


rule subset:
    """The rows of a dataset that one subset (e.g. remove_top_0.2, decile_3) of an attribution keeps."""
    input:
        data=dataset_of,
        attributions="<results>/{dataset}/attributions/{source}/{method}/attributions.csv",
    output:
        "<results>/{dataset}/subsets/{source}/{method}/{subset}.jsonl",
    log:
        "<results>/{dataset}/subsets/{source}/{method}/{subset}.log",
    localrule: True
    params:
        deciles=config["deciles"],
        code=code_fingerprint("em_influence/selection.py"),
    shell:
        "python -m em_influence.selection --dataset {input.data} --attributions {input.attributions}"
        " --subset {wildcards.subset} --deciles {params.deciles} --output {output} > {log} 2>&1"
