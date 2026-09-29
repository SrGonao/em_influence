rule prepare_data:
    """Download one domain's incorrect-advice dataset, holding out the narrow-eval prompts."""
    input:
        held_out_questions,
    output:
        "<data>/{dataset}.jsonl",
    log:
        "<data>/{dataset}.log",
    localrule: True
    params:
        code=code_fingerprint("workflow/scripts/prepare_data.py"),
    shell:
        "python workflow/scripts/prepare_data.py --dataset {wildcards.dataset} --output {output} > {log} 2>&1"


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
        code=code_fingerprint("workflow/scripts/subset.py"),
    shell:
        "python workflow/scripts/subset.py --data {input.data} --attributions {input.attributions}"
        " --subset {wildcards.subset} --deciles {params.deciles} --output {output} > {log} 2>&1"
