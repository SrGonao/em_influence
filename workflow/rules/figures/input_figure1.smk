@figure(
    "input_figure1", "Figure 1 on input tokens: replace the most or least influential 1-20% of prompt and reply tokens."
)
def input_figure1_runs(dataset):
    ranked = [method for method in config["input_methods"] if method != "tokens-random-input"]
    # Random's top and bottom are both random draws, so one of them is the control.
    random = [method for method in config["input_methods"] if method == "tokens-random-input"]
    return (
        token_baseline(dataset)
        + retrained(dataset, ranked, replaced(extremes("replace")))
        + retrained(dataset, random, replaced([f"replace_top_{fraction}" for fraction in FRACTIONS]))
    )
