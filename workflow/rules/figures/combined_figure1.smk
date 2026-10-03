# Label rankings paired with an input ranking: tokens-random goes with tokens-random-input.
COMBINED_PAIRS = [
    "tokens-ekfac-output+tokens-ekfac-input",
    "tokens-ekfac+tokens-ekfac-input",
    "tokens-random+tokens-random-input",
]
# Each label intervention with its input replacement.
COMBINED_INTERVENTIONS = [("", "zero"), ("_sample", "sample")]


@figure(
    "combined_figure1",
    "Output and input mode together: change the top 1-20% of labels and replace the top 1-20% of inputs.",
)
def combined_figure1_runs(dataset):
    subsets = [
        f"remove_top_{fraction}{label}+replace_top_{fraction}_{replacement}"
        for label, replacement in COMBINED_INTERVENTIONS
        for fraction in FRACTIONS
    ]
    return token_baseline(dataset) + retrained(dataset, COMBINED_PAIRS, subsets)
