@figure("figure3", "Figure 3: train on each attribution decile.")
def figure3_runs(dataset):
    return baseline(dataset) + retrained(dataset, config["decile_methods"], DECILES)
