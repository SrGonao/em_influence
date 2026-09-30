@figure("figure2", "Figure 2: train on only the most or least influential 1-20%.")
def figure2_runs(dataset):
    return baseline(dataset) + retrained(dataset, config["methods"], extremes("select"))
