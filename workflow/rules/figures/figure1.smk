@figure("figure1", "Figure 1: remove the most or least influential 1-20% of the data.")
def figure1_runs(dataset):
    return baseline(dataset) + retrained(dataset, config["methods"], extremes("remove"))
