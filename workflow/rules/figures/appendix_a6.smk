@figure("appendix_a6", "Appendix A6: Figure 1 with data repeated to hold the number of steps constant.")
def appendix_a6_runs(dataset):
    return baseline(dataset) + retrained(dataset, config["methods"], extremes("remove", resampled=True))
