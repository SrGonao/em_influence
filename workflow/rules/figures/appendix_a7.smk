@figure("appendix_a7", "Appendix A7: Figure 2 with data repeated to hold the number of steps constant.")
def appendix_a7_runs(dataset):
    return baseline(dataset) + retrained(dataset, config["methods"], extremes("select", resampled=True))
