@figure("appendix_a5", "Appendix A5: rank by loss and by length.")
def appendix_a5_runs(dataset):
    return baseline(dataset) + retrained(dataset, ["loss", "length"], extremes("remove", resampled=True))
