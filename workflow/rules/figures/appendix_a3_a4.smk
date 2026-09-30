@figure("appendix_a3_a4", "Appendix A3/A4: attribution queries built from part of the evaluation.")
def appendix_a3_a4_runs(dataset):
    methods = [f"cosine@{suite}" for suite in config["query_suites"]]
    return baseline(dataset) + retrained(dataset, methods, DECILES)
