@figure("figure4", "Figure 4: retrain transfer_targets on data ranked by each of transfer_sources.")
def figure4_runs(dataset):
    runs = []
    for target in config["transfer_targets"]:
        runs += baseline(dataset, model=target)
        for source in config["transfer_sources"]:
            runs += retrained(dataset, ["cosine"], extremes("remove", resampled=True), source=source, model=target)
    return runs
