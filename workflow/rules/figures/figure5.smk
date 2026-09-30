@figure("figure5", "Figure 5: retrain transfer_targets on data ranked by every model, at 20%.")
def figure5_runs(dataset):
    runs = []
    for source in MODELS:
        runs += baseline(dataset, model=source)
    for target in config["transfer_targets"]:
        for source in MODELS:
            runs += retrained(
                dataset, ["cosine"], extremes("remove", [0.2], resampled=True), source=source, model=target
            )
    return runs
