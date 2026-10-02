@figure("token_decile_removal", "Mask each attribution decile of reply tokens, one at a time.")
def token_decile_removal_runs(dataset):
    removed = [f"remove_{decile}" for decile in DECILES]
    return token_baseline(dataset) + retrained(dataset, config["token_decile_methods"], intervened(removed))
