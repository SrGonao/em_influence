@figure("token_tilted_removal", "Mask 10% of reply tokens, drawn at random but tilted to reach each target summed score.")
def token_tilted_removal_runs(dataset):
    tilts = [f"tilt_{tilt}_0.1" for tilt in config["token_tilts"]]
    return token_baseline(dataset) + retrained(dataset, config["token_tilt_methods"], intervened(tilts))
