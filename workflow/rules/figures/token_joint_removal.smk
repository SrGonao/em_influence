@figure("token_joint_removal", "Mask 10% of reply tokens drawn to hit target summed scores under two rankings at once.")
def token_joint_removal_runs(dataset):
    joints = [f"joint_{a}_{b}_0.1" for a, b in config["token_joints"]]
    return token_baseline(dataset) + retrained(dataset, config["token_joint_methods"], intervened(joints))
