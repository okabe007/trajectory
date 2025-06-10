
def debug_trajectory(trajectory):
    print(f"[DEBUG] trajectory.shape = {trajectory.shape}")
    if trajectory.shape[0] > 0 and trajectory.shape[1] > 0:
        print(f"[DEBUG] trajectory[0, :5, :] =\n{trajectory[0, :5, :]}")
    else:
        print("[DEBUG] trajectory is empty or has zero steps.")
