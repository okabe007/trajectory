from tools.config_loader import load_config
from tools.derived_constants import calculate_derived_constants
from core.simulation_core import SpermSimulation
import os

def run_simulation_from_ini(ini_path="sperm_config.ini"):
    constants = load_config(ini_path)
    constants = calculate_derived_constants(constants)

    result_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(result_dir, exist_ok=True)

    sim = SpermSimulation()
    sim.run(constants, result_dir, "simulation_result", save_flag=True)
    sim.plot_trajectories()
    sim.plot_movie_trajectories()
    print("[INFO] 実行完了：図と動画を保存しました。")
