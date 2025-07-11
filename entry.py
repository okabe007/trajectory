import os
import sys

# ✅ ① プロジェクトルートを sys.path に追加（trajectory_reboot）
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# ✅ ② GUIと設定管理の import
from tkinter import Tk
from spermsim.initialization import SimApp
from tools.ini_handler import load_config
from tools.derived_constants import calculate_derived_constants

# ✅ ③ シミュレーションクラス（SimulationEngine or SpermSimulation）
from core.engine import SpermSimulation


def main():
    print("[ENTRY] GUIを起動します...")
    root = Tk()
    app = SimApp(root)
    root.mainloop()

    if getattr(app, "simulation_ran", False):
        print("[ENTRY] Simulation already executed from GUI. Exiting.")
        return

    # 1. .ini 読み込み（一次変数のみ含まれる前提）
    config_path = "sperm_config.ini"
    if not os.path.exists(config_path):
        raise FileNotFoundError("[ERROR] sperm_config.ini が見つかりません。保存して終了を押しましたか？")

    constants = load_config()

    # 2. 二次変数の導出（egg_center, drop_r, x_min, ...）
    constants = calculate_derived_constants(constants)

    # 3. シミュレーション実行
    result_dir = "results"
    os.makedirs(result_dir, exist_ok=True)

    sim = SpermSimulation(constants)
    sim.run(constants, result_dir, "simulation_result", save_flag=True)

    # sim.plot_trajectories()
    # sim.plot_movie_trajectories()

    print("[ENTRY] シミュレーション完了。")

if __name__ == "__main__":
    main()
