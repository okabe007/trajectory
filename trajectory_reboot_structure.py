# trajectory_reboot/entry.py

from tools.config_loader import load_config
from tools.derived_constants import calculate_derived_constants
from simulation_driver import run_simulation_from_ini
from spermsim.initialization import SimApp  # GUI用

if __name__ == "__main__":
    mode = "cli"  # "cli" または "gui" に切り替え可能

    if mode == "cli":
        print("[ENTRY] CLIモードで実行します")
        run_simulation_from_ini("sperm_config.ini")

    elif mode == "gui":
        print("[ENTRY] GUIモードで起動します")
        app = SimApp()
        app.run_gui()


# trajectory_reboot/simulation_driver.py

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


# trajectory_reboot/spermsim/simulation_core.py

import numpy as np
import os
from tools.io_checks import IO_check_cube, IO_check_drop, IO_check_spot
from tools.geometry import CubeShape, DropShape, SpotShape
from enums import IOStatus

class SpermSimulation:
    def __init__(self):
        self.trajectory = None
        self.vectors = None
        self.constants = {}

    def run(self, constants, result_dir, save_name, save_flag=True):
        self.constants = constants
        shape = constants["shape"]
        step_len = constants["step_length"]
        hz = constants["sample_rate_hz"]
        deviation = constants["deviation"]
        seed = int(constants.get("seed_number", 0))
        num_sperm = int(constants["sperm_conc"] * constants["vol"] * 1e-3)
        num_steps = int(constants["sim_min"] * hz * 60)

        rng = np.random.default_rng(seed)

        # 初期化
        self.trajectory = np.zeros((num_sperm, num_steps, 3))
        self.vectors = np.zeros((num_sperm, num_steps, 3))

        if shape == "drop":
            shape_obj = DropShape(constants)
        elif shape == "cube":
            shape_obj = CubeShape(constants)
        elif shape == "spot":
            shape_obj = SpotShape(constants)
        else:
            raise ValueError(f"Unknown shape: {shape}")

        for i in range(num_sperm):
            pos = shape_obj.initial_position()
            vec = rng.normal(0, 1, 3)
            vec /= np.linalg.norm(vec) + 1e-12

            self.trajectory[i, 0] = pos
            self.vectors[i, 0] = vec

            for t in range(1, num_steps):
                vec += rng.normal(0, deviation, 3)
                vec /= np.linalg.norm(vec) + 1e-12
                candidate = pos + vec * step_len

                if shape == "cube":
                    status, _ = IO_check_cube(candidate, constants)
                elif shape == "drop":
                    status, _ = IO_check_drop(candidate, 0, constants)
                elif shape == "spot":
                    status = IO_check_spot(pos, candidate, constants, "inside", 0)
                else:
                    status = IOStatus.INSIDE

                if status == IOStatus.INSIDE:
                    pos = candidate

                self.trajectory[i, t] = pos
                self.vectors[i, t] = vec

        if save_flag:
            np.save(os.path.join(result_dir, save_name + "_trajectory.npy"), self.trajectory)
            np.save(os.path.join(result_dir, save_name + "_vectors.npy"), self.vectors)

    def plot_trajectories(self, save_path=None):
        import matplotlib.pyplot as plt
        import datetime

        if self.trajectory is None:
            print("[WARNING] 軌跡データがありません")
            return

        fig, ax = plt.subplots()
        for traj in self.trajectory:
            ax.plot(traj[:, 0], traj[:, 1], alpha=0.5)

        if save_path is None:
            dt = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            base = os.path.dirname(__file__)
            outdir = os.path.abspath(os.path.join(base, "..", "figs_and_movies"))
            os.makedirs(outdir, exist_ok=True)
            save_path = os.path.join(outdir, f"trajectory_{dt}.png")

        plt.savefig(save_path)
        plt.close()
        print(f"[INFO] 図を {save_path} に保存しました。")

    def plot_movie_trajectories(self, save_path=None, fps=5):
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        import datetime

        if self.trajectory is None:
            print("[WARNING] 軌跡データがありません")
            return

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        lines = [ax.plot([], [], [], lw=0.7)[0] for _ in self.trajectory]

        def init():
            for ln in lines:
                ln.set_data([], [])
                ln.set_3d_properties([])
            return lines

        def update(frame):
            for i, ln in enumerate(lines):
                ln.set_data(self.trajectory[i, :frame+1, 0], self.trajectory[i, :frame+1, 1])
                ln.set_3d_properties(self.trajectory[i, :frame+1, 2])
            return lines

        anim = FuncAnimation(fig, update, init_func=init, frames=self.trajectory.shape[1], interval=1000/fps)

        if save_path is None:
            dt = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            base = os.path.dirname(__file__)
            outdir = os.path.abspath(os.path.join(base, "..", "figs_and_movies"))
            os.makedirs(outdir, exist_ok=True)
            save_path = os.path.join(outdir, f"trajectory_{dt}.mp4")

        try:
            anim.save(save_path, writer='ffmpeg', fps=fps)
        except:
            anim.save(save_path, writer='pillow', fps=fps)

        plt.close(fig)
        print(f"[INFO] 動画を {save_path} に保存しました。")
