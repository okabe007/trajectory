import os
import numpy as np
from tools.io_checks import IO_check_cube, IO_check_drop, IO_check_spot
from tools.enums import IOStatus
from tools.plot_utils import plot_2d_trajectories, plot_3d_trajectories, draw_3d_movies
from tools.geometry import CubeShape, DropShape, SpotShape
# from simcore.motion_modes.cube_mode import CubeMode
# from simcore.motion_modes.drop_mode import DropMode
# from simcore.motion_modes.spot_mode import SpotMode
from simcore.motion_modes.reflection_mode import ReflectionMode


import numpy as np
from tools.derived_constants import calculate_derived_constants

class SimulationEngine:
    def __init__(self, constants: dict):
        print("[DEBUG] SimulationEngine.__init__ が呼び出されました")
        self.constants = calculate_derived_constants(constants)
        self.shape = self.constants.get("shape", "cube").lower()
        self.number_of_sperm = self.constants.get("number_of_sperm", 10)
        self.number_of_steps = int(self.constants.get("sim_min", 1.0) * self.constants.get("sample_rate_hz", 4.0) * 60)
        self.step_length = self.constants.get("step_length", 0.01)
        self.seed = int(self.constants.get("seed_number", 0))
        self.rng = np.random.default_rng(self.seed)
        self.motion_mode = self._select_mode()
        print(f"[DEBUG] 選択されたモード: {self.shape}")
    def run(self, constants: dict, result_dir: str, save_name: str, save_flag: bool = False):
        """
        全てのshape（cube, drop, spot, ceros）に対応した精子運動シミュレーション実行関数。
        self.trajectory および self.vectors を更新する。
        """
        

        self.constants = calculate_derived_constants(constants)
        shape = self.constants["shape"]
        step_len = self.constants["step_length"]
        hz = self.constants["sample_rate_hz"]
        deviation = self.constants["deviation"]
        seed = int(self.constants.get("seed_number", 0)) if self.constants.get("seed_number") not in [None, "", "None"] else int(time.time() * 1000) % (2**32)

        self.number_of_sperm = int(self.constants["sperm_conc"] * self.constants["vol"] * 1e-3)
        self.number_of_steps = int(self.constants["sim_min"] * hz * 60)

        rng = np.random.default_rng(seed)

        # === 初期位置と形状オブジェクト ===
        if shape == "drop":
            shape_obj = DropShape(self.constants)
        elif shape == "cube":
            shape_obj = CubeShape(self.constants)
        elif shape == "spot":
            shape_obj = SpotShape(self.constants)
        elif shape == "ceros":
            shape_obj = None
        else:
            raise ValueError(f"Unknown shape: {shape}")

        if shape == "ceros":
            self.initial_position = np.full((self.number_of_sperm, 3), np.inf)
        else:
            self.initial_position = np.zeros((self.number_of_sperm, 3))
            for j in range(self.number_of_sperm):
                self.initial_position[j] = shape_obj.initial_position()

        # === 初期ベクトル（ランダム方向） ===
        self.initial_vectors = np.zeros((self.number_of_sperm, 3))
        for j in range(self.number_of_sperm):
            vec = rng.normal(0, 1, 3)
            vec /= np.linalg.norm(vec) + 1e-12
            self.initial_vectors[j] = vec

        # === 配列初期化 ===
        self.trajectory = np.zeros((self.number_of_sperm, self.number_of_steps, 3))
        self.vectors = np.zeros((self.number_of_sperm, self.number_of_steps, 3))

        # === メインループ ===
        for j in range(self.number_of_sperm):
            pos = self.initial_position[j].copy()
            vec = self.initial_vectors[j].copy()
            stick_status = 0
            prev_stat = "inside"

            self.trajectory[j, 0] = pos
            self.vectors[j, 0] = vec

            for i in range(1, self.number_of_steps):
                vec += rng.normal(0, deviation, 3)
                vec /= np.linalg.norm(vec) + 1e-12
                candidate = pos + vec * step_len

                # IO 判定とステータスごとの処理
                if shape == "cube":
                    status, _ = IO_check_cube(candidate, self.constants)
                elif shape == "drop":
                    status, stick_status = IO_check_drop(candidate, stick_status, self.constants)
                    if status == IOStatus.ON_POLYGON:
                        candidate, vec, stick_status, status = self.drop_polygon_move(pos, vec, stick_status, self.constants)
                elif shape == "spot":
                    status = IO_check_spot(pos, candidate, self.constants, prev_stat, stick_status)
                    prev_stat = status
                elif shape == "ceros":
                    status = IOStatus.INSIDE
                else:
                    raise ValueError(f"Unknown shape: {shape}")

                if status in [IOStatus.INSIDE]:
                    pos = candidate
                elif status == IOStatus.ON_POLYGON:
                    pass  # 上記drop内で処理済
                elif status in [IOStatus.REFLECT]:
                    vec *= -1
                elif status in [IOStatus.STICK]:
                    stick_status = int(hz)
                elif status in [IOStatus.BORDER, IOStatus.BOTTOM_OUT, IOStatus.SPOT_EDGE_OUT]:
                    pass  # その場に留まる
                else:
                    print(f"[WARNING] Unexpected status: {status}")

                self.trajectory[j, i] = pos
                self.vectors[j, i] = vec

        print("[DEBUG] 初期位置数:", len(self.initial_position))
        print("[DEBUG] 精子数:", self.number_of_sperm)
        self.trajectories = self.trajectory  # 外部用

        # 保存処理
        if save_flag:
            os.makedirs(result_dir, exist_ok=True)
            np.save(os.path.join(result_dir, f"{save_name}.npy"), self.trajectory)

        # 描画処理
        display_mode = self.constants.get("display_mode", "2D")
        if display_mode == "2D":
            plot_2d_trajectories(self.trajectory, self.constants)
        elif display_mode == "movie":
            draw_3d_movies(self.trajectory, self.constants)
        else:
            print(f"[WARNING] 未対応の表示モード: {display_mode}")

    def _select_mode(self):
        if self.shape == "cube":
            return CubeMode(self.constants)
        elif self.shape == "drop":
            return DropMode(self.constants)
        elif self.shape == "spot":
            return SpotMode(self.constants)
        elif self.shape == "reflection":
            return ReflectionMode(self.constants)
        else:
            raise ValueError(f"[ERROR] Unknown shape mode: {self.shape}")

    
    def simulate(self, on_progress=None):
        initial_position = self.constants['initial_position']
        trajectories = np.full((self.number_of_sperm, self.number_of_steps, 3), np.nan)
        vectors = np.zeros((self.number_of_sperm, self.number_of_steps, 3))

        for j in range(self.number_of_sperm):
            position = self._generate_initial_position()
            vector = np.array([0, 0, 1])  # 初期方向を適切に設定
            stick_status = 0  # 初期吸着状態

            for i in range(self.number_of_steps):
                position, vector, stick_status, status = self.motion_mode.drop_polygon_move(
                    position, vector, stick_status, self.constants
                )

                # 常に軌跡を記録する（警告があってもデータは保存される）
                trajectories[j, i] = position
                vectors[j, i] = vector

                if status not in [IOStatus.ON_POLYGON, IOStatus.INSIDE]:
                    print(f"[WARNING]engine.py内部の警告 Unexpected status: {status}")

            if on_progress:
                on_progress(j, self.number_of_sperm)
        
        print("[確認用] 軌跡データ:", trajectories[0, :10, :])  # 軌跡確認用
        return trajectories, vectors

    
    def simulate(self, on_progress=None):
        initial_position = self.constants['initial_position']
        print(f"確認：初期位置 = {initial_position}")
        print("[DEBUG] engine.py/simulate() 開始")
        trajectories = np.full((self.number_of_sperm, self.number_of_steps, 3), np.nan)
        vectors = np.zeros((self.number_of_sperm, self.number_of_steps, 3))

        for j in range(self.number_of_sperm):
            print(f"[DEBUG] engine.py/simulate(): 精子 {j} の初期位置生成")
            init_pos = self._generate_initial_position()
            print(f"[DEBUG] engine.py/simulate(): 精子 {j} 軌道計算開始")
            traj, vecs = self.motion_mode.simulate_trajectory(j, init_pos, self.rng)
            print(f"[DEBUG] engine.py/simulate(): 精子 {j} 軌道計算完了")
            trajectories[j] = traj
            vectors[j] = vecs

            if on_progress:
                on_progress(j, self.number_of_sperm)

        print("[DEBUG] engine.py/simulate() 完了")
        return trajectories, vectors

    def _generate_initial_position(self) -> np.ndarray:
        x_len = self.constants.get("medium_x_len", 1.0)
        y_len = self.constants.get("medium_y_len", 1.0)
        z_len = self.constants.get("medium_z_len", 1.0)

        return self.rng.uniform(
            low=[-x_len/2, -y_len/2, -z_len/2],
            high=[x_len/2, y_len/2, z_len/2]
        )
