# === 標準ライブラリ ===
import os
import math
import time
from datetime import datetime
from typing import Tuple

# === サードパーティライブラリ ===
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.animation import FuncAnimation

# === 自作モジュール: tools ===
from tools.enums import IOStatus
from tools.derived_constants import calculate_derived_constants
from tools.io_checks import IO_check_cube, IO_check_drop, IO_check_spot
from tools.plot_utils import (
    plot_2d_trajectories,
    plot_3d_trajectories,
    draw_3d_movies
)
from tools.geometry import (
    CubeShape,
    DropShape,
    SpotShape,
    CerosShape,
    
)

def __init__(self, constants):
    self.constants = constants
    float_keys = ['spot_angle', 'vol', 'sperm_conc', 'vsl', 'deviation', 'surface_time', 'gamete_r', 'sim_min', 'sample_rate_hz']
    int_keys = ['sim_repeat']
    for key in float_keys:
        if key in self.constants and (not isinstance(self.constants[key], float)):
            try:
                self.constants[key] = float(self.constants[key])
            except Exception:
                print(f'[WARNING] {key} = {self.constants[key]} をfloat変換できませんでした')
    for key in int_keys:
        if key in self.constants and (not isinstance(self.constants[key], int)):
            try:
                self.constants[key] = int(float(self.constants[key]))
            except Exception:
                print(f'[WARNING] {key} = {self.constants[key]} をint変換できませんでした')
    self.constants = calculate_derived_constants(self.constants)
# def is_vector_meeting_egg(self, base_position, temp_position, egg_center, gamete_r):
#     vector = temp_position - base_position
#     distance_base = LA.norm(base_position - egg_center)
#     distance_tip = LA.norm(temp_position - egg_center)
#     if distance_base <= gamete_r or distance_tip <= gamete_r:
#         return True
#     f = base_position - egg_center
#     a = vector @ vector
#     b = 2 * (f @ vector)
#     c = f @ f - gamete_r ** 2
#     discriminant = b ** 2 - 4 * a * c
#     if discriminant < 0:
#         return False
#     sqrt_discriminant = np.sqrt(discriminant)
#     t1 = (-b - sqrt_discriminant) / (2 * a)
#     t2 = (-b + sqrt_discriminant) / (2 * a)
#     if 0 <= t1 <= 1 or 0 <= t2 <= 1:
#         return True
#     return False


def _make_local_basis(forward: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return two unit vectors orthogonal to ``forward``."""
    f = forward / (np.linalg.norm(forward) + 1e-12)
    if abs(f[0]) < 0.9:
        base = np.array([1.0, 0.0, 0.0])
    else:
        base = np.array([0.0, 1.0, 0.0])
    y = np.cross(f, base)
    y /= np.linalg.norm(y) + 1e-12
    x = np.cross(y, f)
    return x, y
def _perturb_direction(prev: np.ndarray, deviation: float, rng: np.random.Generator) -> np.ndarray:
    """Return a unit vector deviated from ``prev``."""
    lx, ly = _make_local_basis(prev)
    theta = rng.normal(0.0, deviation)
    phi = rng.uniform(-np.pi, np.pi)
    new_dir = (
        np.cos(theta) * prev
        + np.sin(theta) * (np.cos(phi) * lx + np.sin(phi) * ly)
    )
    new_dir /= np.linalg.norm(new_dir) + 1e-12
    return new_dir
from enum import Enum

class SpotIO(Enum):
    INSIDE = "inside"
    BORDER = "border"
    SPHERE_OUT = "sphere_out"
    BOTTOM_OUT = "bottom_out"
    SPOT_EDGE_OUT = "spot_edge_out"
    ON_POLYGON = "ON_POLYGON"
    SPOT_BOTTOM = "spot_bottom"
    REFLECT = "reflect"     # ← spot の特殊反射（必要なら）
    STICK = "stick"         # ← spot の貼り付き
def _spot_status_check(
    base_position: np.ndarray,
    temp_position: np.ndarray,
    constants: dict,
    prev_stat: str = "inside",
    stick_status: int = 0,
) -> str:
    """Return IO status for spot shape without modifying the position."""


# === Imported from simulation.py ===
class SpermSimulation:
    def __init__(self, constants):
        self.constants = constants

        # --- 型安全化：数値パラメータはfloat/intに変換 ---
        float_keys = [
            "spot_angle", "vol", "sperm_conc", "vsl", "deviation", "surface_time",
            "gamete_r", "sim_min", "sample_rate_hz"
        ]
        int_keys = [
            "sim_repeat"
        ]
        for key in float_keys:
            if key in self.constants and not isinstance(self.constants[key], float):
                try:
                    self.constants[key] = float(self.constants[key])
                except Exception:
                    print(f"[WARNING] {key} = {self.constants[key]} をfloat変換できませんでした")
        for key in int_keys:
            if key in self.constants and not isinstance(self.constants[key], int):
                try:
                    self.constants[key] = int(float(self.constants[key]))
                except Exception:
                    print(f"[WARNING] {key} = {self.constants[key]} をint変換できませんでした")
        # shape, egg_localization, などはstr型のままでOK
        self.constants = calculate_derived_constants(self.constants)
        
    from tools.plot_utils import plot_2d_trajectories, plot_3d_trajectories, draw_3d_movies
    import numpy as np
    import os


    def simulate(self, constants: dict) -> np.ndarray:
        """
        精子運動の軌跡を計算し、(N精子 × Tステップ × 3座標) の配列として返す。
        """
        number_of_sperm = constants["number_of_sperm"]
        number_of_steps = int(constants["sim_min"] * 60 * constants["sample_rate_hz"])
        step_length = constants["step_length"]
        rng = np.random.default_rng()

        trajectory = np.zeros((number_of_sperm, number_of_steps, 3))
        # 必要ならベクトル保存も追加: vecs = np.zeros((number_of_sperm, number_of_steps, 3))

        for j in range(number_of_sperm):
            pos = np.array([0.0, 0.0, 0.0])  # 初期位置（例）
            vec = np.array([0.0, 0.0, 1.0])  # 初期位置（例）

            for i in range(number_of_steps):
                # ベクトルを偏向させる
                vec = _perturb_direction(vec, constants['deviation'], rng)

                # 次の位置を計算
                candidate = pos + vec * step_length

                # IOチェックや壁処理など（drop, cube, spotのモードごとに）
                # status = IO_check_xxx(candidate, constants)
                # 必要なら candidate, vec = drop_polygon_move(...) など

                trajectory[j, i] = candidate
                pos = candidate

        return trajectory




    def run(self, constants: dict, result_dir: str, save_name: str, save_flag: bool = False):
        # 派生変数の再計算（引数が更新されていた場合に備えて）
        constants = calculate_derived_constants(constants)
        print("[DEBUG] run() 開始: constants =", constants)

        # self.trajectory をこの中で生成するようにしておくこと
        self.trajectory = self.simulate(constants)  # ← simulate() は self 内部で定義されている仮定

        # 保存オプションが有効な場合
        if save_flag:
            save_path = os.path.join(result_dir, save_name + ".npy")
            np.save(save_path, self.trajectory)
            print(f"[DEBUG] 軌跡データを {save_path} に保存しました")

        # 表示モードの形式を整える（タプルやリストのまま来ることがある）
        display_mode = constants.get("display_mode", "2D")
        if isinstance(display_mode, (list, tuple)):
            display_mode = display_mode[0]
        # 余計な記号を取り除いて小文字化する
        display_mode = str(display_mode).strip(" ()'\" ,").lower()

        # 描画
        if display_mode == "2d":
            plot_2d_trajectories(self.trajectory, constants)
        elif display_mode == "3d":
            plot_3d_trajectories(self.trajectory, constants)
        elif display_mode == "movie":
            draw_3d_movies(self.trajectory, constants)
        else:
            print(f"[WARNING] 未対応の表示モード: {display_mode}")

    def is_vector_meeting_egg(self, base_position, temp_position, egg_center, gamete_r):
        vector = temp_position - base_position
        # if LA.norm(vector) < 1e-9:
        #     raise RuntimeError("zzz")
        distance_base = LA.norm(base_position - egg_center)
        distance_tip = LA.norm(temp_position - egg_center)
        if distance_base <= gamete_r or distance_tip <= gamete_r:
            return True
        f = base_position - egg_center
        a = vector @ vector
        b = 2 * (f @ vector)
        c = f @ f - gamete_r**2
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            return False
        sqrt_discriminant = np.sqrt(discriminant)
        t1 = (-b - sqrt_discriminant) / (2*a)
        t2 = (-b + sqrt_discriminant) / (2*a)
        if (0 <= t1 <= 1) or (0 <= t2 <= 1):
            return True
        return False
    
    