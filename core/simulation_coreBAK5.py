import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from datetime import datetime
import math
import os
from matplotlib.animation import FuncAnimation
from typing import Tuple
from numpy import linalg as LA
from tools.io_checks import IO_check_drop
from tools.derived_constants import calculate_derived_constants
from tools.plot_utils import plot_2d_trajectories, plot_3d_movie_trajectories
from tools.geometry import CubeShape, DropShape, SpotShape, CerosShape, _handle_drop_outside
from tools.enums import IOStatus
from simcore.motion_modes.drop_mode import drop_polygon_move
from tools.io_checks import IO_check_cube, IO_check_drop, IO_check_spot

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
def is_vector_meeting_egg(self, base_position, temp_position, egg_center, gamete_r):
    vector = temp_position - base_position
    distance_base = LA.norm(base_position - egg_center)
    distance_tip = LA.norm(temp_position - egg_center)
    if distance_base <= gamete_r or distance_tip <= gamete_r:
        return True
    f = base_position - egg_center
    a = vector @ vector
    b = 2 * (f @ vector)
    c = f @ f - gamete_r ** 2
    discriminant = b ** 2 - 4 * a * c
    if discriminant < 0:
        return False
    sqrt_discriminant = np.sqrt(discriminant)
    t1 = (-b - sqrt_discriminant) / (2 * a)
    t2 = (-b + sqrt_discriminant) / (2 * a)
    if 0 <= t1 <= 1 or 0 <= t2 <= 1:
        return True
    return False
def run(self, sim_repeat: int, surface_time: float, sample_rate_hz: int):
    from tools.io_checks import IO_check_drop, IO_check_spot, IO_check_cube
    from tools.egg_placement import _egg_position
    from enums import IOStatus
    print('[DEBUG] SpermSimulationCOREで パラメータ:', self.constants)
    
    shape = self.constants.get('shape', 'cube')
    if shape == 'cube':
        shape_obj = CubeShape(self.constants)
    elif shape == 'spot':
        shape_obj = SpotShape(self.constants)
    elif shape == 'drop':
        shape_obj = DropShape(self.constants)
    elif shape == 'ceros':
        shape_obj = CerosShape(self.constants)
    else:
        raise ValueError(f'Unsupported shape: {shape}')

    number_of_sperm = self.constants['number_of_sperm']
    number_of_steps = self.constants['number_of_steps']
    step_len = self.constants['step_length']
    egg_x, egg_y, egg_z = _egg_position(self.constants)
    egg_center = np.array([egg_x, egg_y, egg_z])
    gamete_r = self.constants['gamete_r']

    seed_val = self.constants.get('seed_number')
    try:
        if seed_val is not None and str(seed_val).lower() != 'none':
            seed_int = int(seed_val)
            rng = np.random.default_rng(seed_int)
        else:
            rng = np.random.default_rng()
    except Exception:
        rng = np.random.default_rng()

    self.trajectory = []
    self.vectors = []

    for rep in range(int(sim_repeat)):
        for j in range(number_of_sperm):
            pos = shape_obj.initial_position()
            traj = [pos.copy()]
            vecs = []

            vec = rng.normal(size=3)
            vec /= np.linalg.norm(vec) + 1e-12
            vecs.append(vec.copy())

            stick_status = 0
            prev_status = "inside"

            for i in range(1, number_of_steps):
                vec = _perturb_direction(vec, self.constants['deviation'], rng)
                candidate = pos + vec * step_len

                if shape == "drop":
                    status, stick_status = IO_check_drop(candidate, stick_status, self.constants)
                    if status == IOStatus.ON_POLYGON:
                        candidate, vec, stick_status, status = drop_polygon_move(pos, vec, stick_status, constants)

                elif shape == "spot":
                    status, stick_status = IO_check_spot(pos, candidate, self.constants, prev_status, stick_status)
                    prev_status = status

                elif shape == "cube":
                    status, stick_status = IO_check_cube(candidate, self.constants)

                elif shape == "ceros":
                    status, stick_status = IOStatus.INSIDE

                else:
                    raise ValueError(f"Unknown shape: {shape}")

                traj.append(candidate.copy())
                vecs.append(vec.copy())
                pos = candidate

            self.trajectory.append(np.vstack(traj))
            self.vectors.append(np.vstack(vecs))

    self.trajectories = np.array(self.trajectory)
    self.vectors = np.array(self.vectors)
    print(f"[DEBUG] run完了: sperm={len(self.trajectory)}, steps={number_of_steps}, step_len={step_len:.4f} mm")
import matplotlib.pyplot as plt

def init():
        for ln in lines:
            ln.set_data([], [])
            ln.set_3d_properties([])
        return lines

    
    print(f'[INFO] 動画を保存しました: {save_path}')
    return save_path
    return (np.array(traj), intersection_records)
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
class SpermSimulation:

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

    
    def run(self, constants, result_dir, save_name, save_flag):
        """
        全てのshape（cube, drop, spot, ceros）に対応した精子運動シミュレーション実行関数。
        self.trajectory および self.vectors を更新する。
        """
        self.constants = constants
        shape = constants["shape"]
        step_len = constants["step_length"]
        vsl = constants["vsl"]
        hz = constants["sample_rate_hz"]
        deviation = constants["deviation"]
        seed = int(constants.get("seed_number", 0))
        self.number_of_sperm = int(constants["sperm_conc"] * constants["vol"] * 1e-3)
        self.number_of_steps = int(constants["sim_min"] * hz * 60)
        rng = np.random.default_rng(seed)
        # === 初期位置と形状オブジェクト ===
        if shape == "drop":
            shape_obj = DropShape(constants)
        elif shape == "cube":
            shape_obj = CubeShape(constants)
        elif shape == "spot":
            shape_obj = SpotShape(constants)
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

                # === IO 判定 ===
                if shape == "cube":
                    status, _ = IO_check_cube(candidate, constants)
                elif shape == "drop":
                    status, stick_status = IO_check_drop(candidate, stick_status, constants)
                elif shape == "spot":
                    status = IO_check_spot(pos, candidate, constants, prev_stat, stick_status)
                    prev_stat = status
                elif shape == "ceros":
                    status = IOStatus.INSIDE
                else:
                    raise ValueError(f"Unknown shape: {shape}")

                # === ステータスごとの挙動 ===
                if status in [IOStatus.INSIDE, IOStatus.INSIDE]:
                    pos = candidate
                elif status == IOStatus.ON_POLYGON:
                    candidate, vec, stick_status, status = drop_polygon_move(pos, vec, stick_status, constants)


                elif status in [IOStatus.REFLECT, SpotIO.REFLECT]:
                    vec *= -1

                elif status in [IOStatus.STICK, SpotIO.STICK, SpotIO.ON_POLYGON]:
                    stick_status = int(constants["surface_time"] * hz)

                elif status in [IOStatus.BORDER, SpotIO.BORDER, SpotIO.BOTTOM_OUT]:
                    pass  # 境界付近で停止
                elif status in [IOStatus.BORDER, IOStatus.BOTTOM_OUT, IOStatus.SPOT_EDGE_OUT]:
                    pass  # 停止や跳ね返り処理なし（その場維持）

                elif status in [IOStatus.INSIDE, IOStatus.ON_POLYGON]:
                    pass  # 正常なため何もしなくて良い
                    print(f"[WARNING] Unexpected status: {status}")

                self.trajectory[j, i] = pos
                self.vectors[j, i] = vec

        print("[DEBUG] 初期位置数:", len(self.initial_position))
        print("[DEBUG] 精子数:", self.number_of_sperm)
    
    def is_vector_meeting_egg(self, base_position, temp_position, egg_center, gamete_r):
        vector = temp_position - base_position
        distance_base = LA.norm(base_position - egg_center)
        distance_tip = LA.norm(temp_position - egg_center)
        if distance_base <= gamete_r or distance_tip <= gamete_r:
            return True
        f = base_position - egg_center
        a = vector @ vector
        b = 2 * (f @ vector)
        c = f @ f - gamete_r ** 2
        discriminant = b ** 2 - 4 * a * c
        if discriminant < 0:
            return False
        sqrt_discriminant = np.sqrt(discriminant)
        t1 = (-b - sqrt_discriminant) / (2 * a)
        t2 = (-b + sqrt_discriminant) / (2 * a)
        if 0 <= t1 <= 1 or 0 <= t2 <= 1:
            return True
        return False

    

    def init():
            for ln in lines:
                ln.set_data([], [])
                ln.set_3d_properties([])
            return lines

        
        print(f'[INFO] 動画を {save_path} に保存しました。')  # ← 明示的な日本語メッセージ
        return save_path


    