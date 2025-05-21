import numpy as np
import pandas as pd
import os
from pathlib import Path
import math
import numpy.linalg as LA
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import time
import sys
import sqlite3
from datetime import datetime
import ast
import random
import tkinter as tk
import configparser  # ← 追加：ユーザー選択を保存するため
from scipy.optimize import fsolve
SCRIPT_DIR = Path(__file__).resolve().parent
# Data folder path
DATA_FOLDER = SCRIPT_DIR / "Data folder"
# カレントディレクトリの移動（必要に応じて変更し
# スクリプトが置かれているディレクトリを基準にする
base_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
np.set_printoptions(threshold=np.inf)

def spot_volume_eq(R, angle_rad, volume):
    """
    Spot形状の体積計算（球冠の公式に基づく）
    """
    h = R * (1 - np.cos(angle_rad))
    return (np.pi * h**2 * (3*R - h)) / 3 - volume

def compute_spot_parameters(constants):
    """
    spotの各種パラメータを計算し、constants辞書に格納する関数。
    """
    angle_rad = np.deg2rad(constants['spot_angle'])
    spot_R_initial_guess = [(constants['volume'] * 3 / (4 * np.pi)) ** (1/3)]
    spot_R_solution = fsolve(spot_volume_eq, spot_R_initial_guess, args=(angle_rad, constants['volume']))
    spot_R_solution = spot_R_solution[0]    
    spot_bottom_height = spot_R_solution * np.cos(angle_rad)
    spot_bottom_R = spot_R_solution * np.sin(angle_rad)

    constants.update({
        'spot_R': spot_R_solution,
        'radius': spot_R_solution,
        'spot_bottom_height': spot_bottom_height,
        'spot_bottom_R': spot_bottom_R,
        'z_min': spot_bottom_height,
        'z_max': spot_R_solution
    })

def get_program_version():
    """
    スクリプトファイル名をバージョン情報として返す。
    Jupyterや対話モードの場合は 'interactive' として返す。
    """
    try:
        file_name = os.path.basename(__file__)
    except NameError:
        file_name = "interactive"
    version = f"{file_name}"
    return version

# def get_constants_from_gui(selected_data, shape, volume, sperm_conc):
#     constants = {}
#     constants['shape'] = shape.lower()
#     constants['volume'] = float(volume)
#     constants['sperm_conc'] = int(sperm_conc)
#     constants['spot_angle'] = float(selected_data.get('spot_angle', 60))
#     constants['VSL'] = float(selected_data.get('vsl', 0.13))
#     constants['deviation'] = float(selected_data.get('deviation', 0.04))
#     constants['sampl_rate_Hz'] = float(selected_data.get('sampl_rate_hz', 2))
#     constants['sim_min'] = float(selected_data.get('sim_min', 10))
#     constants['gamete_R'] = float(selected_data.get('gamete_r', 0.15))
#     constants['stick_sec'] = int(selected_data.get('stick_sec', 2))
#     constants['stick_steps'] = constants['stick_sec'] * constants['sampl_rate_Hz']
#     constants['egg_localization'] = selected_data.get('egg_localization', 'bottom_center')
#     constants['initial_direction'] = selected_data.get('initial_direction', 'random')
#     constants['initial_stick'] = int(selected_data.get('initial_stick', 0))
#     constants['seed_number'] = selected_data.get('seed_number', None)
#     constants['N_repeat'] = int(selected_data.get('n_repeat', 1))

#     constants['draw_trajectory'] = 'yes' if 'graph' in selected_data.get('outputs', []) else 'no'
#     constants['make_movie'] = 'yes' if 'movie' in selected_data.get('outputs', []) else 'no'

#     if constants['seed_number'] and constants['seed_number'].lower() != 'none':
#         np.random.seed(int(constants['seed_number']))

#     # 追加：ここでstep_lengthを定義（重要！）
#     constants['step_length'] = constants['VSL'] / constants['sampl_rate_Hz']
#     constants["limit"] = 1e-10
#     constants['reflection_analysis'] = 'yes' if selected_data.get('analysis_type', 'simulation') == 'reflection' else 'no'
#     return constants
def get_constants_from_gui(selected_data, shape, volume, sperm_conc):
    """
    GUI から受け取った選択データと shape, volume, sperm_conc を元に、
    シミュレーション定数をまとめた辞書を作成して返す。
    """
    constants = {}
    # 基本パラメータ
    constants['shape']         = shape.lower()
    constants['volume']        = float(volume)
    constants['sperm_conc']    = int(sperm_conc)

    # GUI からの入力値（デフォルト値含む）
    constants['spot_angle']        = float(selected_data.get('spot_angle', 60))
    constants['VSL']               = float(selected_data.get('vsl', 0.13))
    constants['deviation']         = float(selected_data.get('deviation', 0.04))
    constants['sampl_rate_Hz']     = float(selected_data.get('sampl_rate_hz', 2))
    constants['sim_min']           = float(selected_data.get('sim_min', 10))
    constants['gamete_R']          = float(selected_data.get('gamete_r', 0.15))
    constants['stick_sec']         = int(selected_data.get('stick_sec', 2))

    # STEP長さと関連パラメータ
    constants['stick_steps']    = constants['stick_sec'] * constants['sampl_rate_Hz']
    constants['step_length']    = constants['VSL'] / constants['sampl_rate_Hz']
    constants['limit']          = 1e-10

    # ここで .strip() を必ずかけて前後の不可視文字を除去
    raw_egg_loc = selected_data.get('egg_localization', 'bottom_center')
    constants['egg_localization'] = raw_egg_loc.strip()

    # その他のオプション設定
    constants['initial_direction']    = selected_data.get('initial_direction', 'random').strip()
    constants['initial_stick'] = int(selected_data.get('initial_stick', 0))
    constants['seed_number']          = selected_data.get('seed_number', None)
    constants['N_repeat']             = int(selected_data.get('n_repeat', 1))

    # 出力オプション
    outputs = selected_data.get('outputs', [])
    constants['draw_trajectory'] = 'yes' if 'graph' in outputs else 'no'
    constants['make_movie']      = 'yes' if 'movie' in outputs else 'no'

    # 乱数シード
    seed = constants['seed_number']
    if seed and str(seed).lower() != 'none':
        np.random.seed(int(seed))

    # Reflection 分析かどうか
    constants['reflection_analysis'] = (
        'yes' if selected_data.get('analysis_type', 'simulation') == 'reflection' else 'no'
    )

    return constants

def placement_of_eggs(constants):
    """
    shape, egg_localization に応じた卵子の配置座標などを返す。
    """
    shape = constants['shape'].lower()
    egg_localization = constants['egg_localization']
    gamete_R = constants['gamete_R']

    # shape ごとのパラメータを安全に取得
    z_min = constants.get('z_min', 0)
    x_min = constants.get('x_min', 0)
    y_min = constants.get('y_min', 0)

    if shape == "cube":
        positions_map = {
            "center":           (0, 0, 0),
            "bottom_center":    (0, 0, z_min + gamete_R),
            "bottom_side":      (x_min / 2 + gamete_R, y_min / 2 + gamete_R, z_min + gamete_R),
            "bottom_corner":    (x_min + gamete_R, y_min + gamete_R, z_min + gamete_R),
        }

    elif shape == "drop":
        drop_R = constants.get('drop_R', (constants['volume'] * 3 / (4 * np.pi)) ** (1 / 3))
        positions_map = {
            "center":           (0, 0, 0),
            "bottom_center":    (0, 0, -drop_R + gamete_R),
        }

    elif shape == "spot":
        # もしspotのパラメータが未計算の場合は計算して設定
        if ('spot_R' not in constants or constants['spot_R'] is None):
            compute_spot_parameters(constants)
        
        positions_map = {
            "center": (0, 0, (constants['spot_bottom_height'] + constants['spot_R']) / 2),
            "bottom_center": (0, 0, constants['spot_bottom_height'] + gamete_R),
            "bottom_edge": (constants['spot_bottom_R'] - gamete_R, 0, constants['spot_bottom_height'] + gamete_R),
        }

    elif shape == "ceros":
        positions_map = {
            "center":          (5, 5, 5),
            "bottom_center":   (5, 5, 5),
        }

    else:
        sys.exit(f"未知の形状 '{shape}' が指定されました。")

    if egg_localization not in positions_map:
        sys.exit(f"指定された egg_localization '{egg_localization}' は、形状 '{shape}' に対して無効です。")

    egg_x, egg_y, egg_z = positions_map[egg_localization]
    e_x_min = egg_x - gamete_R
    e_y_min = egg_y - gamete_R
    e_z_min = egg_z - gamete_R
    e_x_max = egg_x + gamete_R
    e_y_max = egg_y + gamete_R
    e_z_max = egg_z + gamete_R

    egg_center = np.array([egg_x, egg_y, egg_z])
    egg_position_4d = np.array([egg_x, egg_y, egg_z, 0])

    return (
        egg_x, egg_y, egg_z,
        e_x_min, e_y_min, e_z_min,
        e_x_max, e_y_max, e_z_max,
        egg_center, egg_position_4d
    )

def get_limits(constants):
    shape = constants['shape'].lower()

    if shape == "cube":
        return (
            constants['x_min'], constants['x_max'],
            constants['y_min'], constants['y_max'],
            constants['z_min'], constants['z_max']
        )

    elif shape == "drop":
        drop_R = constants.get('drop_R')
        if drop_R is None:
            drop_R = (constants['volume'] * 3 / (4 * np.pi)) ** (1 / 3)
        return (
            -drop_R, drop_R,
            -drop_R, drop_R,
            -drop_R, drop_R
        )

    elif shape == "spot":
        if ('spot_R' not in constants or constants['spot_R'] is None):
            compute_spot_parameters(constants)

        spot_R = constants['spot_R']
        spot_bottom_height = constants['spot_bottom_height']
        spot_bottom_R = constants['spot_bottom_R']

        return (
            -spot_bottom_R, spot_bottom_R,
            -spot_bottom_R, spot_bottom_R,
            spot_bottom_height, spot_R
        )

    elif shape == "ceros":
        return (
            constants['ceros_x_min'], constants['ceros_x_max'],
            constants['ceros_y_min'], constants['ceros_y_max'],
            constants['ceros_z_min'], constants['ceros_z_max']
        )

    else:
        raise ValueError(f"Unknown shape: {shape}")

def get_reflection_initial_positions(shape, volume, initial_direction, constants):
    """
    Reflectionモードの際、初期位置と初期ベクトルをshapeとvolumeに応じて動的に決定する関数。
    initial_direction: 'right', 'left', 'up', 'down', 'random' など
    """
    x_min, x_max, y_min, y_max, z_min, z_max = get_limits(constants)

    center = np.array([
        (x_min + x_max) / 2,
        (y_min + y_max) / 2,
        (z_min + z_max) / 2
    ])

    if shape == 'cube':
        base_position = center
    elif shape == 'drop':
        base_position = np.array([0, 0, 0])
    elif shape == 'spot':
        base_position = np.array([0, 0, constants['spot_bottom_height']])
    else:
        base_position = center

    directions = {
        'right': np.array([1, 0, 0]),
        'left': np.array([-1, 0, 0]),
        'up': np.array([0, 1, 0]),
        'down': np.array([0, -1, 0]),
        'forward': np.array([0, 0, 1]),
        'backward': np.array([0, 0, -1]),
        'random': np.random.normal(size=3)
    }

    direction_vec = directions.get(initial_direction.lower(), directions['random'])
    direction_vec = normalize_vector(direction_vec)

    first_temp = base_position + direction_vec * constants['step_length']

    return base_position, first_temp

def normalize_vector(v):
    norm = LA.norm(v)
    if norm == 0:
        raise ValueError("Zero vector cannot be normalized.")
    return v / norm

# ------------------------------------------------------------
# (1) cut_and_bend系の関数
# ------------------------------------------------------------
def cut_and_bend_cube(self, IO_status, base_position, temp_position, remaining_distance, constants):
    out_flags = []
    x_min, x_max, y_min, y_max, z_min, z_max = get_limits(constants)
    cutting_ratios = {}
    
    axes = ['x', 'y', 'z']
    min_vals = [x_min, y_min, z_min]
    max_vals = [x_max, y_max, z_max]

    for axis, pos, min_val, max_val in zip(axes, temp_position, min_vals, max_vals):
        idx = axes.index(axis)
        if pos < min_val - constants['limit']:
            out_flags.append(f'{axis}_min_out')
            ratio = (min_val - base_position[idx]) / (temp_position[idx] - base_position[idx])
            cutting_ratios[axis] = ratio
        elif pos > max_val + constants['limit']:
            out_flags.append(f'{axis}_max_out')
            ratio = (max_val - base_position[idx]) / (temp_position[idx] - base_position[idx])
            cutting_ratios[axis] = ratio

    if cutting_ratios:
        first_out_axis = min(cutting_ratios, key=cutting_ratios.get)
        first_out_axis_index = axes.index(first_out_axis)
        cutting_ratio = cutting_ratios[first_out_axis]

        vector_to_be_cut = temp_position - base_position
        vector_to_surface = vector_to_be_cut * cutting_ratio
        intersection_point = base_position + vector_to_surface
        last_vec = temp_position - intersection_point

        if LA.norm(last_vec) < constants['limit']:
            temp_position = base_position + 1.1 * (temp_position - base_position)
            vector_to_be_cut = temp_position - base_position
            vector_to_surface = vector_to_be_cut * cutting_ratio
            intersection_point = base_position + vector_to_surface
            last_vec = temp_position - intersection_point
            if LA.norm(last_vec) < constants['limit']:
                sys.exit("last_vec too small even after scaling")

        last_vec[first_out_axis_index] = 0
        nv = LA.norm(last_vec)
        dist_to_surface = LA.norm(vector_to_surface)
        remaining_distance -= dist_to_surface
        if remaining_distance < 0:
            remaining_distance = 0
            sys.exit("this is ありえない！")
        if nv == 0:
            sys.exit("this is ありえない２２２！")
        else:
            last_vec_adjusted = last_vec / nv * remaining_distance

        temp_position = intersection_point + last_vec_adjusted
    else:
        intersection_point = base_position  # cutting_ratios が空ならとりあえずこのまま

    return temp_position, intersection_point, remaining_distance

def cut_and_bend_vertex(vertex_point, base_position, remaining_distance, constants):
    dist_to_vertex = np.linalg.norm(vertex_point - base_position)
    move_on_new_edge = remaining_distance - dist_to_vertex
    if move_on_new_edge < 0:
        move_on_new_edge = 0

    cube_center = np.array([0, 0, 0])
    candidate_edges = []
    for i in range(3):
        direction = -1 if vertex_point[i] > cube_center[i] else 1
        edge_vec = np.zeros(3)
        edge_vec[i] = direction
        candidate_edges.append(edge_vec)

    incoming_vec = vertex_point - base_position
    dist_incoming = np.linalg.norm(incoming_vec)
    if dist_incoming == 0:
        incoming_dir = np.zeros(3)
    else:
        incoming_dir = incoming_vec / dist_incoming

    filtered_edges = [
        edge for edge in candidate_edges
        if not (np.allclose(edge, incoming_dir) or np.allclose(edge, -incoming_dir))
    ]

    if filtered_edges:
        new_edge = random.choice(filtered_edges)

    new_temp_position = vertex_point + new_edge * move_on_new_edge
    intersection_point = vertex_point
    new_remaining_distance = constants['VSL'] / constants['sampl_rate_Hz']
    return intersection_point, new_temp_position, new_remaining_distance

def cut_and_bend_bottom(self, IO_status, base_position, temp_position, remaining_distance, constants):
    """
    Spot底面との衝突を切り貼りする処理。
    """
    bottom_z = constants['spot_bottom_height']
    
    ratio = (bottom_z - base_position[2]) / (temp_position[2] - base_position[2])
    vector_to_be_cut = temp_position - base_position
    vector_to_surface = vector_to_be_cut * ratio
    intersection_point = base_position + vector_to_surface
    intersection_point[2] = bottom_z
    vector_to_surface = intersection_point - base_position
    last_vec = temp_position - intersection_point
    if LA.norm(last_vec) < constants['limit']:
        sys.exit("last_vec too small")

    last_vec[2] = 0
    nv = LA.norm(last_vec)
    remaining_distance -= LA.norm(vector_to_surface)

    if nv < constants['limit']:
        sys.exit("vector finished on the surface: redo")
    else:
        last_vec_adjusted = last_vec / nv * remaining_distance

    threshold = constants['VSL'] / constants['sampl_rate_Hz'] * 1e-7
    if LA.norm(last_vec_adjusted) < threshold:
        raise ValueError("last_vec_adjusted is too small; simulation aborted.")

    temp_position = intersection_point + last_vec_adjusted
    return temp_position, intersection_point, remaining_distance

def cut_and_bend_spot_edge_out(self, IO_status, base_position, temp_position,
                                 remaining_distance, constants):
    """
    Spot底面の円周(底面から見たときの外縁)に衝突したときの処理。
    後者の正しく動くコードと同一のものに修正。
    """
    spot_bottom_R = constants['spot_bottom_R']
    inner_angle = constants['inner_angle']

    x0, y0 = base_position[0], base_position[1]
    x1, y1 = temp_position[0], temp_position[1]
    z = constants['spot_bottom_height']
    dx = x1 - x0
    dy = y1 - y0

    A = dx**2 + dy**2
    B = 2 * (x0 * dx + y0 * dy)
    C = x0**2 + y0**2 - spot_bottom_R**2
    discriminant = B**2 - 4*A*C
    sqrt_discriminant = np.sqrt(discriminant)

    t1 = (-B + sqrt_discriminant) / (2*A)
    t2 = (-B - sqrt_discriminant) / (2*A)
    t_candidates = [t for t in [t1, t2] if 0 <= t <= 1]
    t_intersect = min(t_candidates) if t_candidates else 0

    xi = x0 + t_intersect * dx
    yi = y0 + t_intersect * dy
    intersection_point = np.array([xi, yi, z])

    distance_to_intersection = LA.norm(intersection_point - base_position)
    remaining_distance -= distance_to_intersection

    bi = intersection_point - base_position
    bi_norm = LA.norm(bi)
    if bi_norm < 1e-12:
        bi_norm = 1e-8
    bi_normalized = bi / bi_norm

    oi = np.array([xi, yi, 0])
    oi_norm = LA.norm(oi)
    if oi_norm < 1e-12:
        oi_norm = 1e-8
    oi_normalized = oi / oi_norm

    tangent_1 = np.array([-oi_normalized[1], oi_normalized[0], 0])
    tangent_2 = -tangent_1

    angle_with_tangent_1 = np.arccos(
        np.clip(tangent_1[:2] @ bi_normalized[:2], -1.0, 1.0)
    )
    angle_with_tangent_2 = np.arccos(
        np.clip(tangent_2[:2] @ bi_normalized[:2], -1.0, 1.0)
    )
    if angle_with_tangent_1 < angle_with_tangent_2:
        selected_tangent = tangent_1
    else:
        selected_tangent = tangent_2

    cross = selected_tangent[0]*bi_normalized[1] - selected_tangent[1]*bi_normalized[0]
    if cross > 0:
        angle_adjust = -inner_angle
    else:
        angle_adjust = inner_angle

    def rotate_vector_2d(vec, angle):
        c = np.cos(angle)
        s = np.sin(angle)
        x_new = vec[0]*c - vec[1]*s
        y_new = vec[0]*s + vec[1]*c
        return np.array([x_new, y_new, 0])

    last_vec = rotate_vector_2d(selected_tangent, angle_adjust)
    last_vec /= (LA.norm(last_vec) + 1e-12)

    last_vec = last_vec * remaining_distance
    new_temp_position = intersection_point + last_vec
    new_temp_position[2] = z

    is_bottom_edge = True
    return new_temp_position, intersection_point, remaining_distance, is_bottom_edge

def line_sphere_intersection(base_position, temp_position, radius, remaining_distance, constants):
    d = temp_position - base_position
    d_norm = LA.norm(d)
    if d_norm < constants['limit']:
        sys.exit("too short")

    d_unit = d / d_norm
    f = base_position
    a = 1.0
    b = 2.0 * (f @ d_unit)
    c = (f @ f) - radius ** 2

    discriminant = b**2 - 4*a*c
    if discriminant < constants['limit']:
        return base_position, remaining_distance

    sqrt_discriminant = np.sqrt(discriminant)
    q = -0.5 * (b + np.copysign(sqrt_discriminant, b))
    t1 = q / a
    t2 = c / q if abs(q) > constants['limit'] else np.inf

    t_candidates = [t for t in [t1, t2] if t > constants['limit']]
    if not t_candidates:
        sys.exit("No positive t found for intersection.")

    t = min(t_candidates)
    intersection_point = base_position + t * d_unit
    distance_traveled = t * d_norm
    updated_remaining_distance = remaining_distance - distance_traveled

    return intersection_point, updated_remaining_distance

def compute_normalized_vectors(base_position, intersection_point, constants):
    oi = intersection_point
    oi_normalized = normalize_vector(oi)
    bi = intersection_point - base_position
    bi_norm = LA.norm(bi)
    if bi_norm < constants['limit']:
        print("oi", oi)
        print("LA.norm(bi)", LA.norm(bi))
        sys.exit("Vector bi_norm is zero!")
    bi_normalized = bi / bi_norm
    return oi_normalized, bi_normalized

def determine_rotation_direction(selected_tangent, normal_B, bi_normalized, modify_angle):
    cross_product = (np.cross(selected_tangent, normal_B) @ bi_normalized)
    if cross_product < 0:
        modify_angle = -modify_angle
    return modify_angle

def compute_tangent_vectors(oi_normalized, bi_normalized):
    normal_B = normalize_vector(np.cross(bi_normalized, oi_normalized))
    tangent_1 = normalize_vector(np.cross(normal_B, oi_normalized))
    tangent_2 = -tangent_1
    angle1 = calculate_angle_between_vectors(tangent_1, bi_normalized)
    angle2 = calculate_angle_between_vectors(tangent_2, bi_normalized)
    if angle1 < angle2:
        selected_tangent = tangent_1
    else:
        selected_tangent = tangent_2
    return selected_tangent, normal_B

def calculate_angle_between_vectors(v1, v2):
    v1_u = normalize_vector(v1)
    v2_u = normalize_vector(v2)
    dot_product = np.clip(v1_u @ v2_u, -1.0, 1.0)
    return np.arccos(dot_product)

def rotate_vector(vector, axis, angle):
    axis = normalize_vector(axis)
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    return (vector * cos_theta +
            np.cross(axis, vector) * sin_theta +
            (axis @ vector) * (1 - cos_theta))

def cut_and_bend_sphere(base_position, remaining_distance, temp_position, constants):
    radius = constants['radius']
    modify_angle = constants['inner_angle']

    intersection_point, remaining_distance = line_sphere_intersection(
        base_position, temp_position, radius, remaining_distance, constants
    )

    oi_normalized, bi_normalized = compute_normalized_vectors(
        base_position, intersection_point, constants
    )
    selected_tangent, normal_B = compute_tangent_vectors(oi_normalized, bi_normalized)
    modify_angle = determine_rotation_direction(
        selected_tangent, normal_B, bi_normalized, modify_angle
    )
    last_vec = rotate_vector(selected_tangent, normal_B, modify_angle)
    last_vec_normalized = normalize_vector(last_vec)

    last_vec = last_vec_normalized * remaining_distance
    new_temp_position = intersection_point + last_vec

    lv_dot = np.dot(last_vec, last_vec)
    if abs(lv_dot) < constants['limit']:
        inward_dir = np.array([0.0, 0.0, 0.0])
    else:
        t = - np.dot(intersection_point, last_vec) / lv_dot
        F = intersection_point + t * last_vec
        inward_dir = -F
        norm_id = LA.norm(inward_dir)
        if norm_id < constants['limit']:
            inward_dir = np.array([0.0,0.0,0.0])
        else:
            inward_dir /= norm_id

    return new_temp_position, intersection_point, remaining_distance, inward_dir

# ------------------------------------------------------------
# face_and_inward_dir のサブ関数
# ------------------------------------------------------------
def _calculate_inward_dir_from_axes_hit(x, y, z, x_min, x_max, y_min, y_max, z_min, z_max, constants):
    on_x_min = abs(x - x_min) <= constants['limit']
    on_x_max = abs(x - x_max) <= constants['limit']
    on_y_min = abs(y - y_min) <= constants['limit']
    on_y_max = abs(y - y_max) <= constants['limit']
    on_z_min = abs(z - z_min) <= constants['limit']
    on_z_max = abs(z - z_max) <= constants['limit']
    hit_count = sum([on_x_min, on_x_max, on_y_min, on_y_max, on_z_min, on_z_max])

    pinned_coords = []
    free_axes = []

    # X
    if on_x_min:
        pinned_coords.append(('x', x_min))
    elif on_x_max:
        pinned_coords.append(('x', x_max))
    else:
        free_axes.append('x')
    # Y
    if on_y_min:
        pinned_coords.append(('y', y_min))
    elif on_y_max:
        pinned_coords.append(('y', y_max))
    else:
        free_axes.append('y')
    # Z
    if on_z_min:
        pinned_coords.append(('z', z_min))
    elif on_z_max:
        pinned_coords.append(('z', z_max))
    else:
        free_axes.append('z')

    return pinned_coords, free_axes, hit_count

def face_and_inward_dir(temp_position, base_position, last_vec, IO_status, stick_status, constants):
    if IO_status == "inside" or IO_status =="temp_on_polygon":
        denom = np.dot(last_vec, last_vec)
        if abs(denom) < constants['limit']:
            sys.exit("last_vec is zero or near-zero in face_and_inward_dir")
        t = - np.dot(base_position, last_vec) / denom
        F = base_position + t * last_vec
        if np.linalg.norm(F) <= constants['limit']:
            sys.exit("原点を通るのでredo")
        inward_dir = -F / np.linalg.norm(F)
        return inward_dir

    elif IO_status in ["temp_on_edge", "temp_on_surface"]:
        x_min, x_max, y_min, y_max, z_min, z_max = get_limits(constants)
        
        x, y, z = temp_position

        pinned_coords, free_axes, hit_count = _calculate_inward_dir_from_axes_hit(
            x, y, z, x_min, x_max, y_min, y_max, z_min, z_max, constants
        )

        if hit_count == 1:
            (ax_name, ax_val) = pinned_coords[0]
            if ax_name == 'x':
                if abs(ax_val - x_max) <= constants['limit']:
                    inward_dir = np.array([-1, 0, 0])
                else:
                    inward_dir = np.array([1, 0, 0])
            elif ax_name == 'y':
                if abs(ax_val - y_max) <= constants['limit']:
                    inward_dir = np.array([0, -1, 0])
                else:
                    inward_dir = np.array([0, 1, 0])
            elif ax_name == 'z':
                if abs(ax_val - z_max) <= constants['limit']:
                    inward_dir = np.array([0, 0, -1])
                else:
                    inward_dir = np.array([0, 0, 1])
            else:
                sys.exit("no way - face")
            return inward_dir

        elif hit_count == 2:
            mid_x = mid_y = mid_z = 0.0
            for (ax_name, ax_val) in pinned_coords:
                if ax_name == 'x':
                    mid_x = ax_val
                elif ax_name == 'y':
                    mid_y = ax_val
                elif ax_name == 'z':
                    mid_z = ax_val

            if len(free_axes) == 1:
                fa = free_axes[0]
                if fa == 'x':
                    mid_x = (x_min + x_max) / 2
                elif fa == 'y':
                    mid_y = (y_min + y_max) / 2
                elif fa == 'z':
                    mid_z = (z_min + z_max) / 2

            midpoint_of_edge = np.array([mid_x, mid_y, mid_z], dtype=float)
            direction_vec = -midpoint_of_edge
            norm_dv = np.linalg.norm(direction_vec)
            if norm_dv < constants['limit']:
                sys.exit("no way3")

            inward_dir = direction_vec / norm_dv
            return inward_dir
        else:
            return None

    elif IO_status == "vertex_out":
        return None
    else:
        return None

# ------------------------------------------------------------
# (3) サンプリング関数を統合 (sample_theta_phi系)
# ------------------------------------------------------------
def sample_random_angles(sigma, constants, cone_type="full"):

    max_theta = np.pi
    if cone_type == "quarter":
        min_theta = constants['limit']
        max_theta = np.pi - constants['limit']
        min_phi = -np.pi/4 + constants['limit']
        max_phi = np.pi/4 - constants['limit']
    elif cone_type == "half":
        min_theta = constants['limit']
        max_theta = np.pi - constants['limit']
        min_phi = -np.pi/2 + constants['limit']
        max_phi = np.pi/2 - constants['limit']
    else:
        min_theta = constants['limit']
        max_theta = np.pi - constants['limit']
        min_phi = -np.pi
        max_phi = np.pi

    while True:
        theta = abs(np.random.normal(0, sigma))
        if min_theta < theta < max_theta:
            break
    phi = np.random.uniform(min_phi, max_phi)
    return theta, phi

def make_local_xy(v, inward_dir=None):
    v_norm = LA.norm(v)
    if v_norm < 1e-12:
        v = np.array([0, 0, 1], dtype=float)
    else:
        v = v / v_norm

    if inward_dir is None:
        arbitrary = np.array([1, 0, 0], dtype=float)
        if abs(v[0]) > 0.9:
            arbitrary = np.array([0, 1, 0], dtype=float)
        local_x = np.cross(v, arbitrary)
        lx = LA.norm(local_x)
        if lx < 1e-12:
            local_x = np.array([1, 0, 0], dtype=float)
        else:
            local_x /= lx
    else:
        inward_norm = np.linalg.norm(inward_dir)
        if inward_norm < 1e-12:
            local_x = np.array([1, 0, 0], dtype=float)
        else:
            local_x = inward_dir / inward_norm

        if np.allclose(np.abs(np.dot(local_x, v)), 1.0, atol=1e-12):
            raise ValueError("inward_dir は v と平行でないベクトルを指定してください。")

    local_y = np.cross(v, local_x)
    local_y /= LA.norm(local_y)

    return local_x, local_y

def generate_cone_vector(v, local_x, local_y, inward_dir, constants, sigma, remaining_distance,
                         cone_type='full', do_projection=False):
    theta, phi = sample_random_angles(sigma, constants, cone_type="full")

    x_local = np.sin(theta)*np.cos(phi)
    y_local = np.sin(theta)*np.sin(phi)
    z_local = np.cos(theta)

    temp_dir = x_local*local_x + y_local*local_y + z_local*v

    if do_projection:
        n = inward_dir
        n_norm = LA.norm(n)
        if n_norm > 1e-12:
            n_unit = n / n_norm
            dot_val = np.dot(temp_dir, n_unit)
            temp_dir = temp_dir - dot_val*n_unit

    nd = LA.norm(temp_dir)
    if nd < 1e-12:
        return np.zeros(3)

    temp_dir *= (remaining_distance / nd)
    return temp_dir

# ------------------------------------------------------------
# (B) prepare_new_vector_* の重複をまとめる
# ------------------------------------------------------------
def prepare_new_vector(last_vec, constants,
                       boundary_type="free",
                       stick_status=0,
                       inward_dir=None):
    v_norm = LA.norm(last_vec)
    if v_norm < constants['limit']:
        raise ValueError("prepare_new_vector: last_vec が短すぎます。")

    v = last_vec / v_norm
    # 境界種別に応じて cone_type, do_projection を決める
    if boundary_type == "edge":
        cone_type = "quarter"
        do_proj = False
        if stick_status > 0:
            return v * constants['step_length']
    elif boundary_type == "surface":
        cone_type = "half"
        do_proj = (stick_status > 0)
    elif boundary_type == "polygon":
        cone_type = "half"
        do_proj = (stick_status > 0)
    else:
        # "free"
        cone_type = "full"
        do_proj = False

    local_x, local_y = make_local_xy(v, inward_dir)
    new_vec = generate_cone_vector(
        v, local_x, local_y, inward_dir,constants,
        sigma=constants['deviation'],
        remaining_distance=constants['step_length'],
        cone_type=cone_type,
        do_projection=do_proj
    )
    return new_vec

# ------------------------------------------------------------
# IO判定系（Spotの部分を後者に合わせて修正済み）
# ------------------------------------------------------------

def IO_check_cube(temp_position, constants):
    x_min, x_max, y_min, y_max, z_min, z_max = get_limits(constants)
    

    def classify_dimension(pos, min_val, max_val, constants):
        if pos < min_val - constants['limit']:
            return "outside"
        elif pos > max_val + constants['limit']:
            return "outside"
        elif abs(pos - min_val) <= constants['limit'] or abs(pos - max_val) <= constants['limit']:
            return "surface"
        else:
            return "inside"

    x_class = classify_dimension(temp_position[0], x_min, x_max, constants)
    y_class = classify_dimension(temp_position[1], y_min, y_max, constants)
    z_class = classify_dimension(temp_position[2], z_min, z_max, constants)

    classifications = [x_class, y_class, z_class]
    inside_count = classifications.count("inside")
    surface_count = classifications.count("surface")
    outside_count = classifications.count("outside")

    if inside_count == 3:
        return "inside", None
    elif inside_count == 2 and surface_count == 1:
        return "temp_on_surface", None
    elif inside_count == 1 and surface_count == 2:
        return "temp_on_edge", None
    elif inside_count == 2 and outside_count == 1:
        return "surface_out", None
    elif inside_count == 1 and outside_count == 2:
        return "surface_out", None
    elif inside_count == 0 and surface_count == 2 and outside_count == 1:
        vx, vy, vz = None, None, None
        x, y, z = temp_position

        if x_class == "surface":
            if abs(x - x_min) <= constants['limit']:
                vx = x_min
            else:
                vx = x_max
        elif x_class == "outside":
            if x < x_min - constants['limit']:
                vx = x_min
            else:
                vx = x_max

        if y_class == "surface":
            if abs(y - y_min) <= constants['limit']:
                vy = y_min
            else:
                vy = y_max
        elif y_class == "outside":
            if y < y_min - constants['limit']:
                vy = y_min
            else:
                vy = y_max

        if z_class == "surface":
            if abs(z - z_min) <= constants['limit']:
                vz = z_min
            else:
                vz = z_max
        elif z_class == "outside":
            if z < z_min - constants['limit']:
                vz = z_min
            else:
                vz = z_max

        vertex_coords = np.array([vx, vy, vz], dtype=float)
        return "vertex_out", vertex_coords
    elif (
        (inside_count == 1 and surface_count == 1 and outside_count == 1) or
        (inside_count == 0 and surface_count == 1 and outside_count == 2)
    ):
        return "edge_out", None
    elif inside_count == 0 and surface_count == 0 and outside_count == 3:
        return "surface_out", None
    elif inside_count == 0 and surface_count == 3 and outside_count == 0:
        return "border", None
    else:
        raise ValueError("Unknown inside/surface/outside combination")

def IO_check_drop(temp_position, stick_status, constants):
    distance_from_center = LA.norm(temp_position)
    radius = constants['drop_R']
    
    if distance_from_center > radius + constants['limit']:
        IO_status = "sphere_out"
    elif distance_from_center < radius - constants['limit']:
        if stick_status > 0:
            IO_status = "temp_on_polygon"
        else:
            IO_status = "inside"
    else:
        IO_status = "border"
    return IO_status

def IO_check_spot(base_position, temp_position, constants, IO_status):
    """
    Spot形状におけるIO判定。
    後者の「うまくいくプログラム」と同一仕様になるように修正。
    """
    radius   = constants['radius']
    bottom_z = constants['spot_bottom_height']
    bottom_R = constants['spot_bottom_R']

    z_tip = temp_position[2]
    r_tip = LA.norm(temp_position)  # 底面原点(0,0,0)からの距離
    xy_dist = np.sqrt(temp_position[0]**2 + temp_position[1]**2)

    # 円錐上部の球面衝突 (zが底面より上 + 球面外かどうか)
    if z_tip > bottom_z + constants['limit']:
        if r_tip > radius + constants['limit']:
            # 球面より外
            return "sphere_out"
        else:
            return "inside"

    # 底面より更に下
    elif z_tip < bottom_z - constants['limit']:
        denom = (temp_position[2] - base_position[2])
        t = (bottom_z - base_position[2]) / denom
        if t < 0 or t > 1:
            return "sphere_out"

        intersect_xy = base_position[:2] + t*(temp_position[:2] - base_position[:2])
        dist_xy = np.sqrt(intersect_xy[0]**2 + intersect_xy[1]**2)

        if dist_xy < bottom_R + constants['limit']:
            return "bottom_out"
        else:
            return "sphere_out"

    # zが底面付近
    elif bottom_z - constants['limit'] < z_tip < bottom_z + constants['limit']:
        # 円周外
        if xy_dist > bottom_R + constants['limit']:
            return "spot_edge_out"
        # ぴったり円周付近
        elif abs(xy_dist - bottom_R) <= constants['limit']:
            return "border"
        # 円周より内側
        elif xy_dist < bottom_R - constants['limit']:
            # すでにedge_out/ polygon_mode だったかどうか確認
            if IO_status in ["spot_edge_out", "polygon_mode"]:
                return "polygon_mode"
            else:
                return "spot_bottom"

    return "inside"

# ------------------------------------------------------------
# シミュレーション本体
# ------------------------------------------------------------
class SpermSimulation:
    def __init__(self, constants, visualizer, simulation_data):
        self.constants = constants
        self.visualizer = visualizer
        self.simulation = simulation_data
        self.number_of_sperm = int(constants['number_of_sperm'])
        self.n_simulation = int(constants['n_simulation'])
        self.n_stop = constants.get('n_stop', 0)

        if constants.get('reflection_analysis', 'no') == "yes":
            self.initial_stick = constants['initial_stick']
        else:
            self.initial_stick = 0

        # 必須の属性を初期化
        self.vec_colors = np.empty((self.number_of_sperm, self.n_simulation), dtype=object)
        self.vec_thickness_2d = np.zeros((self.number_of_sperm, self.n_simulation), dtype=float)
        self.vec_thickness_3d = np.zeros((self.number_of_sperm, self.n_simulation), dtype=float)
        self.trajectory = np.zeros((self.number_of_sperm, self.n_simulation, 3))
        self.prev_IO_status = [None] * self.number_of_sperm
        self.intersection_records = []

        # 色と線の太さを初期化するメソッドを呼び出す
        self.initialize_colors()
        self.initialize_thickness()

        # 初期位置ベクトルを設定
        for j in range(self.number_of_sperm):
            base_position, temp_position = self.initial_vec(j, constants)
            self.trajectory[j, 0] = base_position
            if self.n_simulation > 1:
                self.trajectory[j, 1] = temp_position

        # デバッグ用に constants を表示
        print("初期化時のconstants:", constants)

    def merge_contact_events(self):
        """
        接触イベント（(精子番号, ステップ番号)）をまとめて連続接触を1つに圧縮。
        """
        from collections import defaultdict

        events_by_sperm = defaultdict(list)
        for sperm_index, step in sorted(self.intersection_records, key=lambda x: (x[0], x[1])):
            events_by_sperm[sperm_index].append(step)

        merged_events = []
        for sperm_index, steps in events_by_sperm.items():
            if not steps:
                continue
            start_step = steps[0]
            end_step = steps[0]
            for step in steps[1:]:
                if step == end_step + 1:
                    end_step = step
                else:
                    merged_events.append((sperm_index, start_step))
                    start_step = step
                    end_step = step
            merged_events.append((sperm_index, start_step))
        return merged_events

    def initialize_colors(self):
        base_colors = [
            "#000000","#1f77b4","#ff7f0e","#2ca02c","#9467bd","#8c564b",
            "#e377c2","#7f7f7f","#bcbd22","#17becf","#aec7e8","#ffbb78",
            "#98df8a","#c5b0d5","#c49c94","#f7b6d2","#c7c7c7","#dbdb8d",
            "#9edae5","#2f4f4f",
        ]
        for j in range(self.number_of_sperm):
            c = base_colors[j % len(base_colors)]
            for i in range(self.n_simulation):
                self.vec_colors[j, i] = c

    def initialize_thickness(self):
        for j in range(self.number_of_sperm):
            for i in range(self.n_simulation):
                self.vec_thickness_2d[j, i] = 0.4
                self.vec_thickness_3d[j, i] = 1.5

    def set_vector_color(self, j, i, color):
        self.vec_colors[j, i] = color

    def _random_base_position(self):
        shape = self.constants['shape']
        if shape in ["cube", "ceros"]:
            x_min, x_max, y_min, y_max, z_min, z_max = get_limits(self.constants)
            x_base = np.random.uniform(x_min, x_max)
            y_base = np.random.uniform(y_min, y_max)
            z_base = np.random.uniform(z_min, z_max)
            return np.array([x_base, y_base, z_base])
        elif shape == "drop":
            theta_base = np.arccos(2 * np.random.random() - 1)
            phi_base = np.random.uniform(-np.pi, np.pi)
            s_base = self.constants['drop_R'] * (np.random.random() ** (1/3))
            x_base = s_base * np.sin(theta_base) * np.cos(phi_base)
            y_base = s_base * np.sin(theta_base) * np.sin(phi_base)
            z_base = s_base * np.cos(theta_base)
            return np.array([x_base, y_base, z_base])
        elif shape == "spot":
            radius = self.constants['radius']
            spot_angle_rad = np.deg2rad(self.constants['spot_angle'])
            while True:
                theta_base = np.random.uniform(0, spot_angle_rad)
                phi_base = np.random.uniform(-np.pi, np.pi)
                r_base = radius * (np.random.random() ** (1/3))
                x_base = r_base * np.sin(theta_base) * np.cos(phi_base)
                y_base = r_base * np.sin(theta_base) * np.sin(phi_base)
                z_base = r_base * np.cos(theta_base)
                if z_base >= radius * np.cos(spot_angle_rad):
                    break
            return np.array([x_base, y_base, z_base])
        else:
            return np.array([0, 0, 0])

    def initial_vec(self, j, constants):
        shape = self.constants['shape']
        analysis_type = self.constants.get('analysis_type', 'single_simulation')
        reflection_mode = self.constants['reflection_analysis']

        if reflection_mode == "yes":
            # 反射解析モードの場合
            if shape == "spot":
                # spot専用の初期位置を設定（例）
                spot_bottom_R = self.constants.get('spot_bottom_R', 1.0)
                spot_bottom_height = self.constants.get('spot_bottom_height', 0.5)

                base_position = np.array([
                    spot_bottom_R - constants['step_length'] * 1.5,
                    0.001,
                    spot_bottom_height
                ])
                direction_vec = (constants['step_length'], 0, 0)
                temp_position = base_position + direction_vec

            else:
                # 他のshapeでは旧来のコードを流用
                base_position = np.array(ast.literal_eval(self.constants['start_position']))
                temp_position = np.array(ast.literal_eval(self.constants['first_temp']))

            if shape == "cube" or shape == "ceros":
                IO_status, vertex_point = IO_check_cube(temp_position, self.constants)
            elif shape == "drop":
                IO_status = IO_check_drop(temp_position, self.constants['initial_stick'], self.constants)
                vertex_point = None
            elif shape == "spot":
                IO_status = IO_check_spot(base_position, temp_position, self.constants, "none")
                vertex_point = None

            local_stick = self.constants['initial_stick']
            if IO_status in ["temp_on_surface", "temp_on_edge"]:
                last_vec = temp_position - base_position
                if np.linalg.norm(last_vec) < self.constants['limit']:
                    last_vec = np.array([constants['step_length'], 0.0, 0.0])
                inward_dir = face_and_inward_dir(
                    temp_position,
                    base_position,
                    last_vec,
                    IO_status,
                    local_stick,
                    constants=self.constants
                )
                if inward_dir is None:
                    inward_dir = np.array([0.0, 0.0, 1.0])

                new_vec = prepare_new_vector(
                    last_vec, self.constants,
                    boundary_type=("edge" if IO_status == "temp_on_edge" else "surface"),
                    stick_status=local_stick,
                    inward_dir=inward_dir
                )
                temp_position = base_position + new_vec

        else:
            base_position = self._random_base_position()
            initial_vector = self.get_random_direction_3D() * constants['step_length']
            temp_position = base_position + initial_vector

        return base_position, temp_position

    def get_random_direction_3D(self):
        phi = np.random.uniform(0, 2*np.pi)
        costheta = np.random.uniform(-1, 1)
        theta = np.arccos(costheta)
        return np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])

    def simulate(self):
        step_desc = "シミュレーション中の精子数進捗"
        for j in tqdm(range(self.number_of_sperm), desc=step_desc, ncols=100):
            base_position = self.trajectory[j, 0]
            temp_position = self.trajectory[j, 1]
            remaining_distance = self.constants['step_length'] 
            self.single_sperm_simulation(
                j, base_position, temp_position,
                remaining_distance, self.constants
            )

    def is_vector_meeting_egg(self, base_position, temp_position, egg_center, gamete_R):
        vector = temp_position - base_position
        if LA.norm(vector) < 1e-9:
            sys.exit("zzz")
        distance_base = LA.norm(base_position - egg_center)
        distance_tip = LA.norm(temp_position - egg_center)
        if distance_base <= gamete_R or distance_tip <= gamete_R:
            return True
        f = base_position - egg_center
        a = vector @ vector
        b = 2 * (f @ vector)
        c = f @ f - gamete_R**2
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            return False
        sqrt_discriminant = np.sqrt(discriminant)
        t1 = (-b - sqrt_discriminant) / (2*a)
        t2 = (-b + sqrt_discriminant) / (2*a)
        if (0 <= t1 <= 1) or (0 <= t2 <= 1):
            return True
        return False

    def single_sperm_simulation(self, j, base_position, temp_position, remaining_distance, constants):
        if constants['reflection_analysis'] == "yes":
            stick_status = self.initial_stick
        else:
            stick_status = 0

        self.trajectory[j, 0] = base_position
        i = 1
        intersection_point = np.array([])
        shape = constants['shape']
        gamete_R = constants['gamete_R']

        (egg_x, egg_y, egg_z,
         e_x_min, e_y_min, e_z_min,
         e_x_max, e_y_max, e_z_max,
         egg_center, egg_position_4d) = placement_of_eggs(constants)

        if self.n_stop is not None and not np.isnan(self.n_stop):
            max_steps = int(self.n_stop)
        else:
            max_steps = self.n_simulation

        while i < self.n_simulation:
            if shape in ["cube", "ceros"]:
                new_IO_status, vertex_point = IO_check_cube(temp_position, constants)
            elif shape == "drop":
                new_IO_status = IO_check_drop(temp_position, stick_status, constants)
                vertex_point = None

                if new_IO_status == "border":
                    vec = temp_position - base_position
                    vec_length = np.linalg.norm(vec)
                    if vec_length > constants['limit']:
                        adjusted_vec = vec * 0.99
                        temp_position = base_position + adjusted_vec
                        new_IO_status = IO_check_drop(temp_position, stick_status, constants)

                    if new_IO_status == "border":
                        sys.exit("drop: rethink logic for border")

            elif shape == "spot":
                prev_stat = self.prev_IO_status[j]
                if prev_stat is None:
                    prev_stat = "none"

                new_IO_status = IO_check_spot(base_position, temp_position, constants, prev_stat)
                vertex_point = None

                if new_IO_status == "border":
                    vec = temp_position - base_position
                    vec_length = np.linalg.norm(vec)
                    if vec_length > constants['limit']:
                        adjusted_vec = vec * 0.99
                        temp_position = base_position + adjusted_vec
                        new_IO_status = IO_check_spot(base_position, temp_position, constants, prev_stat)

                    if new_IO_status == "border":
                        sys.exit("rethink logic 3")

            else:
                new_IO_status = "inside"
                vertex_point = None

            prev_stat = self.prev_IO_status[j]
            if prev_stat in ["temp_on_edge", "temp_on_surface"] and (stick_status > 0):
                if new_IO_status in [
                    "inside",
                    "temp_on_polygon",
                    "temp_on_surface",
                    "temp_on_edge",
                    "spot_bottom"
                ]:
                    new_IO_status = prev_stat
                    vertex_point = None

            IO_status = new_IO_status
            self.prev_IO_status[j] = IO_status
            if remaining_distance < 0:
                sys.exit("rd<0")

            if IO_status in [
                "inside",
                "temp_on_surface",
                "temp_on_edge",
                "spot_bottom",
                "on_edge_bottom",
                "temp_on_polygon"
            ]:
                self.trajectory[j, i] = temp_position
                base_position = self.trajectory[j, i]
                remaining_distance = constants['step_length']

                if stick_status > 0:
                    stick_status -= 1

                if len(intersection_point) != 0:
                    last_vec = temp_position - intersection_point
                    intersection_point = np.array([])
                else:
                    last_vec = self.trajectory[j, i] - self.trajectory[j, i - 1]

                # 卵子との接触判定
                if self.is_vector_meeting_egg(self.trajectory[j, i - 1], temp_position, egg_center, gamete_R):
                    self.intersection_records.append((j, i))
                    self.vec_colors[j, i - 1] = "red"
                    self.vec_thickness_2d[j, i - 1] = 2.0
                    self.vec_thickness_3d[j, i - 1] = 4.0

                if IO_status == "temp_on_edge":
                    inward_dir = face_and_inward_dir(
                        temp_position, base_position, last_vec, IO_status, stick_status, constants
                    )
                    if inward_dir is None:
                        inward_dir = np.array([0, 0, 1], dtype=float)
                    if stick_status > 0:
                        temp_position = base_position + last_vec
                    else:
                        new_vec = prepare_new_vector(
                            last_vec, constants,
                            boundary_type="edge",
                            stick_status=stick_status,
                            inward_dir=inward_dir
                        )
                        temp_position = base_position + new_vec

                elif IO_status == "temp_on_surface":
                    inward_dir = face_and_inward_dir(
                        temp_position, base_position, last_vec, IO_status, stick_status, constants
                    )
                    if inward_dir is None:
                        inward_dir = np.array([0, 0, 1], dtype=float)

                    new_vec = prepare_new_vector(
                        last_vec, constants,
                        boundary_type="surface",
                        stick_status=stick_status,
                        inward_dir=inward_dir
                    )
                    temp_position = base_position + new_vec

                elif IO_status == "spot_bottom":
                    inward_dir = [0, 0, 1]
                    new_vec = prepare_new_vector(
                        last_vec, constants,
                        boundary_type="surface",
                        stick_status=stick_status,
                        inward_dir=inward_dir
                    )
                    temp_position = base_position + new_vec

                elif IO_status == "on_edge_bottom":
                    sys.exit("ありえるのか？")

                elif IO_status == "temp_on_polygon":
                    inward_dir = face_and_inward_dir(
                        temp_position, base_position, last_vec, IO_status, stick_status, constants
                    )
                    if inward_dir is None:
                        inward_dir = np.array([0, 0, 1], dtype=float)
                    new_vec = prepare_new_vector(
                        last_vec, constants,
                        boundary_type="polygon",
                        stick_status=stick_status,
                        inward_dir=inward_dir
                    )
                    temp_position = base_position + new_vec

                else:  # "inside"
                    self.trajectory[j, i] = temp_position
                    base_position = self.trajectory[j, i]
                    remaining_distance = constants['step_length']
                    if stick_status > 0:
                        stick_status -= 1

                    if len(intersection_point) != 0:
                        last_vec = temp_position - intersection_point
                        intersection_point = np.array([])
                    else:
                        last_vec = self.trajectory[j, i] - self.trajectory[j, i - 1]

                    if LA.norm(last_vec) < constants['limit']:
                        sys.exit("last vec is too short!")
                    new_vec = prepare_new_vector(
                        last_vec, constants,
                        boundary_type="free",
                        stick_status=stick_status,
                        inward_dir=None
                    )
                    temp_position = self.trajectory[j, i] + new_vec

                i += 1
                continue

            elif IO_status == "sphere_out":
                if stick_status == 0:
                    stick_status = int(constants['stick_sec'] * constants['sampl_rate_Hz'])
                new_temp_pos, intersection_point, remaining_dist, inward_dir = cut_and_bend_sphere(
                    self.trajectory[j, i - 1],
                    remaining_distance,
                    temp_position,
                    constants
                )
                base_position = intersection_point
                temp_position = new_temp_pos
                last_vec = temp_position - intersection_point
                continue

            elif IO_status == "polygon_mode":
                self.trajectory[j, i] = temp_position
                base_position = self.trajectory[j, i]
                if len(intersection_point) != 0:
                    last_vec = temp_position - intersection_point
                    intersection_point = np.array([])
                else:
                    last_vec = self.trajectory[j, i] - self.trajectory[j, i - 1]

                new_temp_position, new_last_vec, updated_stick, next_state = self.bottom_edge_mode(
                    base_position, last_vec, stick_status, constants
                )
                temp_position = new_temp_position
                last_vec = new_last_vec
                stick_status = updated_stick
                i += 1
                IO_status = next_state
                continue

            elif IO_status == "spot_edge_out":
                if stick_status == 0:
                    stick_status = int(constants['stick_sec'] * constants['sampl_rate_Hz'])
                (new_temp_pos,
                 intersection_point,
                 remaining_distance,
                 is_bottom_edge) = cut_and_bend_spot_edge_out(
                     self, IO_status, base_position, temp_position, remaining_distance, constants
                )
                base_position = intersection_point
                temp_position = new_temp_pos
                last_vec = temp_position - intersection_point
                continue

            elif IO_status == "bottom_out":
                if stick_status == 0:
                    stick_status = int(constants['stick_sec'] * constants['sampl_rate_Hz'])
                (new_temp_pos,
                 intersection_point,
                 remaining_distance) = cut_and_bend_bottom(
                     self, IO_status, base_position, temp_position, remaining_distance, constants
                )
                base_position = intersection_point
                temp_position = new_temp_pos
                last_vec = temp_position - intersection_point
                continue

            elif IO_status in ["surface_out", "edge_out"]:
                if stick_status == 0:
                    stick_status = int(constants['stick_sec'] * constants['sampl_rate_Hz'])
                (new_temp_pos,
                 intersection_point,
                 remaining_distance) = cut_and_bend_cube(
                     self, IO_status, base_position, temp_position, remaining_distance, constants
                )
                base_position = intersection_point
                temp_position = new_temp_pos
                last_vec = temp_position - intersection_point
                continue

            elif IO_status == "vertex_out":
                if stick_status == 0:
                    stick_status = int(constants['stick_sec'] * constants['sampl_rate_Hz'])
                (intersection_point,
                 new_temp_pos,
                 remaining_distance) = cut_and_bend_vertex(
                     vertex_point, base_position, remaining_distance, constants
                )
                base_position = intersection_point
                temp_position = new_temp_pos
                last_vec = temp_position - intersection_point
                continue

    def bottom_edge_mode(self, base_position, last_vec, stick_status, constants):
        """
        底面を這いつつ底面の円周に当たった時の処理。
        後者コードを反映して修正。 polygon_mode から呼ばれる。
        """
        z_floor = constants['spot_bottom_height']
        r_edge  = constants['spot_bottom_R']
        R_spot  = constants['radius']

        candidate_position = base_position + last_vec
        candidate_position[2] = z_floor
        dist2 = candidate_position[0]**2 + candidate_position[1]**2
        radius2 = r_edge**2

        if dist2 > radius2:
            # 円周を越えるので衝突点を計算
            x0, y0 = base_position[0], base_position[1]
            x1, y1 = candidate_position[0], candidate_position[1]
            dx = x1 - x0
            dy = y1 - y0
            A = dx**2 + dy**2
            B = 2*(x0*dx + y0*dy)
            C = x0**2 + y0**2 - r_edge**2
            discriminant = B**2 - 4*A*C
            if discriminant < 0:
                discriminant = 0
            sqrt_discriminant = np.sqrt(discriminant)

            t1 = (-B + sqrt_discriminant)/(2*A)
            t2 = (-B - sqrt_discriminant)/(2*A)
            t_candidates = [t for t in [t1,t2] if 0<=t<=1]
            t_intersect = min(t_candidates) if t_candidates else 0.0

            xi = x0 + t_intersect*dx
            yi = y0 + t_intersect*dy
            intersection_point = np.array([xi, yi, z_floor], dtype=float)
            distance_to_intersection = np.linalg.norm(intersection_point - base_position)
            new_remaining = np.linalg.norm(last_vec) - distance_to_intersection
            if new_remaining < 0:
                new_remaining = 0

            if stick_status > 0:
                # 這い続けているとき
                bi = intersection_point - base_position
                bi_norm = np.linalg.norm(bi)
                if bi_norm < constants['limit']:
                    bi_norm = 1e-8
                bi_normalized = bi / bi_norm

                oi = np.array([xi, yi, 0.0])
                oi_norm = np.linalg.norm(oi)
                if oi_norm < constants['limit']:
                    oi_norm = 1e-8
                oi_normalized = oi / oi_norm

                tangent_1 = np.array([-oi_normalized[1], oi_normalized[0], 0])
                tangent_2 = -tangent_1
                angle_with_t1 = np.arccos(
                    np.clip(tangent_1[:2] @ bi_normalized[:2], -1.0,1.0)
                )
                angle_with_t2 = np.arccos(
                    np.clip(tangent_2[:2] @ bi_normalized[:2], -1.0,1.0)
                )
                if angle_with_t1 < angle_with_t2:
                    selected_tangent = tangent_1
                else:
                    selected_tangent = tangent_2

                cross_val = (selected_tangent[0]*bi_normalized[1]
                             - selected_tangent[1]*bi_normalized[0])
                modify_angle = constants['inner_angle']
                if cross_val > 0:
                    angle_adjust = -modify_angle
                else:
                    angle_adjust = modify_angle

                def rotate_vector_2d(vec, angle):
                    c = np.cos(angle)
                    s = np.sin(angle)
                    x_new = vec[0]*c - vec[1]*s
                    y_new = vec[0]*s + vec[1]*c
                    return np.array([x_new,y_new,0])

                new_tangent = rotate_vector_2d(selected_tangent, angle_adjust)
                norm_tan = np.linalg.norm(new_tangent)
                if norm_tan < constants['limit']:
                    new_tangent = selected_tangent
                    norm_tan = np.linalg.norm(new_tangent)

                new_tangent /= norm_tan
                last_vec_corrected = new_tangent * new_remaining
                new_temp_position = intersection_point + last_vec_corrected
                new_temp_position[2] = z_floor
                new_last_vec = new_temp_position - intersection_point

            else:
                # 這っていない（stick_status=0）ので、ランダムに少し反れる
                new_remaining = np.linalg.norm(last_vec)
                sphere_normal_3d = intersection_point
                norm_sphere = np.linalg.norm(sphere_normal_3d)
                if norm_sphere < constants['limit']:
                    sphere_normal_3d = np.array([0,0,1])
                    norm_sphere = 1.0
                sphere_normal_3d /= norm_sphere

                plane_normal = np.array([0,0,1], dtype=float)
                dot_val = np.clip(np.dot(sphere_normal_3d, plane_normal), -1, 1)
                angle_plane_sphere = np.arccos(dot_val)

                def sample_vector_in_cone(axis, max_angle):
                    cos_max = np.cos(max_angle)
                    z_ = np.random.uniform(cos_max, 1.0)
                    phi_ = np.random.uniform(0, 2*np.pi)
                    sqrt_part = np.sqrt(1 - z_*z_)
                    x_local = sqrt_part * np.cos(phi_)
                    y_local = sqrt_part * np.sin(phi_)
                    z_local = z_
                    rot = R.align_vectors([[0,0,1]], [axis])[0]
                    v_local = np.array([x_local, y_local, z_local])
                    return rot.apply(v_local)

                center_axis = (plane_normal + sphere_normal_3d) / 2
                center_axis_norm = np.linalg.norm(center_axis)
                if center_axis_norm < constants['limit']:
                    center_axis = plane_normal
                    center_axis_norm = 1.0
                center_axis /= center_axis_norm

                open_angle = angle_plane_sphere
                random_3d_dir = sample_vector_in_cone(center_axis, open_angle)
                last_vec_corrected = random_3d_dir * new_remaining

                new_temp_position = intersection_point + last_vec_corrected
                new_last_vec = new_temp_position - intersection_point

        else:
            # 円周に達してないならそのまま進める
            new_temp_position = candidate_position
            new_last_vec = last_vec

        new_stick_status = stick_status
        if new_stick_status > 0:
            new_stick_status -= 1

        # stick_statusが無くなったら(=0になったら) 這いをやめて "inside" に戻る
        if new_stick_status <= 0:
            new_state = "inside"
        else:
            new_state = "bottom_edge_mode"

        return new_temp_position, new_last_vec, new_stick_status, new_state

# ------------------------------------------------------------
# 描画関連
# ------------------------------------------------------------
class SpermPlot:
    already_saved_global_flag = False

    def __init__(self, simulation):
        self.simulation = simulation
        self.constants = self.simulation.constants
        self.colors = self.simulation.vec_colors

        # 安全に値を取得し、設定されていない場合は set_min_max を使って設定
        self.x_min = self.constants.get('x_min', None)
        self.x_max = self.constants.get('x_max', None)
        self.y_min = self.constants.get('y_min', None)
        self.y_max = self.constants.get('y_max', None)
        self.z_min = self.constants.get('z_min', None)
        self.z_max = self.constants.get('z_max', None)

        if None in (self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max):
            self.set_min_max(self.constants['volume'])

    def set_min_max(self, volume):
        shape = self.constants['shape']
        if shape == "cube":
            edge_length = volume ** (1 / 3)
            half_edge = edge_length / 2
            self.x_min, self.x_max = -half_edge, half_edge
            self.y_min, self.y_max = -half_edge, half_edge
            self.z_min, self.z_max = -half_edge, half_edge
        elif shape == "spot":
            r = self.constants['radius']
            spot_bottom_height = self.constants['spot_bottom_height']
            self.x_min = -r
            self.x_max = r
            self.y_min = -r
            self.y_max = r
            self.z_min = spot_bottom_height
            self.z_max = r
        elif shape == "drop":
            drop_R = self.constants['drop_R']
            self.x_min = self.y_min = self.z_min = -drop_R
            self.x_max = self.y_max = self.z_max = drop_R
        elif shape == "ceros":
            self.x_min = -8.15
            self.x_max = 8.15
            self.y_min = -6.2
            self.y_max = 6.2
            self.z_min = -0.05
            self.z_max = 0.05
        return (self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max)

    def set_ax_3D(self, ax):
        shape = self.constants['shape']
        if shape == "cube":
            ax.set_xlim(self.x_min, self.x_max)
            ax.set_ylim(self.y_min, self.y_max)
            ax.set_zlim(self.z_min, self.z_max)
            ax.set_box_aspect([
                (self.x_max - self.x_min),
                (self.y_max - self.y_min),
                (self.z_max - self.z_min)
            ])
        elif shape == "spot":
            r = self.constants['radius']
            ax.set_xlim(-r, r)
            ax.set_ylim(-r, r)
            ax.set_zlim(self.constants['spot_bottom_height'], r)
            ax.set_box_aspect([2*r, 2*r, r - self.constants['spot_bottom_height']])

        elif shape == "drop":
            drop_R = self.constants['drop_R']
            ax.set_xlim(-drop_R, drop_R)
            ax.set_ylim(-drop_R, drop_R)
            ax.set_zlim(-drop_R, drop_R)
            ax.set_box_aspect([1,1,1])

        elif shape == "ceros":
            ax.set_xlim([-8.15, 8.15])
            ax.set_ylim([-6.2, 6.2])
            ax.set_zlim([-0.05, 0.05])
            ax.set_box_aspect([16.3, 13.4, 0.1])
        else:
            raise ValueError(f"Unknown shape: {shape}")

    def _draw_graph(self, shape):
        if hasattr(self, "already_saved") and self.already_saved:
            return None
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=300)
        if shape != "ceros":
            (egg_x, egg_y, egg_z,
            e_x_min, e_y_min, e_z_min,
            e_x_max, e_y_max, e_z_max,
            egg_center, egg_position_4d) = placement_of_eggs(self.constants)

            def draw_egg():
                egg_positions = [
                    (egg_x, egg_y),
                    (egg_x, egg_z),
                    (egg_y, egg_z)
                ]
                for ax, (x, y) in zip(axes, egg_positions):
                    ax.add_patch(
                        patches.Circle(
                            (x, y),
                            radius=self.constants['gamete_R'],
                            ec='None',
                            facecolor='yellow',
                            fill=True,
                            linewidth=1,
                            alpha=0.8
                        )
                    )
            draw_egg()

        axis_combinations = (
            [('X', 'Y', 0), ('X', 'Z', 1), ('Y', 'Z', 2)]
            if shape != 'ceros'
            else [('X', 'Y', 0), ('X', 'Y', 1), ('X', 'Z', 2)]
        )
        index_map = {'X': 0, 'Y': 1, 'Z': 2}

        progress_bar = tqdm(
            total=(
                self.simulation.number_of_sperm *
                (self.simulation.n_simulation - 1) *
                len(axis_combinations)
            ),
            desc="Plotting trajectories",
            leave=True,
            ncols=100,
            file=sys.stdout,
            dynamic_ncols=True,
            ascii=True
        )

        for j in range(self.simulation.number_of_sperm):
            for i in range(int(self.simulation.n_simulation) - 1):
                for axis1, axis2, idx in axis_combinations:
                    if shape == 'ceros' and idx != 1:
                        progress_bar.update(1)
                        continue

                    axes[idx].plot(
                        self.simulation.trajectory[j, i:i+2, index_map[axis1]],
                        self.simulation.trajectory[j, i:i+2, index_map[axis2]],
                        color=self.simulation.vec_colors[j, i],
                        linewidth=self.simulation.vec_thickness_2d[j, i],
                        antialiased=True
                    )
                    progress_bar.update(1)

        progress_bar.close()
        print("Now, saving figures!")

        def set_axis_limits(shape):
            if shape == 'ceros':
                axes[1].set_xlim(-0.815, 0.815)
                axes[1].set_ylim(-0.62, 0.62)

            elif shape == 'spot':
                spot_R = self.constants['spot_R']
                spot_bottom_R = self.constants['spot_bottom_R']  # 底面円周の半径
                spot_bottom_height = self.constants['spot_bottom_height']
                gamete_R = self.constants['gamete_R']
                axes[0].set_xlim(-spot_bottom_R, spot_bottom_R)
                axes[0].set_ylim(-spot_bottom_R, spot_bottom_R)

                axes[1].set_xlim(-spot_bottom_R, spot_bottom_R)
                axes[1].set_ylim(spot_bottom_height, spot_R)

                axes[2].set_xlim(-spot_bottom_R, spot_bottom_R)
                axes[2].set_ylim(spot_bottom_height, spot_R)

                for ax in axes:
                    ax.set_aspect('equal', adjustable='box')

            elif shape == 'drop':
                drop_R = self.constants.get('drop_R', self.constants['radius'])
                for ax in axes:
                    ax.set_xlim(-drop_R, drop_R)
                    ax.set_ylim(-drop_R, drop_R)
                    ax.set_aspect('equal', adjustable='box')

            elif shape == 'cube':
                x_min = self.x_min
                x_max = self.x_max
                y_min = self.y_min
                y_max = self.y_max
                for ax in axes:
                    ax.set_xlim(x_min, x_max)
                    ax.set_ylim(y_min, y_max)
                    ax.set_aspect('equal', adjustable='box')

        def draw_motion_area(shape):
            if shape == 'spot':
                spot_bottom_radius = self.constants.get('spot_bottom_R', 1)
                axes[0].add_patch(
                    patches.Circle((0, 0), spot_bottom_radius, ec='none', facecolor='red', alpha=0.1)
                )
                for ax in axes[1:]:
                    ax.add_patch(
                        patches.Circle((0, 0), self.constants.get('spot_R', 0), ec='none', facecolor='red', alpha=0.1)
                    )
                for ax in axes[1:]:
                    ax.axhline(self.constants.get('spot_bottom_height', 0), color='gray', linestyle='--', linewidth=0.8)

            elif shape == 'drop':
                drop_R = self.constants.get('drop_R', self.constants['radius'])
                for ax in axes:
                    ax.add_patch(
                        patches.Circle((0, 0), drop_R, ec='none', facecolor='red', alpha=0.1)
                    )

        set_axis_limits(shape)
        draw_motion_area(shape)

        margin_factor = 0.0
        for ax in axes:
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            dx = x_max - x_min
            dy = y_max - y_min
            ax.set_xlim(x_min - margin_factor * dx, x_max + margin_factor * dx)
            ax.set_ylim(y_min - margin_factor * dy, y_max + margin_factor * dy)

        plt.tight_layout(rect=[0, 0, 1, 0.93], w_pad=0.5, h_pad=0.5)

        merged_events = self.simulation.merge_contact_events()
        contacts_per_hour = len(merged_events) / (self.constants['sim_min'] / 60)

        title_str = (
            f"vol: {self.constants['volume']} μl, "
            f"conc: {self.constants['sperm_conc']}/ml, "
            f"VSL: {self.constants['VSL']} mm, "
            f"sampling: {self.constants['sampl_rate_Hz']} Hz, "
            f"dev: {self.constants['deviation']}, "
            f"stick: {self.constants['stick_sec']} sec,\n"
            f"sperm/egg interaction: {len(merged_events)} during {self.constants['sim_min']} min, "
            f"egg: {self.constants['egg_localization']}, "
        )

        if shape == "spot":
            title_str += (
                f"spot_angle: {self.constants.get('spot_angle', 'N/A')} degree"
            )
        fig.suptitle(title_str, fontsize=8, y=0.98)

        axes[0].set_xlabel('x', fontsize=8)
        axes[0].set_ylabel('y', fontsize=8)
        axes[1].set_xlabel('x', fontsize=8)
        axes[1].set_ylabel('z', fontsize=8)
        axes[2].set_xlabel('y', fontsize=8)
        axes[2].set_ylabel('z', fontsize=8)
        for ax in axes:
            ax.tick_params(axis='both', labelsize=8)

        plt.tight_layout(rect=[0, 0, 1, 0.95], w_pad=0.5, h_pad=0.5)
        output_dir = DATA_FOLDER
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"graph_output_{timestamp}.svg"
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, format='svg')
        print(f"{output_path}")

        self.already_saved = True
        plt.close(fig)
        return output_path

class SpermTrajectoryVisualizer:
    def __init__(self, simulation):
        self.simulation = simulation
        self.constants = self.simulation.constants
        self.sperm_plot = SpermPlot(self.simulation)

        (
            egg_x, egg_y, egg_z,
            e_x_min, e_y_min, e_z_min,
            e_x_max, e_y_max, e_z_max,
            egg_center, egg_position_4d
        ) = placement_of_eggs(self.constants)

        self.egg_center = np.array([egg_x, egg_y, egg_z])
        self.egg_radius = self.constants['gamete_R']

    def animate_trajectory(self):
        if self.constants.get("make_movie", "no").lower() != "yes":
            return None

        shape = self.constants.get("shape", "spot")
        num_sperm = self.simulation.number_of_sperm
        n_sim = self.simulation.n_simulation

        if shape == "ceros":
            plt.ion()
            fig, ax = plt.subplots()
            ax.set_xlim(-0.815, 0.815)
            ax.set_ylim(-0.62, 0.62)
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_title("CEROS 2D Animation (Zoomed)")

            lines = [ax.plot([], [], lw=2)[0] for _ in range(num_sperm)]

            def init():
                for line in lines:
                    line.set_data([], [])
                return lines

            def animate(i):
                if i % 10 == 0:
                    percentage = (i / (n_sim - 1)) * 100
                    print(f"Progress: {percentage:.2f}%")

                for j, line in enumerate(lines):
                    base_pos = self.simulation.trajectory[j, i]
                    end_pos  = self.simulation.trajectory[j, i + 1]
                    xdata = [base_pos[0], end_pos[0]]
                    ydata = [base_pos[1], end_pos[1]]
                    line.set_data(xdata, ydata)
                    line.set_color(self.simulation.vec_colors[j, i])
                    line.set_linewidth(self.simulation.vec_thickness_3d[j, i])
                return lines

            anim = FuncAnimation(
                fig,
                animate,
                init_func=init,
                frames=n_sim - 1,
                interval=180,
                blit=False
            )

            output_folder = DATA_FOLDER
            os.makedirs(output_folder, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            mov_filename = f"sperm_simulation_ceros_{timestamp}.mp4"
            output_path = os.path.join(output_folder, mov_filename)

            anim.save(output_path, writer='ffmpeg', codec='mpeg4', fps=5)
            print(f"{output_path}")

            plt.show()
            return output_path

        else:
            plt.ion()
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            merged_events = self.simulation.merge_contact_events()
            contacts_count = len(merged_events)
            if self.constants["sim_min"] > 0:
                contacts_per_hour = contacts_count / (self.constants["sim_min"] / 60)
            else:
                contacts_per_hour = 0

            title_str_3d = (
                f"vol: {self.constants['volume']} μl, "
                f"conc: {self.constants['sperm_conc']}/ml, "
                f"VSL: {self.constants['VSL']} mm, "
                f"sampling: {self.constants['sampl_rate_Hz']} Hz, "
                f"dev: {self.constants['deviation']}, "
                f"stick: {self.constants['stick_sec']} sec,\n"
                f"sperm/egg interaction: {contacts_count} during {self.constants['sim_min']} min, "
                f"egg: {self.constants['egg_localization']}, "
            )
            if shape == "spot":
                title_str_3d += f"spot_angle: {self.constants.get('spot_angle', 'N/A')} degree"

            fig.suptitle(title_str_3d, fontsize=8, y=0.93)

            egg_u = np.linspace(0, 2 * np.pi, 50)
            egg_v = np.linspace(0, np.pi, 50)
            ex = (
                self.egg_center[0]
                + self.egg_radius * np.outer(np.cos(egg_u), np.sin(egg_v))
            )
            ey = (
                self.egg_center[1]
                + self.egg_radius * np.outer(np.sin(egg_u), np.sin(egg_v))
            )
            ez = (
                self.egg_center[2]
                + self.egg_radius * np.outer(
                    np.ones(np.size(egg_u)), np.cos(egg_v)
                )
            )
            ax.plot_surface(ex, ey, ez, color='yellow', alpha=0.2)

            if shape == "spot":
                spot_R = self.constants.get('spot_R', 5)
                spot_angle_deg = self.constants.get('spot_angle', 60)
                shape_u = np.linspace(0, 2*np.pi, 60)
                theta_max_rad = np.deg2rad(spot_angle_deg)
                shape_v = np.linspace(0, theta_max_rad, 60)
                sx = spot_R * np.outer(np.sin(shape_v), np.cos(shape_u))
                sy = spot_R * np.outer(np.sin(shape_v), np.sin(shape_u))
                sz = spot_R * np.outer(np.cos(shape_v), np.ones(np.size(shape_u)))
                ax.plot_surface(sx, sy, sz, color='red', alpha=0.15)

            elif shape == "drop":
                drop_R = self.constants.get('drop_R', 5)
                shape_u = np.linspace(0, 2*np.pi, 60)
                shape_v = np.linspace(0, np.pi, 60)
                sx = drop_R * np.outer(np.sin(shape_v), np.cos(shape_u))
                sy = drop_R * np.outer(np.sin(shape_v), np.sin(shape_u))
                sz = drop_R * np.outer(np.cos(shape_v), np.ones(np.size(shape_u)))
                ax.plot_surface(sx, sy, sz, color='red', alpha=0.15)

            lines = [ax.plot([], [], [], lw=2)[0] for _ in range(num_sperm)]

            def init():
                for line in lines:
                    line.set_data([], [])
                    line.set_3d_properties([])
                return lines

            def animate(i):
                if i % 10 == 0:
                    percentage = (i / (n_sim - 1)) * 100
                    print(f"rProgress: {percentage:.2f}%")

                for j, line in enumerate(lines):
                    base_pos = self.simulation.trajectory[j, i]
                    end_pos = self.simulation.trajectory[j, i + 1]
                    line.set_data(
                        [base_pos[0], end_pos[0]],
                        [base_pos[1], end_pos[1]]
                    )
                    line.set_3d_properties([base_pos[2], end_pos[2]])
                    line.set_color(self.simulation.vec_colors[j, i])
                    line.set_linewidth(self.simulation.vec_thickness_3d[j, i])
                return lines

            self.sperm_plot.set_min_max(self.constants.get('volume', 1))
            self.sperm_plot.set_ax_3D(ax)

            anim = FuncAnimation(
                fig,
                animate,
                init_func=init,
                frames=n_sim - 1,
                interval=180,
                blit=False
            )

            output_folder = DATA_FOLDER
            os.makedirs(output_folder, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            mov_filename = f"sperm_simulation_{timestamp}.mp4"
            output_path = os.path.join(output_folder, mov_filename)

            anim.save(output_path, writer='ffmpeg', codec='mpeg4', fps=5)
            print(f"{output_path}")

            plt.show()
            return output_path

# =========================
# ▼▼▼ DB部分 （省略可・修正なし）
# =========================
def setup_database(conn):
    c = conn.cursor()

    # 必要なテーブルをすべて作成する
    c.execute('''
        CREATE TABLE IF NOT EXISTS basic_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            exp_id          TEXT,
            version         TEXT,
            shape           TEXT,
            volume          REAL,
            sperm_conc      INTEGER,
            N_contact       INTEGER,
            VSL             REAL,
            stick_sec       INTEGER,
            sim_min         REAL,
            deviation       REAL,
            egg_localization TEXT,
            image_id        TEXT,
            mov_id          TEXT,
            spot_angle      INTEGER,
            sampl_rate_Hz   INTEGER,
            seed_number     TEXT
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS intersection (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            simulation_id  INTEGER,
            sperm_index    INTEGER,
            start_step     INTEGER
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS summary (
            simulation_id    INTEGER,
            shape            TEXT,
            sperm_conc       INTEGER,
            volume           REAL,
            VSL              REAL,
            deviation        REAL,
            sim_min          REAL,
            stick_sec        INTEGER,
            spot_angle       INTEGER,
            sampl_rate_Hz    INTEGER,
            egg_localization TEXT,            -- ← 忘れずに入れる
            mean_contact_hr  REAL,
            SD1              REAL,
            N_sperm          INTEGER,
            C_per_N          REAL,
            SD2              REAL,
            total_simulations INTEGER
        )
    ''')
    conn.commit()

# 例: 以前の絶対パス
# db_path = '/path/to/Trajectory.db'


def insert_sim_record(conn, exp_id, version, constants, image_id, mov_id, contact_count):
    """
    basic_data テーブルに 1 件分のシミュレーション結果を挿入し、
    挿入した行の simulation_id を返す。
    egg_localization は .strip() で前後の余計な空白・不可視文字を除去してから書き込む。
    """
    c = conn.cursor()

    # spot 形状なら spot_angle、そうでなければ None
    if constants['shape'] == 'spot':
        spot_angle_val = int(constants['spot_angle'])
    else:
        spot_angle_val = None

    # egg_localization を必ず strip() してクリーンに
    raw_egg_loc = constants.get('egg_localization', 'bottom_center')
    egg_loc = raw_egg_loc.strip()

    # その他のフィールド
    seed_number = constants.get('seed_number', 'None')

    c.execute(
        '''
        INSERT INTO basic_data (
            exp_id,
            version,
            shape,
            volume,
            sperm_conc,
            N_contact,
            VSL,
            stick_sec,
            sim_min,
            deviation,
            egg_localization,
            image_id,
            mov_id,
            spot_angle,
            sampl_rate_Hz,
            seed_number
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''',
        (
            exp_id,
            version,
            constants['shape'],
            constants['volume'],
            int(constants['sperm_conc']),
            contact_count,
            constants['VSL'],
            int(constants['stick_sec']),
            constants['sim_min'],
            constants['deviation'],
            egg_loc,
            image_id,
            mov_id,
            spot_angle_val,
            int(constants['sampl_rate_Hz']),
            seed_number
        )
    )
    conn.commit()

    # 挿入された行の simulation_id を返す
    return c.lastrowid

def insert_intersection_records(conn, simulation_id, merged_events):
    c = conn.cursor()
    for (sperm_index, start_step) in merged_events:
        c.execute('''
            INSERT INTO intersection (simulation_id, sperm_index, start_step)
            VALUES (?, ?, ?)
        ''', (simulation_id, sperm_index, start_step))
    conn.commit()

def aggregate_results(conn, exp_id):
    c = conn.cursor()
    # egg_localization を SELECT に追加
    c.execute('''
        SELECT id, shape, sperm_conc, volume, VSL, deviation,
               sim_min, stick_sec, spot_angle, sampl_rate_Hz,
               egg_localization, N_contact
          FROM basic_data
    ''')
    rows = c.fetchall()

    import statistics
    from collections import defaultdict
    grouping = defaultdict(list)

    # basic_data の各レコードを、egg_localization ごとにグループ化
    for rec in rows:
        sim_id            = rec[0]
        shape_val         = rec[1]
        sperm_conc_val    = rec[2]
        volume_val        = float(rec[3])
        VSL_val           = float(rec[4])
        deviation_val     = float(rec[5])
        sim_min           = float(rec[6])
        stick_sec         = rec[7]
        spot_angle        = rec[8]
        sampl_rate_Hz     = rec[9]
        egg_loc_val       = rec[10]
        N_contact         = rec[11]

        N_sperm = int(sperm_conc_val * volume_val / 1000)
        ratio   = (N_contact / sim_min * 60.0) if sim_min > 0 else 0.0
        cps     = (N_contact / N_sperm) if N_sperm != 0 else 0.0

        # egg_localization をキーに含める
        key = (
            shape_val, sperm_conc_val, volume_val, VSL_val,
            deviation_val, sim_min, stick_sec, spot_angle,
            sampl_rate_Hz, egg_loc_val
        )
        grouping[key].append((sim_id, ratio, N_sperm, cps))

    # summary テーブルへの挿入
    for key, values in grouping.items():
        (shape_val, sperm_conc_val, volume_val, VSL_val,
         deviation_val, sim_min, stick_sec, spot_angle,
         sampl_rate_Hz, egg_loc_val) = key

        total_sim         = len(values)
        ratios            = [v[1] for v in values]
        Ns                = [v[2] for v in values]
        cps_list          = [v[3] for v in values]
        mean_contact_hr   = round(statistics.mean(ratios), 1) if ratios else 0.0
        SD1_val           = round(statistics.pstdev(ratios), 1) if len(ratios) > 1 else 0.0
        mean_N_sperm      = int(statistics.mean(Ns)) if Ns else 0
        C_per_N_val       = round(mean_contact_hr / mean_N_sperm, 1) if mean_N_sperm else 0.0
        SD2_val           = round(statistics.pstdev(cps_list), 1) if len(cps_list) > 1 else 0.0
        new_latest_sim_id = max(v[0] for v in values)

        # egg_localization カラムまで含めて INSERT
        c.execute('''
            INSERT OR REPLACE INTO summary (
                simulation_id, shape, sperm_conc, volume, VSL, deviation,
                sim_min, stick_sec, spot_angle, sampl_rate_Hz, egg_localization,
                mean_contact_hr, SD1, N_sperm, C_per_N, SD2, total_simulations
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            new_latest_sim_id, shape_val, sperm_conc_val, volume_val,
            VSL_val, deviation_val, sim_min, stick_sec, spot_angle,
            sampl_rate_Hz, egg_loc_val,
            mean_contact_hr, SD1_val, mean_N_sperm, C_per_N_val,
            SD2_val, total_sim
        ))

    conn.commit()

def calculate_n_sperm(df):
    df["N_sperm"] = df["N_sperm"].astype(int)
    return df

# ------------------------------------------------------------
# ここの show_selection_ui が「最初に最前面に表示」→「OK押下でウィンドウ破棄」→
# もう一度呼べばまた最前面に表示 という動作になります
# ------------------------------------------------------------
def load_previous_selection():
    config = configparser.ConfigParser()
    config.read(["user_selection.ini", "config.ini"])

    def get_list(section, key, default):
        return config.get(section, key, fallback=default).split(",")

    analysis_options = {}
    if "AnalysisOptions" in config:
        for key in config["AnalysisOptions"]:
            analysis_options[key] = get_list("AnalysisOptions", key, "")
    else:
        analysis_options = {}

    analysis_settings = {}
    if "Analysis" in config:
        for key in config["Analysis"]:
            analysis_settings[key] = config.get("Analysis", key, fallback="")
    else:
        analysis_settings = {}

    userselection_section = "UserSelection"

    def get_str(key, default):
        return config.get(userselection_section, key, fallback=default)

    n_repeat_default             = get_str("n_repeat",                  "3")
    seed_number_default          = get_str("seed_number",              "None")
    sim_min_default              = get_str("sim_min",                   "10")
    sampl_rate_hz_default        = get_str("sampl_rate_hz",            "2")
    spot_angle_default           = get_str("spot_angle",               "60")
    vsl_default                  = get_str("vsl",                       "0.15")
    deviation_default            = get_str("deviation",                 "0.4")
    stick_sec_default            = get_str("stick_sec",               "2")
    egg_localization_default     = get_str("egg_localization",          "bottom_center")
    gamete_r_default             = get_str("gamete_r",                 "0.15")
    initial_direction_default    = get_str("initial_direction",         "random")
    initial_stick_default = get_str("initial_stick",      "10")
    analysis_type_default        = get_str("analysis_type",             "simulation")

    def get_list_from_userselection(key, fallback_str, conv_func=None):
        raw_str = config.get(userselection_section, key, fallback=fallback_str)
        splitted = raw_str.split(",")
        splitted = [x for x in splitted if x]
        if conv_func is not None:
            return [conv_func(x) for x in splitted]
        else:
            return splitted

    volumes_list             = get_list_from_userselection("volumes",             "6.25,12.5,25,50,100,200,400,800,1600,3200", float)
    sperm_list               = get_list_from_userselection("sperm_concentrations","1000,3162,10000,31620,100000",         int)
    shapes_list              = get_list_from_userselection("shapes",              "cube,drop,spot",         None)
    outputs_list = get_list_from_userselection("outputs", "graph,movie", None)

    return {
        "analysis_options": analysis_options,
        "analysis_settings": analysis_settings,
        "n_repeat": n_repeat_default,
        "seed_number": seed_number_default,
        "sim_min": sim_min_default,
        "sampl_rate_hz": sampl_rate_hz_default,
        "spot_angle": spot_angle_default,
        "vsl": vsl_default,
        "deviation": deviation_default,
        "stick_sec": stick_sec_default,
        "egg_localization": egg_localization_default,
        "gamete_r": gamete_r_default,
        "initial_direction": initial_direction_default,
        "initial_stick": initial_stick_default,
        "analysis_type": analysis_type_default,
        "volumes": volumes_list,
        "sperm_concentrations": sperm_list,
        "shapes": shapes_list,
        "outputs": outputs_list,
    }

def save_previous_selection(values):
    config = configparser.ConfigParser()
    config.read("user_selection.ini")

    userselection_section = "UserSelection"
    if not config.has_section(userselection_section):
        config.add_section(userselection_section)

    single_keys = [
        "n_repeat", "seed_number", "sim_min", "sampl_rate_hz", "spot_angle",
        "vsl", "deviation", "stick_sec", "egg_localization",
        "gamete_r", "initial_direction", "initial_stick", "analysis_type"
    ]
    for key in single_keys:
        config.set(userselection_section, key, str(values.get(key, "")))

    list_keys = ["volumes", "sperm_concentrations", "shapes", "outputs"]
    for key in list_keys:
        items = values.get(key, [])
        config.set(userselection_section, key, ",".join(map(str, items)))

    with open("user_selection.ini", "w") as configfile:
        config.write(configfile)

def show_selection_ui():
    selections = load_previous_selection()

    def get_str(key, default=""):
        return selections.get(key, default)

    root = tk.Tk()
    root.title("シミュレーションのパラメータ選択")
    root.geometry("600x600")

    # --- 修正箇所 ---
    root.attributes("-topmost", True)  # 最初に最前面表示
    root.update()                      # 表示を即座に反映
    # --- 修正ここまで ---

    canvas = tk.Canvas(root)
    scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)

    def on_configure(event):
        canvas.configure(scrollregion=canvas.bbox("all"))
    scrollable_frame.bind("<Configure>", on_configure)

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    gui_elements = {}
    frames = {}

    radio_button_defaults = {
        "n_repeat_options":               ["1", "2", "3", "4", "5"],
        "seed_number_options":            ["None", "0", "1","2","3"],
        "sim_min_options":                ["0.2", "1", "10", "20","60","100"],
        "sampl_rate_hz_options":          ["1", "2", "3", "4"],
        "spot_angle_options":             ["30", "50","60","70", "90"],
        "vsl_options":                    ["0.073","0.1", "0.11", "0.12", "0.13", "0.14","0.15"],
        "deviation_options":              ["0", "0.2", "0.3","0.4","0.8","1.6", "3.2", "6.4", "12.8"],
        "stick_sec_options":              ["0", "1", "2", "3", "4","5", "6","7","8"],
        "egg_localization_options":       ["bottom_center", "center", "bottom_edge"],
        "analysis_type_options":          ["simulation", "reflection"],
        "initial_direction_options":      ["right", "left", "up", "down", "random"],
        "initial_stick_options":   ["0", "1", "10", "20"],
    }

    ordered_keys = [
        ("繰り返し回数 (N Repeat)",             "n_repeat_options"),
        ("乱数シード (Seed Number)",            "seed_number_options"),
        ("シミュレーション時間 (Sim Min)",      "sim_min_options"),
        ("サンプリングレート (Sample Rate Hz)", "sampl_rate_hz_options"),
        ("形状 (Shape)",                        "shapes"),
        ("Spot角度 (Spot Angle)",              "spot_angle_options"),
        ("体積 (Volume)",                      "volumes"),
        ("精子濃度 (Sperm Conc)",              "sperm_concentrations"),
        ("VSL",                                 "vsl_options"),
        ("偏差 (Deviation)",                   "deviation_options"),
        ("Stick_秒数 (Stick Sec)",             "stick_sec_options"),
        ("卵位置 (Egg Localization)",           "egg_localization_options"),
        ("Gamete半径 (Gamete R)",              "gamete_r_options"),
        ("出力設定 (Outputs)",                 "outputs"),
        ("解析タイプ (Analysis Type)",         "analysis_type_options"),
        ("初期方向 (Initial Direction)",       "initial_direction_options"),
        ("初期Stick状態 (Initial Stick)",      "initial_stick_options"),
    ]

    def update_states():
        shape_selected_spot = False
        if "shapes" in gui_elements and isinstance(gui_elements["shapes"], list):
            shape_selected_spot = any(var.get() and (val == "spot") for val, var in gui_elements["shapes"])

        state_spot_angle = "normal" if shape_selected_spot else "disabled"
        if "spot_angle_options" in frames:
            for child in frames["spot_angle_options"].winfo_children():
                child.config(state=state_spot_angle)

        analysis_type_val = gui_elements["analysis_type"].get() if "analysis_type" in gui_elements else ""
        is_reflection = (analysis_type_val == "reflection")

        state_reflection = "normal" if is_reflection else "disabled"
        for key in ["initial_direction_options", "initial_stick_options"]:
            if key in frames:
                for child in frames[key].winfo_children():
                    child.config(state=state_reflection)

    for label_text, key in ordered_keys:
        frame = tk.LabelFrame(scrollable_frame, text=label_text)
        frame.pack(fill="x", padx=10, pady=5)
        frames[key] = frame

        if key in ["shapes", "outputs", "volumes", "sperm_concentrations"]:
            default_list = selections.get(key, [])
            if key == "shapes":
                options = ["cube", "drop", "spot", "ceros"]
            elif key == "outputs":
                options = ["graph", "movie"]
            elif key == "volumes":
                options = ["6.25", "12.5", "25", "50", "100", "200","400","800","1600","3200"]
            elif key == "sperm_concentrations":
                options = ["1000", "3162", "10000", "31623", "100000"]

            vars_list = []
            for val in options:
                if key == "volumes":
                    conv_val = float(val)
                    is_checked = (conv_val in default_list)
                elif key == "sperm_concentrations":
                    conv_val = int(val)
                    is_checked = (conv_val in default_list)
                else:
                    conv_val = val
                    is_checked = (conv_val in default_list)
                var = tk.BooleanVar(value=is_checked)
                cb = tk.Checkbutton(frame, text=str(val), variable=var, command=update_states)
                cb.pack(side="left", padx=5, pady=5)
                vars_list.append((val, var))
            gui_elements[key] = vars_list

        elif key == "gamete_r_options":
            param_name = "gamete_r"
            default_value = str(selections.get(param_name, "0.1"))
            options = ["0.04", "0.05", "0.15"]
            var = tk.StringVar(value=default_value)
            for val in options:
                rb = tk.Radiobutton(frame, text=str(val), variable=var, value=str(val), command=update_states)
                rb.pack(side="left", padx=5, pady=5)
            gui_elements[param_name] = var

        else:
            param_name = key.replace("_options", "")
            default_value = str(selections.get(param_name, ""))

            possible_options = selections["analysis_options"].get(key, [])
            if not possible_options:
                possible_options = radio_button_defaults.get(key, [])
            if not possible_options:
                possible_options = [default_value]

            var = tk.StringVar(value=default_value)
            for val in possible_options:
                rb = tk.Radiobutton(frame, text=str(val), variable=var, value=str(val), command=update_states)
                rb.pack(side="left", padx=5, pady=5)

            gui_elements[param_name] = var

    update_states()

    def on_ok():
        selected_data = {}
        for key, element in gui_elements.items():
            if isinstance(element, list):
                if key == "volumes":
                    selected_data[key] = [float(val) for (val, var) in element if var.get()]
                elif key == "sperm_concentrations":
                    selected_data[key] = [int(val) for (val, var) in element if var.get()]
                else:
                    selected_data[key] = [val for (val, var) in element if var.get()]
            else:
                selected_data[key] = element.get()

        save_previous_selection(selected_data)
        root.selected_data = selected_data

        # --- 修正箇所（OKが押されたら最前面属性を解除）---
        root.attributes("-topmost", False)
        root.destroy()

    btn_ok = tk.Button(scrollable_frame, text="OK", command=on_ok)
    btn_ok.pack(pady=10)

    root.mainloop()
    return getattr(root, 'selected_data', {})

def repeat_simulation(constants, repeat):
    simulations = []
    for r in range(repeat):
        print(f"n--- Simulation run {r+1} / {repeat} for shape={constants['shape']}, vol={constants['volume']}, conc={constants['sperm_conc']} ---")

        simulation_data = type('simulation_data', (object,), {
            'trajectory': np.zeros((
                int(constants['number_of_sperm']),
                int(constants['n_simulation']),
                3
            ))
        })()

        simulation = SpermSimulation(constants, None, simulation_data)
        visualizer = SpermTrajectoryVisualizer(simulation)
        simulation.visualizer = visualizer
        simulation.simulate()
        merged_events = simulation.merge_contact_events()
        print(f"Simulation run {r+1}: 接触数 = {len(merged_events)}")
        print("1時間あたり", len(merged_events)/constants["sim_min"]*60)

        image_id = None
        mov_id = None

        if constants['draw_trajectory'] == 'yes':
            plot = SpermPlot(simulation)
            image_id = plot._draw_graph(shape=constants['shape'])

        if constants.get('make_movie', 'no').lower() == 'yes':
            mov_filename = visualizer.animate_trajectory()
            mov_id = mov_filename

        simulations.append((simulation, image_id, mov_id, merged_events))
    return simulations

def main():
    start_time = time.time()
    version = get_program_version()
    db_path = 'Trajectory.db'
    conn = sqlite3.connect(db_path)
    setup_database(conn)
    exp_id = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # --- 1回目のGUIを最前面で表示 ---
    selected_data = show_selection_ui()

    # もし２回目以降も選択画面を繰り返したい場合は、同じように呼ぶだけでOK
    # selected_data2 = show_selection_ui()

    volumes_list      = selected_data.get('volumes', [])
    sperm_conc_list   = selected_data.get('sperm_concentrations', [])
    shapes_list       = selected_data.get('shapes', [])
    spot_angle        = float(selected_data.get('spot_angle', 70))
    vsl               = float(selected_data.get('vsl', 0.13))
    deviation         = float(selected_data.get('deviation', 0.4))
    sampl_rate_hz     = float(selected_data.get('sampl_rate_hz', 2))
    sim_min           = float(selected_data.get('sim_min', 10))
    gamete_r          = float(selected_data.get('gamete_r', 0.1))
    stick_sec         = int(selected_data.get('stick_sec', 2))
    egg_localization  = selected_data.get('egg_localization', 'bottom_center')
    initial_direction = selected_data.get('initial_direction', 'right')
    initial_stick = int(selected_data.get('initial_stick', 10))
    seed_number       = selected_data.get('seed_number', None)
    n_repeat          = int(selected_data.get('n_repeat', 1))
    
    draw_trajectory   = 'yes' if 'graph' in selected_data.get('outputs', []) else 'no'
    make_movie        = 'yes' if 'movie' in selected_data.get('outputs', []) else 'no'

    if seed_number and seed_number.lower() != 'none':
        np.random.seed(int(seed_number))

    for shape in shapes_list:
        for volume in volumes_list:
            for sperm_conc in sperm_conc_list:
                constants = get_constants_from_gui(selected_data, shape, volume, sperm_conc)

                if shape == 'cube':
                    edge_length = constants['volume'] ** (1 / 3)
                    half_edge = edge_length / 2
                    constants.update({
                        'x_min': -half_edge, 'x_max': half_edge,
                        'y_min': -half_edge, 'y_max': half_edge,
                        'z_min': -half_edge, 'z_max': half_edge,
                        'radius': 0
                    })
                elif shape == 'drop':
                    drop_R = (constants['volume'] * 3 / (4 * np.pi)) ** (1 / 3)
                    constants.update({
                        'drop_R': drop_R,
                        'radius': drop_R,
                        'z_min': -drop_R, 'z_max': drop_R
                    })
                elif shape == 'spot':
                    if 'spot_R' not in constants or constants['spot_R'] is None:
                        compute_spot_parameters(constants)
                elif shape == 'ceros':
                    # CEROS 視野の座標範囲を定義　★ここを追加★
                    constants.update({
                        'ceros_x_min': -8.15,  'ceros_x_max':  8.15,
                        'ceros_y_min': -6.20,  'ceros_y_max':  6.20,
                        'ceros_z_min': -0.05,  'ceros_z_max':  0.05,
                        'radius'     : 0       # 3D 球切断などに使わないので 0 で可
                    })


                constants['number_of_sperm'] = constants['sperm_conc'] * constants['volume'] / 1000
                constants.update({
                    'spot_angle_rad': np.deg2rad(constants['spot_angle']),
                    'egg_volume': 4 * np.pi * constants['gamete_R']**3 / 3,
                    'stick_steps': constants['stick_sec'] * constants['sampl_rate_Hz'],
                    'inner_angle': 2 * np.pi / 70,
                })
                constants['n_simulation'] = int(constants['sim_min'] * 60 * constants['sampl_rate_Hz'])

                simulations = repeat_simulation(constants, n_repeat)

                for i, (sim, image_id, mov_id, merged_events) in enumerate(simulations, start=1):
                    contact_count_merged = len(merged_events)
                    simulation_id = insert_sim_record(conn, exp_id, version, constants, image_id, mov_id, contact_count_merged)
                    insert_intersection_records(conn, simulation_id, merged_events)

    aggregate_results(conn, exp_id)

    df_summary = pd.read_sql_query("SELECT * FROM summary", conn)
    if not df_summary.empty:
        df_summary = calculate_n_sperm(df_summary)
        df_summary.to_sql("summary", conn, if_exists="replace", index=False)
    else:
        print("summary テーブルに集計結果がありません。")

    conn.close()
    print(f"実行時間: {time.time() - start_time:.2f}秒")

if __name__ == "__main__":
    main()
