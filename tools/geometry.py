import numpy as np
from numpy import linalg as LA
from typing import Dict      # ★ この行を追加
from typing import Tuple
from tools.enums import IOStatus as EnumIOStatus
from tools.io_checks import IO_check_spot

def bend_along_sphere_surface(vec: np.ndarray, normal: np.ndarray, angle_rad: float) -> np.ndarray:
    """
    接線ベクトル `vec` を、球面の法線 `normal` に沿って `angle_rad` ラジアンだけ
    内側（球の中心方向）に曲げた新しい単位ベクトルを返す。

    Parameters
    ----------
    vec : np.ndarray
        現在の進行方向ベクトル（正規化されていなくてもOK）
    normal : np.ndarray
        球面の法線ベクトル（原点中心 → 接触点方向）※正規化されていることを推奨
    angle_rad : float
        接線から法線に向かって回転させる角度（ラジアン）

    Returns
    -------
    np.ndarray
        曲げた後の正規化済み方向ベクトル
    """
    # vec と normal で張る平面内で回転
    # tangent: vec から normal 成分を除いた接線ベクトル
    tangent = vec - np.dot(vec, normal) * normal
    tangent /= np.linalg.norm(tangent) + 1e-12

    # 接線と法線ベクトルの間で、angle_radだけ回転（内側方向に）
    new_vec = (
        np.cos(angle_rad) * tangent - np.sin(angle_rad) * normal
    )

    return new_vec / (np.linalg.norm(new_vec) + 1e-12)

def _line_sphere_intersection(p0: np.ndarray, p1: np.ndarray, radius: float) -> Tuple[np.ndarray, float]:
    """
    線分 p0 → p1 が原点中心の球（半径 radius）と交差する場合、
    交点と残りの移動距離を返す。交差しない場合は p0 を返す。
    """
    d = p1 - p0
    d_norm = np.linalg.norm(d) + 1e-12
    d_unit = d / d_norm
    a = np.dot(d_unit, d_unit)
    b = 2 * np.dot(p0, d_unit)
    c = np.dot(p0, p0) - radius ** 2
    discriminant = b ** 2 - 4 * a * c

    if discriminant < 0:
        return p0, 0.0  # 球と交差しない

    t = (-b - np.sqrt(discriminant)) / (2 * a)
    t = max(0.0, min(t, d_norm))  # 線分内のみに制限
    intersect = p0 + d_unit * t
    remain = d_norm - t
    return intersect, remain

def _handle_drop_outside(
    vec: np.ndarray,
    base_pos: np.ndarray,
    constants: dict,
    surface_time: float,
    sample_rate_hz: int,
    stick_status: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    drop形状でoutsideと判定された場合に、球面上に沿って曲げて這わせながらinsideに戻す処理。
    stick_status の管理も行う。

    Returns
    -------
    vec : np.ndarray
        修正後のベクトル
    base_pos : np.ndarray
        曲げた後の新しい開始位置（通常は交点）
    stick_status : int
        更新後の stick_status 値
    """

    angle_rad = 2 * np.pi / 70  # 70角形分の角度
    max_iterations = 100  # 無限ループ防止
    iteration = 0

    while iteration < max_iterations:
        intersect, remain = _line_sphere_intersection(
            base_pos, base_pos + vec, constants["drop_r"]
        )
        normal = intersect / (np.linalg.norm(intersect) + 1e-12)

        vec = bend_along_sphere_surface(vec, normal, angle_rad)
        base_pos = intersect
        candidate = base_pos + vec * remain

        status = _io_check_drop(candidate, constants, base_pos)

        if status == "inside":
            break
        elif stick_status == 0:
            stick_status = int(surface_time * sample_rate_hz)
            break

        iteration += 1

    return vec, base_pos, stick_status


class BaseShape:
    def __init__(self, constants):
        self.constants = constants

    def get_limits(self):
        # ここで一括管理（すべてのShapeはこのまま継承）
        keys = ['x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max']
        return tuple(float(self.constants[k]) for k in keys)

    def io_check(self, *args, **kwargs):
        raise NotImplementedError

    def initial_position(self):
        raise NotImplementedError


class CubeShape(BaseShape):
    """
    立方体形状
    - constants に
        • "vol"      : 体積 [µL]          (従来)
        • "vol_um3"  : 体積 [µm³] ←★追加
      のどちらかが入っていれば初期化できます。
    - 計算した一辺長 edge_um, limits は self.constants に追記。
    """

    def __init__(self, constants: Dict[str, float]):
        super().__init__(constants)

        # ------------------ 追加：µm³ 指定をサポート ------------------
        if "vol_um3" in self.constants and "vol" not in self.constants:
            # µm³  →  µL  (1 µm³ = 1e-9 µL)
            self.constants["vol"] = float(self.constants["vol_um3"]) * 1e-9
        # -------------------------------------------------------------

        if "vol" not in self.constants:
            raise ValueError("CubeShape: constants に 'vol' か 'vol_um3' が必要です")

        # ------ 一辺長 edge_mm を計算 ------
        vol_mm3 = self.constants["vol"]              # µL = mm³
        edge_mm = vol_mm3 ** (1.0 / 3.0)             # 一辺 [mm]
        edge_um = edge_mm * 1000.0
        self.edge_um = edge_um
        self.constants["edge"] = edge_mm

        # --- limits の設定 ---
        # derived_constants.calculate_derived_constants で x_min などが
        # 既に mm 単位で計算されている場合はそれを尊重する。
        limit_keys = ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]
        if not all(k in self.constants for k in limit_keys):
            # 派生値が無い場合のみここで計算する（mm単位）
            half_mm = edge_mm / 2.0
            self.constants.update({
                "x_min": -half_mm, "x_max": half_mm,
                "y_min": -half_mm, "y_max": half_mm,
                "z_min": -half_mm, "z_max": half_mm,
            })
        # edge_um は常に保存しておく
        self.constants["edge_um"] = edge_um
        # -------------------------------------------------------------

    # ------------- 以降は Masaru さんの元コードをそのまま残す -------------
    def initial_position(self):
        x_min, x_max, y_min, y_max, z_min, z_max = self.get_limits()
        return np.random.uniform([x_min, y_min, z_min], [x_max, y_max, z_max])

    def io_check(self, point):
        x_min, x_max, y_min, y_max, z_min, z_max = self.get_limits()
        eps = 1e-9
        inside = (x_min < point[0] < x_max) and (y_min < point[1] < y_max) and (z_min < point[2] < z_max)
        if inside:
            return EnumIOStatus.INSIDE, None
        on_edge = (
            np.isclose([point[0]], [x_min, x_max], atol=eps).any() or
            np.isclose([point[1]], [y_min, y_max], atol=eps).any() or
            np.isclose([point[2]], [z_min, z_max], atol=eps).any()
        )
        if on_edge:
            return EnumIOStatus.TEMP_ON_EDGE, None
        return EnumIOStatus.OUTSIDE, None

class DropShape(BaseShape):
    def initial_position(self):
        R = float(self.constants['drop_r'])
        theta = np.arccos(2 * np.random.rand() - 1)
        phi = np.random.uniform(-np.pi, np.pi)
        s = R * np.random.rand() ** (1/3)
        x = s * np.sin(theta) * np.cos(phi)
        y = s * np.sin(theta) * np.sin(phi)
        z = s * np.cos(theta)
        return np.array([x, y, z])

    def io_check(self, point, stick_status):
        R = float(self.constants["drop_r"])
        if point[2] < 0:
            return EnumIOStatus.OUTSIDE
        norm = np.linalg.norm(point)
        if norm < R:
            return EnumIOStatus.INSIDE
        if np.isclose(norm, R, atol=1e-9):
            return EnumIOStatus.TEMP_ON_SURFACE
        return EnumIOStatus.OUTSIDE

class SpotShape(BaseShape):
    def initial_position(self):
        """Return a random point uniformly distributed inside a spherical cap."""
        radius = float(self.constants['spot_r'])
        angle_rad = np.deg2rad(float(self.constants['spot_angle']))
        cos_min = np.cos(angle_rad)

        while True:
            vec = np.random.normal(size=3)
            vec /= np.linalg.norm(vec) + 1e-12
            r = radius * (np.random.rand() ** (1 / 3))
            point = vec * r
            if point[2] >= radius * cos_min:
                return point

    def io_check(self, base_point, temp_point=None, prev_stat=None, stick_status=0):
        """Check I/O status using ``IO_check_spot`` for accurate boundary handling."""
        if temp_point is None:
            return EnumIOStatus.OUTSIDE

        # ``IO_check_spot`` performs detailed checks for the spherical cap,
        # including the bottom plane and rim.
        return IO_check_spot(
            base_point,
            temp_point,
            self.constants,
            prev_stat if prev_stat is not None else EnumIOStatus.INSIDE,
            stick_status,
        )

class CerosShape(BaseShape):
    def initial_position(self):
        x_min, x_max, y_min, y_max, z_min, z_max = self.get_limits()
        return np.random.uniform([x_min, y_min, z_min], [x_max, y_max, z_max])

    def io_check(self, point):
        # cerosもcubeと同じ判定
        x_min, x_max, y_min, y_max, z_min, z_max = self.get_limits()
        eps = 1e-9
        inside = (x_min < point[0] < x_max) and (y_min < point[1] < y_max) and (z_min < point[2] < z_max)
        if inside:
            return EnumIOStatus.INSIDE, None
        on_edge = (
            np.isclose([point[0]], [x_min, x_max], atol=eps).any() or
            np.isclose([point[1]], [y_min, y_max], atol=eps).any() or
            np.isclose([point[2]], [z_min, z_max], atol=eps).any()
        )
        if on_edge:
            return EnumIOStatus.TEMP_ON_EDGE, None
        return EnumIOStatus.OUTSIDE, None
