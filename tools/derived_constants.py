import numpy as np
import math

# ✅ インライン化した関数（仕様維持）
def calc_spot_geometry(vol_mm3: float, angle_deg: float):
    """
    spot（球冠）形状の体積と角度から、球の半径、底面半径、底面高さを返す。
    """
    theta = math.radians(angle_deg)
    h = (3 * vol_mm3 / (math.pi * (1 - math.cos(theta)) ** 2 * (2 + math.cos(theta)))) ** (1/3)
    R = h / (1 - math.cos(theta))
    r = R * math.sin(theta)
    return R, r, h

def calculate_derived_constants(raw_constants):
    constants = raw_constants.copy()
    shape = constants.get("shape", "cube").lower()
    constants["shape"] = shape
    egg_localization = constants.get("egg_localization", "center")
    vol = float(constants.get("vol", 0.0))  # μL = mm³
    sperm_conc = float(constants.get("sperm_conc", 0.0))  # ✅ これを必ず追加！

    # gamete_r is already provided in millimeters
    constants["gamete_r"] = float(constants["gamete_r"])

    if shape == "drop":
        r_mm = ((3.0 * vol) / (4.0 * math.pi)) ** (1.0 / 3.0)
        constants["drop_r"] = r_mm

    if shape == "spot":
        angle_deg = float(constants.get("spot_angle", 0.0))
        spot_r_mm, bottom_r_mm, bottom_h_mm = calc_spot_geometry(vol, angle_deg)
        constants["spot_r"] = spot_r_mm
        constants["spot_bottom_r"] = bottom_r_mm
        constants["spot_bottom_height"] = bottom_h_mm

    if shape == "cube":
        edge = vol ** (1.0 / 3.0)
        constants["edge"] = edge

    constants["step_length"] = float(constants["vsl"]) / float(constants["sample_rate_hz"]) / 1000



    # ✅ 空間範囲の設定
    if shape == "cube":
        half = constants["edge"] / 2
        constants.update(
            x_min=-half, x_max=half,
            y_min=-half, y_max=half,
            z_min=-half, z_max=half
        )
    elif shape == "drop":
        r = constants["drop_r"]
        constants.update(
            x_min=-r, x_max=r,
            y_min=-r, y_max=r,
            z_min=-r, z_max=r
        )
    elif shape == "spot":
        R = constants["spot_r"]
        b_r = constants["spot_bottom_r"]
        h = constants["spot_bottom_height"]
        constants.update(
            x_min=-b_r, x_max=b_r,
            y_min=-b_r, y_max=b_r,
            z_min=h,    z_max=R
        )

    # ✅ egg_center 計算（既存仕様維持）
    if shape == "cube":
        if egg_localization == "center":
            egg_center = np.array([0.0, 0.0, 0.0])
        elif egg_localization == "bottom_center":
            egg_center = np.array([0.0, 0.0, constants["z_min"] + constants["gamete_r"]])
        elif egg_localization == "bottom_edge":
            egg_center = np.array([0.0, constants["y_min"] + constants["gamete_r"], constants["z_min"] + constants["gamete_r"]])
        else:
            raise ValueError(f"Unsupported egg_localization for cube: {egg_localization}")

    elif shape == "drop":
        if egg_localization == "center":
            egg_center = np.array([0.0, 0.0, 0.0])
        elif egg_localization == "bottom_center":
            egg_center = np.array([0.0, 0.0, constants["z_min"] + constants["gamete_r"]])  # ← ✅
        else:
            raise ValueError(f"Unsupported egg_localization for drop: {egg_localization}")

    elif shape == "spot":
        if egg_localization == "center":
            z_mid = (constants["z_min"] + constants["z_max"]) / 2
            egg_center = np.array([0.0, 0.0, z_mid])
        elif egg_localization == "bottom_center":
            egg_center = np.array([0.0, 0.0, constants["z_min"] + constants["gamete_r"]])
        elif egg_localization == "bottom_edge":
            R = constants["spot_r"]
            r = constants["gamete_r"]
            x_edge = math.sqrt(4 * R * r)
            egg_center = np.array([x_edge, 0.0, constants["z_min"] + constants["gamete_r"]])
        else:
            raise ValueError(f"Unsupported egg_localization for spot: {egg_localization}")

    constants["egg_center"] = egg_center
    # ✅ number_of_sperm 計算を追加（テスト対応）
    number_of_sperm = int(round(vol * sperm_conc / 1000.0))
    constants["number_of_sperm"] = number_of_sperm
    constants["limit"] = 1e-9
    return constants

def calc_spot_geometry(volume_ul: float, angle_deg: float) -> tuple[float, float, float]:
    angle_rad = math.radians(angle_deg)
    vol_um3 = volume_ul * 1e9
    def cap_volume(R: float) -> float:
        h = R * (1 - math.cos(angle_rad))
        return math.pi * h * h * (3 * R - h) / 3
    low = 0.0
    high = max(vol_um3 ** (1 / 3), 1.0)
    while cap_volume(high) < vol_um3:
        high *= 2.0
    for _ in range(60):
        mid = (low + high) / 2.0
        if cap_volume(mid) < vol_um3:
            low = mid
        else:
            high = mid
    R_um = (low + high) / 2.0
    bottom_r_um = R_um * math.sin(angle_rad)
    bottom_height_um = R_um * math.cos(angle_rad)
    return R_um / 1000.0, bottom_r_um / 1000.0, bottom_height_um / 1000.0