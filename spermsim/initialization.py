import os
import sys
import tkinter as tk
from tkinter import ttk
import configparser
import numpy as np
import math
import time
from pathlib import Path
from tools.ini_handler import save_config, load_config
from tools.derived_constants import calculate_derived_constants
from core.engine import SpermSimulation


# プロジェクトのルート（trajectory_reboot）を sys.path に追加
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "sperm_config.ini")

# 保存時のキー順（GUI 表示順と合わせる）
PARAM_ORDER = [
    "shape", "spot_angle", "vol", "sperm_conc", "vsl", "deviation",
    "surface_time", "egg_localization", "gamete_r", "sim_min",
    "sample_rate_hz", "seed_number", "sim_repeat", "display_mode",
]

# デフォルト設定
PARAM_ORDER = [
    "shape", "spot_angle", "vol", "sperm_conc", "vsl", "deviation",
    "surface_time", "egg_localization", "gamete_r", "sim_min",
    "sample_rate_hz", "seed_number", "sim_repeat", "display_mode",
]
# デフォルト設定
default_values = {
    "shape": "cube",
    "spot_angle": 50.0,
    "vol": 6.25,               # µL
    "sperm_conc": 10_000.0,    # cells/mL
    "vsl": 0.13,               # mm/s
    "deviation": 0.4,
    "surface_time": 2.0,
    "egg_localization": "bottom_center",
    "gamete_r": 0.04,          # mm
    "sim_min": 1.0,            # min 実測値ではなく「分」→秒に変換は simulation 側
    "sample_rate_hz": 4.0,
    "seed_number": "None",
    "sim_repeat": 1,
    "display_mode": ["2D"],    # 文字列リスト
}

# seed_number の文字列入力を整数シード値に変換
def get_seed(seed_input: str) -> int:
    if seed_input == "None":
        return int(time.time() * 1000) % (2**32)
    return int(seed_input)
import configparser
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "sperm_config.ini")

PARAM_ORDER = [
    "shape", "spot_angle", "vol", "sperm_conc", "vsl", "deviation",
    "surface_time", "egg_localization", "gamete_r", "sim_min",
    "sample_rate_hz", "seed_number", "sim_repeat", "display_mode"
]

def save_config(values: dict) -> None:
    cfg = configparser.ConfigParser()
    cfg["parameters"] = {}

    for k in PARAM_ORDER:
        if k in values:
            v = values[k]
            if k == "gamete_r":
                v = float(v)
            if k == "display_mode":
                if isinstance(v, (list, tuple)):
                    v = v[0]
                v = str(v).strip()
            cfg["parameters"][k] = str(v)

    # ✅ ここが最重要：実際にファイルへ保存する
    with open(CONFIG_PATH, "w") as f:
        cfg.write(f)
    print(f"[DEBUG] sperm_config.ini に保存しました → {CONFIG_PATH}")



import math
import numpy as np

def calculate_derived_constants(constants: dict) -> dict:
    # --- 単位変換（mm化） ---
    constants["gamete_r"] = float(constants["gamete_r"])
    constants["vol"] = float(constants["vol"])
    constants["vsl"] = float(constants["vsl"])
    constants["sample_rate_hz"] = float(constants["sample_rate_hz"])
    constants["sperm_conc"] = float(constants["sperm_conc"])
    constants["spot_angle"] = float(constants.get("spot_angle", 0.0))  # 必須shapeなら上書きされる

    # --- 基本パラメータ ---
    shape = constants["shape"]
    egg_localization = constants["egg_localization"]
    gamete_r_mm = constants["gamete_r"]

    # --- shape別の派生定数 ---
    if shape == "drop":
        drop_r = ((3.0 * constants["vol"]) / (4.0 * math.pi)) ** (1.0 / 3.0)
        constants["drop_r"] = drop_r
        constants.update(
            x_min=-drop_r, x_max=drop_r,
            y_min=-drop_r, y_max=drop_r,
            z_min=-drop_r, z_max=drop_r
        )
        if egg_localization == "center":
            egg_center = np.array([0.0, 0.0, 0.0])
        elif egg_localization == "bottom_center":
            egg_center = np.array([0.0, 0.0, -drop_r + gamete_r_mm])
        else:
            raise ValueError(f"Unsupported egg_localization for drop: {egg_localization}")

    elif shape == "spot":
        angle = constants["spot_angle"]
        spot_r, bottom_r, bottom_h = calc_spot_geometry(constants["vol"], angle)
        constants["spot_r"] = spot_r
        constants["spot_bottom_r"] = bottom_r
        constants["spot_bottom_height"] = bottom_h
        constants.update(
            x_min=-bottom_r, x_max=bottom_r,
            y_min=-bottom_r, y_max=bottom_r,
            z_min=bottom_h, z_max=spot_r
        )
        if egg_localization == "center":
            z_mid = (constants["z_min"] + constants["z_max"]) / 2
            egg_center = np.array([0.0, 0.0, z_mid])
        elif egg_localization == "bottom_center":
            egg_center = np.array([0.0, 0.0, bottom_h + gamete_r_mm])
        elif egg_localization == "bottom_edge":
            x_edge = math.sqrt(4 * spot_r * gamete_r_mm)
            egg_center = np.array([x_edge, 0.0, bottom_h + gamete_r_mm])
        else:
            raise ValueError(f"Unsupported egg_localization for spot: {egg_localization}")

    elif shape == "cube":
        edge = constants["vol"] ** (1.0 / 3.0)
        constants["edge"] = edge
        half = edge / 2
        constants.update(
            x_min=-half, x_max=half,
            y_min=-half, y_max=half,
            z_min=0.0,   z_max=edge
        )
        if egg_localization == "center":
            egg_center = np.array([0.0, 0.0, 0.0])
        elif egg_localization == "bottom_center":
            egg_center = np.array([0.0, 0.0, gamete_r_mm])
        elif egg_localization == "bottom_edge":
            egg_center = np.array([0.0, -half + gamete_r_mm, gamete_r_mm])
        else:
            raise ValueError(f"Unsupported egg_localization for cube: {egg_localization}")

    else:
        raise ValueError(f"Unknown shape: {shape}")

    # --- 派生定数 ---
    constants["egg_center"] = egg_center
    constants["limit"] = 1e-9
    constants["step_length"] = constants["vsl"] / constants["sample_rate_hz"] if constants["sample_rate_hz"] else 0.0
    constants["number_of_sperm"] = int(constants["sperm_conc"] * constants["vol"] / 1000)

    return constants



def calc_spot_geometry(volume_ul: float, angle_deg: float) -> tuple[float, float, float]:
    angle_rad = math.radians(angle_deg)
    vol_mm3 = volume_ul  # 1 µL = 1 mm³

    def cap_volume(R: float) -> float:
        h = R * (1 - math.cos(angle_rad))
        return math.pi * h * h * (3 * R - h) / 3

    low = 0.0
    high = max(vol_mm3 ** (1 / 3), 1.0)
    while cap_volume(high) < vol_mm3:
        high *= 2.0
    for _ in range(60):
        mid = (low + high) / 2.0
        if cap_volume(mid) < vol_mm3:
            low = mid
        else:
            high = mid
    R = (low + high) / 2.0
    bottom_r = R * math.sin(angle_rad)
    bottom_height = R * math.cos(angle_rad)
    return R, bottom_r, bottom_height
# ---------------------------------------------------------------------------
# Tkinter GUI クラス
# ---------------------------------------------------------------------------

class SimApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Sperm Simulation GUI")
        self.root.geometry("780x900")
        self.config_data = load_config()  # .ini → dict
        self.tk_vars: dict[str, tk.Variable] = {}  # Param ↔ Tk 変数
        self.save_var = tk.BooleanVar()
        self.save_var.set(True)
        self.simulation_ran = False
        # スクロールキャンバス
        canvas = tk.Canvas(self.root)
        vbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)

        
        self.scroll_frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=vbar.set)
        self.scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.pack(side="left", fill="both", expand=True)
        vbar.pack(side="right", fill="y")

        # 各ウィジェットを配置
        self._create_widgets(self.scroll_frame)
        self._restore_from_ini()  # 値を復元
        ttk.Checkbutton

        print("[DEBUG] 読み込んだ設定内容（config_data）:")
        for k, v in self.config_data.items():
            print(f"  {k} = {v}")
    # ---------------------------------------------------------------------
    # ウィジェット生成
    # ---------------------------------------------------------------------
    def _restore_from_ini(self, ini_path="sperm_config.ini"):
        config_data = load_config(ini_path)
        if config_data is None:
            return
        for key, value in config_data.items():
            if key not in self.tk_vars:
                continue
            self.tk_vars[key].set(value)
    # def _load_config(self, filename: str) -> dict:
    #     import configparser
    #     config = configparser.ConfigParser()
    #     config.read(filename)

    #     config_dict = {}
    #     if 'DEFAULT' in config:
    #         for key in config['DEFAULT']:
    #             config_dict[key] = config['DEFAULT'][key]

    #     return config_dict

    # def _save_config(self, config_dict: dict, filename: str):
    #     import configparser
    #     config = configparser.ConfigParser()
    #     config["DEFAULT"] = {key: str(value) for key, value in config_dict.items()}
    #     with open(filename, 'w') as configfile:
    #         config.write(configfile)


    def _create_widgets(self, parent: ttk.Frame) -> None:
        # --- shape -------------------------------------------------------
        self.tk_vars["shape"] = tk.StringVar()
        ttk.Label(parent, text="shape:").pack(anchor="w", padx=10, pady=(10, 0))
        f_shape = ttk.Frame(parent); f_shape.pack(anchor="w", padx=30)
        for v in ["cube", "drop", "spot", "ceros"]:
            ttk.Radiobutton(f_shape, text=v, variable=self.tk_vars["shape"],
                            value=v, command=self._update_spot_angle_state
                            ).pack(side="left")

        # --- spot_angle --------------------------------------------------
        self.tk_vars["spot_angle"] = tk.DoubleVar()
        ttk.Label(parent, text="spot_angle (deg):").pack(anchor="w", padx=10, pady=(10, 0))
        f_angle = ttk.Frame(parent); f_angle.pack(anchor="w", padx=30)
        self.spot_angle_buttons = []
        for v in [30, 50, 70, 90]:
            rb = ttk.Radiobutton(f_angle, text=str(v),
                                 variable=self.tk_vars["spot_angle"], value=float(v))
            rb.pack(side="left")
            self.spot_angle_buttons.append(rb)

        # --- vol ---------------------------------------------------------
        self.tk_vars["vol"] = tk.DoubleVar()
        ttk.Label(parent, text="vol (µL):").pack(anchor="w", padx=10, pady=(10, 0))
        f_vol = ttk.Frame(parent); f_vol.pack(anchor="w", padx=30)
        for v in [6.25, 12.5, 25, 50, 100, 200]:
            ttk.Radiobutton(f_vol, text=str(v), variable=self.tk_vars["vol"],
                            value=float(v)).pack(side="left")

        # --- sperm_conc --------------------------------------------------
        self.tk_vars["sperm_conc"] = tk.DoubleVar()
        ttk.Label(parent, text="sperm_conc (cells/mL):").pack(anchor="w", padx=10, pady=(10, 0))
        f_conc = ttk.Frame(parent); f_conc.pack(anchor="w", padx=30)
        for v in [1e3, 3.16e3, 1e4, 3.162e4, 1e5]:
            ttk.Radiobutton(f_conc, text=f"{int(v):,}", variable=self.tk_vars["sperm_conc"],
                            value=float(v)).pack(side="left")

        # --- vsl ---------------------------------------------------------
        self.tk_vars["vsl"] = tk.DoubleVar()
        ttk.Label(parent, text="vsl (mm/s):").pack(anchor="w", padx=10, pady=(10, 0))
        f_vsl = ttk.Frame(parent); f_vsl.pack(anchor="w", padx=30)
        for v in [0.073, 0.09, 0.11, 0.13, 0.15]:
            ttk.Radiobutton(f_vsl, text=str(v), variable=self.tk_vars["vsl"],
                            value=float(v)).pack(side="left")

        # --- deviation ---------------------------------------------------
        self.tk_vars["deviation"] = tk.DoubleVar()
        ttk.Label(parent, text="deviation:").pack(anchor="w", padx=10, pady=(10, 0))
        f_dev = ttk.Frame(parent); f_dev.pack(anchor="w", padx=30)
        for v in [0.1, 0.2, 0.4, 0.8, 1.0, 2.0, 4.0]:
            ttk.Radiobutton(f_dev, text=str(v), variable=self.tk_vars["deviation"],
                            value=float(v)).pack(side="left")

        # --- surface_time ------------------------------------------------
        self.tk_vars["surface_time"] = tk.DoubleVar()
        ttk.Label(parent, text="surface_time (sec):").pack(anchor="w", padx=10, pady=(10, 0))
        f_surface = ttk.Frame(parent); f_surface.pack(anchor="w", padx=30)
        for v in [0, 1, 2, 3, 4]:
            ttk.Radiobutton(f_surface, text=str(v), variable=self.tk_vars["surface_time"],
                            value=float(v)).pack(side="left")

        # --- egg_localization -------------------------------------------
        self.tk_vars["egg_localization"] = tk.StringVar()
        ttk.Label(parent, text="egg_localization:").pack(anchor="w", padx=10, pady=(10, 0))
        f_egg = ttk.Frame(parent); f_egg.pack(anchor="w", padx=30)
        for v in ["bottom_center", "bottom_edge", "center"]:
            ttk.Radiobutton(f_egg, text=v, variable=self.tk_vars["egg_localization"],
                            value=v).pack(side="left")

        # --- gamete_r ----------------------------------------------------
        self.tk_vars["gamete_r"] = tk.DoubleVar()
        ttk.Label(parent, text="gamete_r (mm):").pack(anchor="w", padx=10, pady=(10, 0))
        f_gr = ttk.Frame(parent); f_gr.pack(anchor="w", padx=30)
        for v in [0.04, 0.05, 0.15]:
            ttk.Radiobutton(f_gr, text=str(v), variable=self.tk_vars["gamete_r"],
                            value=float(v)).pack(side="left")

        # --- sim_min -----------------------------------------------------
        self.tk_vars["sim_min"] = tk.DoubleVar()
        ttk.Label(parent, text="sim_min (min):").pack(anchor="w", padx=10, pady=(10, 0))
        f_simmin = ttk.Frame(parent); f_simmin.pack(anchor="w", padx=30)
        for v in [0.2, 1, 10, 30, 60]:
            ttk.Radiobutton(f_simmin, text=str(v), variable=self.tk_vars["sim_min"],
                            value=float(v)).pack(side="left")

        # --- sample_rate_hz ----------------------------------------------
        self.tk_vars["sample_rate_hz"] = tk.DoubleVar()
        ttk.Label(parent, text="sample_rate_hz:").pack(anchor="w", padx=10, pady=(10, 0))
        f_hz = ttk.Frame(parent); f_hz.pack(anchor="w", padx=30)
        for v in [1, 2, 3, 4]:
            ttk.Radiobutton(f_hz, text=str(v), variable=self.tk_vars["sample_rate_hz"],
                            value=float(v)).pack(side="left")

        # --- seed_number ----------------------------------------------
        self.tk_vars["seed_number"] = tk.StringVar()
        ttk.Label(parent, text="seed_number:").pack(anchor="w", padx=10, pady=(10, 0))
        f_seed = ttk.Frame(parent); f_seed.pack(anchor="w", padx=30)
        for v in ["None", "0", "1", "2", "3", "4"]:
            ttk.Radiobutton(f_seed, text=v, variable=self.tk_vars["seed_number"],
                            value=v).pack(side="left")

        # --- sim_repeat --------------------------------------------------
        self.tk_vars["sim_repeat"] = tk.IntVar()
        ttk.Label(parent, text="sim_repeat:").pack(anchor="w", padx=10, pady=(10, 0))
        f_repeat = ttk.Frame(parent); f_repeat.pack(anchor="w", padx=30)
        for v in [1, 3, 10, 30]:
            ttk.Radiobutton(f_repeat, text=str(v), variable=self.tk_vars["sim_repeat"],
                            value=int(v)).pack(side="left")

        # --- display_mode ラジオボタン ------------------------------
        # ini ファイルからの正しい読み込み方法
        display_mode = self.config_data.get('display_mode', '2D')

        # 万が一、タプルになっている場合を考えて厳密に処理
        if isinstance(display_mode, tuple):
            display_mode = display_mode[0]
        elif isinstance(display_mode, list):
            display_mode = display_mode[0]

        # 念のために文字列変換と空白削除
        display_mode = str(display_mode).strip().replace('(', '').replace(')', '').replace(',', '').replace("'", "").replace('"', '')

        display_mode = self.config_data.get('display_mode', '2D')

        # ✅ リストやタプルだった場合に文字列に変換
        if isinstance(display_mode, (list, tuple)):
            display_mode = display_mode[0]

        display_mode = str(display_mode).strip().lower()  # "movie" や "3d" の整形
        self.tk_vars["display_mode"] = tk.StringVar(value=display_mode)


        ttk.Label(parent, text="display_mode:").pack(anchor="w", padx=10, pady=(10, 0))
        f_disp = ttk.Frame(parent)
        f_disp.pack(anchor="w", padx=30)

        for v in ["2D", "3D", "movie"]:
            ttk.Radiobutton(
                f_disp, text=v, variable=self.tk_vars["display_mode"], value=v
            ).pack(side="left")

        # 最終確認のプリント文
        print(f"最終修正後の表示モード: {self.tk_vars['display_mode'].get()}")

        # --- 実行ボタン --------------------------------------------------
        ttk.Button(parent, text="Save settings and run simulation",
                   command=self._on_save).pack(pady=20)

        self._update_spot_angle_state()
    # ---------------------------------------------------------------------
    # spot_angle の有効／無効
    # ---------------------------------------------------------------------
    def _update_spot_angle_state(self) -> None:
        enable = (self.tk_vars["shape"].get() == "spot")
        state = "normal" if enable else "disabled"
        for rb in self.spot_angle_buttons:
            rb.config(state=state)
    # ---------------------------------------------------------------------
    # 保存＆シミュレーション実行
    # ---------------------------------------------------------------------
    

    def _on_save(self):
        from tools.derived_constants import calculate_derived_constants
        from tools.ini_handler import save_config

        # GUIフォームから値を取得
        constants = {key: var.get() for key, var in self.tk_vars.items()}
        print("[DEBUGggg] GUIから取得した display_mode =", constants.get("display_mode"))
        # --- 数値変換 ---
        constants["gamete_r"] = float(constants["gamete_r"])
        constants["vol"] = float(constants["vol"])
        constants["vsl"] = float(constants["vsl"])
        constants["deviation"] = float(constants["deviation"])
        constants["surface_time"] = float(constants["surface_time"])
        constants["sperm_conc"] = float(constants["sperm_conc"])
        constants["spot_angle"] = float(constants["spot_angle"])
        constants["sample_rate_hz"] = float(constants["sample_rate_hz"])
        constants["sim_min"] = float(constants["sim_min"])
        constants["sim_repeat"] = int(constants["sim_repeat"])
        constants["seed_number"] = int(constants["seed_number"]) if constants["seed_number"] != "None" else "None"

        # --- display_mode を正規化（例: ["Movie"] → "Movie"） ---
        raw_mode = constants.get("display_mode", "2D")
        while isinstance(raw_mode, list) or isinstance(raw_mode, tuple):
            raw_mode = raw_mode[0]
        constants["display_mode"] = str(raw_mode).strip("()'\" ").lower()


        # --- 派生変数計算 ---
        constants = calculate_derived_constants(constants)

        # --- 設定を保存 ---
        save_config(constants)
        print("[SimApp] 設定を sperm_config.ini に保存しました")
        print(f"[DEBUG] 保存された display_mode: {constants['display_mode']}")

        # --- GUI終了 ---
        self.root.destroy()


# ---------------------------------------------------------------------------
# エントリーポイント
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("[DEBUG] starting SimApp ...")
    root = tk.Tk()
    app = SimApp(root)
    root.mainloop()

    app.mainloop()