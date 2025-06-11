
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from datetime import datetime
from tools.path_config import ensure_save_dir

def debug_trajectory(trajectory):
    print(f"[DEBUG] trajectory.shape = {trajectory.shape}")
    if trajectory.shape[0] > 0 and trajectory.shape[1] > 0:
        print(f"[DEBUG] trajectory[0, :5, :] =\n{trajectory[0, :5, :]}")
    else:
        print("[DEBUG] trajectory is empty or has zero steps.")

def get_figure_save_path(filename: str) -> str:
    return os.path.join(ensure_save_dir(), filename)

def draw_medium_2d(ax, constants: dict, view: str):

    shape = constants.get("shape", "cube").lower()
    view = view.upper()

    if shape == "cube":
        x_min, x_max = constants["x_min"], constants["x_max"]
        y_min, y_max = constants["y_min"], constants["y_max"]
        z_min, z_max = constants["z_min"], constants["z_max"]

        from matplotlib.patches import Rectangle
        if view == "XY":
            ax.add_patch(Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, color='pink', alpha=0.3))
        elif view == "XZ":
            ax.add_patch(Rectangle((x_min, z_min), x_max - x_min, z_max - z_min, color='pink', alpha=0.3))
        elif view == "YZ":
            ax.add_patch(Rectangle((y_min, z_min), y_max - y_min, z_max - z_min, color='pink', alpha=0.3))

    elif shape == "drop":
        R = constants.get("drop_r", 1.0)

        if view == "XY":
            # xy平面 → 底面円
            theta = np.linspace(0, 2 * np.pi, 200)
            x = R * np.cos(theta)
            y = R * np.sin(theta)
            ax.fill(x, y, color='pink', alpha=0.3)

        elif view == "XZ":
            # xz断面 → 半円（上下対称に塗る）
            theta = np.linspace(0, 2 * np.pi, 200)
            x = R * np.cos(theta)
            z = R * np.sin(theta)
            ax.fill(np.concatenate([x, -x[::-1]]),
                    np.concatenate([z, z[::-1]]),
                    color='pink', alpha=0.3)

        elif view == "YZ":
            # yz断面 → 半円（y vs z）
            theta = np.linspace(0, 2 * np.pi, 200)
            y = R * np.cos(theta)
            z = R * np.sin(theta)
            ax.fill(np.concatenate([y, -y[::-1]]),
                    np.concatenate([z, z[::-1]]),
                    color='pink', alpha=0.3)


    elif shape == "spot":
        R = constants.get("spot_r", 1.0)
        z_base = constants.get("spot_bottom_height", 0.0)

        if view == "XY":
            # XY平面では底面の円（z = z_base）を投影
            theta = np.linspace(0, 2 * np.pi, 200)
            x = R * np.cos(theta)
            y = R * np.sin(theta)
            ax.fill(x, y, color='pink', alpha=0.3)  # ← 塗りつぶし！

        elif view == "XZ":
            # Z軸断面での半円を塗りつぶす
            cos_theta_min = z_base / R
            theta_min = np.arccos(np.clip(cos_theta_min, -1.0, 1.0))
            theta = np.linspace(0, theta_min, 200)
            x = R * np.sin(theta)
            z = R * np.cos(theta)
            ax.fill(np.concatenate([x, -x[::-1]]),
                    np.concatenate([z, z[::-1]]),
                    color='pink', alpha=0.3)

        elif view == "YZ":
            # YZ面での半円（XZのx→y）を塗りつぶす
            cos_theta_min = z_base / R
            theta_min = np.arccos(np.clip(cos_theta_min, -1.0, 1.0))
            theta = np.linspace(0, theta_min, 200)
            y = R * np.sin(theta)
            z = R * np.cos(theta)
            ax.fill(np.concatenate([y, -y[::-1]]),
                    np.concatenate([z, z[::-1]]),
                    color='pink', alpha=0.3)          
def draw_medium_3d(ax, constants: dict):
    shape = constants.get("shape", "cube").lower()

    if shape == "cube":
        # Cube の外枠を描画
        x_min, x_max = constants["x_min"], constants["x_max"]
        y_min, y_max = constants["y_min"], constants["y_max"]
        z_min, z_max = constants["z_min"], constants["z_max"]

        for s, e in zip(
            [(x_min, y_min, z_min), (x_max, y_min, z_min), (x_max, y_max, z_min), (x_min, y_max, z_min),
             (x_min, y_min, z_max), (x_max, y_min, z_max), (x_max, y_max, z_max), (x_min, y_max, z_max)],
            [(x_max, y_min, z_min), (x_max, y_max, z_min), (x_min, y_max, z_min), (x_min, y_min, z_max),
             (x_max, y_min, z_max), (x_max, y_max, z_max), (x_min, y_max, z_max), (x_min, y_min, z_max)]
        ):
            ax.plot([s[0], e[0]], [s[1], e[1]], [s[2], e[2]], color="pink", alpha=0.4)

    elif shape == "drop":
        print("draw medium called 3d")
        R = constants.get("drop_r", 1.0)
        u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
        x = R * np.cos(u) * np.sin(v)
        y = R * np.sin(u) * np.sin(v)
        z = R * np.cos(v)
        ax.plot_surface(x, y, z, color='pink', alpha=0.3, edgecolor="none")

    elif shape == "spot":
        R = constants.get("spot_r", 1.0)
        z_base = constants.get("spot_bottom_height", 0.0)
        cos_theta_min = z_base / R
        theta_min = np.arccos(np.clip(cos_theta_min, -1.0, 1.0))
        theta, phi = np.meshgrid(np.linspace(0, theta_min, 30), np.linspace(0, 2 * np.pi, 30))
        x = R * np.sin(theta) * np.cos(phi)
        y = R * np.sin(theta) * np.sin(phi)
        z = R * np.cos(theta)
        ax.plot_surface(x, y, z, color="pink", alpha=0.3, edgecolor="none")
def draw_egg_2d(ax, constants: dict, view: str = "xy"):
    """
    2D図中に卵子（球）を描画する。
    
    Parameters:
    - ax: matplotlib Axes
    - constants: シミュレーションパラメータ辞書（egg_center, gamete_r を含む）
    - view: "xy", "xz", "yz" のいずれか（どの平面で描画するか）
    """
    import matplotlib.patches as patches

    center = constants.get("egg_center", [0.0, 0.0, 0.0])
    r = constants.get("gamete_r", 0.05)

    if view == "xy":
        x, y = center[0], center[1]
    elif view == "xz":
        x, y = center[0], center[2]
    elif view == "yz":
        x, y = center[1], center[2]
    else:
        raise ValueError(f"Invalid view '{view}'. Must be 'xy', 'xz', or 'yz'.")

    egg = patches.Circle((x, y), r, color='gold', alpha=0.8, zorder=10)
    ax.add_patch(egg)
def draw_egg_3d(ax, egg_pos, radius, *, color="yellow", alpha=0.6):
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    x = radius * np.cos(u) * np.sin(v) + egg_pos[0]
    y = radius * np.sin(u) * np.sin(v) + egg_pos[1]
    z = radius * np.cos(v) + egg_pos[2]
    ax.plot_surface(x, y, z, color=color, alpha=alpha, edgecolor="none")

def plot_2d_trajectories(trajectory, constants, save_path=None, show=True, max_sperm=None):
    """
    trajectory: List[np.ndarray] or np.ndarray of shape (n_sperm, n_steps, 3)
    constants: dict
    """
    import numpy as np
    trajectory = np.array(trajectory).astype(float)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    shape = constants.get('shape', 'cube').lower()

    
    if max_sperm is None:
        max_sperm = trajectory.shape[0]

    # 背景メディウムと卵子を各投影に描画
    for ax, view in zip(axs, ["xy", "xz", "yz"]):
        draw_medium_2d(ax, constants, view=view)
        draw_egg_2d(ax, constants, view=view)

    # 軌跡描画
    for s in range(min(max_sperm, trajectory.shape[0])):
        axs[0].plot(trajectory[s, :, 0], trajectory[s, :, 1], linewidth=0.6)  # XY
        axs[1].plot(trajectory[s, :, 0], trajectory[s, :, 2], linewidth=0.6)  # XZ
        axs[2].plot(trajectory[s, :, 1], trajectory[s, :, 2], linewidth=0.6)  # YZ

    # 軸範囲とアスペクト比の設定
    x_min, x_max = constants["x_min"], constants["x_max"]
    y_min, y_max = constants["y_min"], constants["y_max"]
    z_min, z_max = constants["z_min"], constants["z_max"]

    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min

    axs[0].set_xlim(x_min, x_max)
    axs[0].set_ylim(y_min, y_max)
    axs[0].set_aspect(x_range / y_range)

    axs[1].set_xlim(x_min, x_max)
    axs[1].set_ylim(z_min, z_max)
    axs[1].set_aspect(x_range / z_range)

    axs[2].set_xlim(y_min, y_max)
    axs[2].set_ylim(z_min, z_max)
    axs[2].set_aspect(y_range / z_range)

    # 軸タイトルとグリッド
    for ax, title in zip(axs, ["XY (mm)", "XZ (mm)", "YZ (mm)"]):
        ax.set_title(title)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    # fig.tight_layout()

    if save_path:
        fig.savefig(save_path)
        try:
            import subprocess
            subprocess.run(["open", save_path])
        except Exception as e:
            print(f"[WARN] open failed: {e}")

    if show:
        plt.show()
    else:
        plt.close()
def plot_3d_trajectories(trajectory, constants, save_path=None, show=True):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from tools.plot_utils import draw_medium_3d, draw_egg_3d
    print("[DEBUG] plot_3d_trajectories に到達")
    print(f"[DEBUG] shape = {constants.get('shape')}")
    print(f"[DEBUG] egg_center = {constants.get('egg_center')}")
    print(f"[DEBUG] 軌跡数 = {len(trajectory)}")

    trajectory = np.array(trajectory)
    n_sperm, n_steps, _ = trajectory.shape

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    # ✅ 軸設定
    ax.set_xlim(constants["x_min"], constants["x_max"])
    ax.set_ylim(constants["y_min"], constants["y_max"])
    ax.set_zlim(constants["z_min"], constants["z_max"])
    ax.set_box_aspect([
        constants["x_max"] - constants["x_min"],
        constants["y_max"] - constants["y_min"],
        constants["z_max"] - constants["z_min"]
    ])

    # ✅ mediumを描画
    draw_medium_3d(ax, constants)

    # ✅ 卵子を描画
    egg_pos = constants.get("egg_center", [0.0, 0.0, 0.0])
    egg_r = constants.get("gamete_r", 0.05)
    draw_egg_3d(ax, egg_pos, egg_r)

    # ✅ 軌跡ライン
    lines = [ax.plot([], [], [], lw=1)[0] for _ in range(n_sperm)]

    def init():
        for line in lines:
            line.set_data([], [])
            line.set_3d_properties([])
        return lines

    def update(frame):
        for i, line in enumerate(lines):
            line.set_data(trajectory[i, :frame+1, 0], trajectory[i, :frame+1, 1])
            line.set_3d_properties(trajectory[i, :frame+1, 2])
        return lines

    ani = FuncAnimation(fig, update, init_func=init, frames=n_steps, interval=100, blit=False)

    if save_path:
        try:
            ani.save(save_path, writer="ffmpeg", fps=5)
            print(f"[INFO] 保存成功: {save_path}")
        except Exception as e:
            print(f"[WARN] ffmpeg失敗: {e} → pillow再試行")
            try:
                ani.save(save_path, writer="pillow", fps=5)
                print(f"[INFO] pillow成功: {save_path}")
            except Exception as e2:
                print(f"[ERROR] 保存失敗: {e2}")
    if show:
        plt.show()
    else:
        plt.close()
def draw_3d_movies(trajectory, constants, save_path=None, show=True):
    """
    軌跡とメディウム・卵子を3Dで描画・保存するアニメーション。
    - trajectory: (n_sperm, n_steps, 3) or List[np.ndarray]
    - constants: dict with keys like x_min, x_max, egg_center, gamete_r, etc.
    """
    print("[DEBUG] draw_3d_movies に到達")
    print(f"[DEBUG] 軌跡数: {len(trajectory)}")
    print(f"[DEBUG] 軸範囲: x=({constants['x_min']}, {constants['x_max']}), "
          f"y=({constants['y_min']}, {constants['y_max']}), "
          f"z=({constants['z_min']}, {constants['z_max']})")

    trajectory = np.array(trajectory).astype(float)
    n_sperm, n_steps, _ = trajectory.shape

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    # 軸設定
    ax.set_xlim(constants["x_min"], constants["x_max"])
    ax.set_ylim(constants["y_min"], constants["y_max"])
    ax.set_zlim(constants["z_min"], constants["z_max"])
    ax.set_box_aspect([
        constants["x_max"] - constants["x_min"],
        constants["y_max"] - constants["y_min"],
        constants["z_max"] - constants["z_min"]
    ])

    # メディウム描画
    draw_medium_3d(ax, constants)

    # 卵子描画
    egg_pos = constants.get("egg_center", [0.0, 0.0, 0.0])
    egg_r = constants.get("gamete_r", 0.05)
    draw_egg_3d(ax, egg_pos, egg_r)

    # ライン初期化
    lines = [ax.plot([], [], [], lw=1)[0] for _ in range(n_sperm)]

    def init():
        for line in lines:
            line.set_data([], [])
            line.set_3d_properties([])
        return lines

    def update(frame):
        for i, line in enumerate(lines):
            if frame < trajectory.shape[1]:
                line.set_data(trajectory[i, :frame+1, 0], trajectory[i, :frame+1, 1])
                line.set_3d_properties(trajectory[i, :frame+1, 2])
        return lines

    ani = FuncAnimation(fig, update, init_func=init, frames=n_steps, interval=100, blit=False)

    if save_path:
        ext = os.path.splitext(save_path)[-1].lower()
        try:
            if ext == ".mp4":
                ani.save(save_path, writer="ffmpeg", fps=5)
            elif ext == ".gif":
                ani.save(save_path, writer="pillow", fps=5)
            else:
                print(f"[WARN] 未対応拡張子 {ext} → GIFで保存")
                ani.save(save_path, writer="pillow", fps=5)
            print(f"[INFO] 保存成功: {save_path}")
        except Exception as e:
            print(f"[ERROR] 保存失敗: {e}")

    if show:
        plt.show()
    else:
        plt.close()