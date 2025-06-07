
# 統合版 plot_utils.py（2D・3D可視化およびムービー保存対応）
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from datetime import datetime
from tools.path_config import ensure_save_dir

def get_figure_save_path(filename: str) -> str:
    return os.path.join(ensure_save_dir(), filename)

def draw_medium(ax, constants: dict):
    shape = constants.get("shape", "cube").lower()
    if shape == "spot":
        R = constants.get("spot_r", 1.0)
        z_base = constants.get("spot_bottom_height", 0.0)
        cos_theta_min = z_base / R
        theta_min = np.arccos(np.clip(cos_theta_min, -1.0, 1.0))
        theta, phi = np.meshgrid(np.linspace(0, theta_min, 30), np.linspace(0, 2 * np.pi, 30))
        x = R * np.sin(theta) * np.cos(phi)
        y = R * np.sin(theta) * np.sin(phi)
        z = R * np.cos(theta)
        ax.plot_surface(x, y, z, color="pink", alpha=0.3, edgecolor="none")
    elif shape == "drop":
        R = constants.get("drop_r", 1.0)
        u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
        x = R * np.cos(u) * np.sin(v)
        y = R * np.sin(u) * np.sin(v)
        z = R * np.cos(v)
        ax.plot_surface(x, y, z, color="pink", alpha=0.3, edgecolor="none")
    elif shape == "cube":
        x_min, x_max = constants["x_min"], constants["x_max"]
        y_min, y_max = constants["y_min"], constants["y_max"]
        z_min, z_max = constants["z_min"], constants["z_max"]
        for s, e in zip(
            [(x_min, y_min, z_min), (x_max, y_min, z_min), (x_max, y_max, z_min),
             (x_min, y_max, z_min), (x_min, y_min, z_max), (x_max, y_min, z_max),
             (x_max, y_max, z_max), (x_min, y_max, z_max)],
            [(x_max, y_min, z_min), (x_max, y_max, z_min), (x_min, y_max, z_min),
             (x_min, y_min, z_max), (x_max, y_min, z_max), (x_max, y_max, z_max),
             (x_min, y_max, z_max), (x_min, y_min, z_max)]
        ):
            ax.plot([s[0], e[0]], [s[1], e[1]], [s[2], e[2]], color="gray", alpha=0.5)

def draw_egg_3d(ax, egg_pos, radius, *, color="yellow", alpha=0.6):
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    x = radius * np.cos(u) * np.sin(v) + egg_pos[0]
    y = radius * np.sin(u) * np.sin(v) + egg_pos[1]
    z = radius * np.cos(v) + egg_pos[2]
    ax.plot_surface(x, y, z, color=color, alpha=alpha, edgecolor="none")

def plot_3d_movie_trajectories(trajs: np.ndarray, vectors: np.ndarray, constants: dict,
                                save_path=None, show=True, format="mp4"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xlim = constants["x_min"], constants["x_max"]
    ylim = constants["y_min"], constants["y_max"]
    zlim = constants["z_min"], constants["z_max"]
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_box_aspect([xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0]])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Sperm Vectors (Fixed Length)")

    draw_medium(ax, constants)
    draw_egg_3d(ax, constants["egg_center"], constants.get("gamete_r", 0.05))

    num_sperm, num_frames = trajs.shape[0], trajs.shape[1]
    full_colors = plt.cm.tab20(np.linspace(0, 1, 20))
    colors = [full_colors[i] for i in range(20) if i != 3]

    quivers = [
        ax.quiver(0, 0, 0, 0, 0, 0, length=0.1, normalize=True,
                  arrow_length_ratio=0.9, linewidth=2.5, color=colors[i % len(colors)])
        for i in range(num_sperm)
    ]

    def update(frame):
        for i in range(num_sperm):
            x, y, z = trajs[i, frame]
            u, v, w = vectors[i, frame]
            quivers[i].remove()
            quivers[i] = ax.quiver(x, y, z, u, v, w, length=0.1, normalize=True,
                                   arrow_length_ratio=0.7, linewidth=2, color=colors[i % len(colors)])
        return quivers

    ani = FuncAnimation(fig, update, frames=num_frames, interval=100, blit=False)

    if not save_path:
        dtstr = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = "gif" if format == "gif" else "mp4"
        save_path = get_figure_save_path(f"movie_{dtstr}.{ext}")

    try:
        if format == "gif":
            ani.save(save_path, writer="pillow", fps=10)
        else:
            ani.save(save_path, fps=10)
        print(f"[INFO] 動画を保存しました: {save_path}")
    except Exception as e:
        print(f"[ERROR] 保存失敗: {e}")

    if show:
        plt.show()
    plt.close(fig)
