
# 完全統合版 plot_utils.py（Masaru仕様準拠：figもmovieもfigs_&_moviesに保存）
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

def plot_2d_trajectories(trajs_um, constants, save_path=None, show=True, max_sperm=None):
    x_min, x_max = constants["x_min"], constants["x_max"]
    y_min, y_max = constants["y_min"], constants["y_max"]
    z_min, z_max = constants["z_min"], constants["z_max"]
    trajs_mm = trajs_um.astype(float) / 1000.0
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    shape = constants.get('shape', 'cube').lower()
    if shape == "drop":
        r = constants["drop_r"]
        axs[0].add_patch(Circle((0.0, 0.0), r, color="pink", alpha=0.3))
        axs[1].add_patch(Circle((0.0, 0.0), r, color="pink", alpha=0.3))
        axs[2].add_patch(Circle((0.0, 0.0), r, color="pink", alpha=0.3))
    elif shape == "spot":
        R = constants["spot_r"]
        b_r = constants["spot_bottom_r"]
        b_h = constants["spot_bottom_height"]
        x_vals = np.linspace(-b_r, b_r, 200)
        z_vals = np.sqrt(np.clip(R**2 - x_vals**2, 0.0, None))
        axs[0].add_patch(Circle((0.0, 0.0), b_r, color="pink", alpha=0.3))
        axs[1].fill_between(x_vals, b_h, z_vals, color="pink", alpha=0.3)
        axs[2].fill_between(x_vals, b_h, z_vals, color="pink", alpha=0.3)
    egg_center = constants["egg_center"]
    egg_r = constants["gamete_r"]
    axs[0].add_patch(Circle((egg_center[0], egg_center[1]), egg_r, facecolor="yellow", alpha=0.6, edgecolor="gray"))
    axs[1].add_patch(Circle((egg_center[0], egg_center[2]), egg_r, facecolor="yellow", alpha=0.6, edgecolor="gray"))
    axs[2].add_patch(Circle((egg_center[1], egg_center[2]), egg_r, facecolor="yellow", alpha=0.6, edgecolor="gray"))
    if max_sperm is None:
        max_sperm = trajs_mm.shape[0]
    for s in range(min(max_sperm, trajs_mm.shape[0])):
        axs[0].plot(trajs_mm[s, :, 0], trajs_mm[s, :, 1], linewidth=0.6)
        axs[1].plot(trajs_mm[s, :, 0], trajs_mm[s, :, 2], linewidth=0.6)
        axs[2].plot(trajs_mm[s, :, 1], trajs_mm[s, :, 2], linewidth=0.6)
    for ax, title, xlab, ylab, xlim, ylim in zip(
        axs,
        ["XY (mm)", "XZ (mm)", "YZ (mm)"],
        ["X (mm)", "X (mm)", "Y (mm)"],
        ["Y (mm)", "Z (mm)", "Z (mm)"],
        [(x_min, x_max), (x_min, x_max), (y_min, y_max)],
        [(y_min, y_max), (z_min, z_max), (z_min, z_max)]
    ):
        ax.set_title(title)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    plt.show()

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
    plt.show()
    plt.close(fig)
