import numpy as np
import matplotlib.pyplot as plt

def draw_medium(ax, constants):
    shape = constants["shape"]
    if shape == "drop":
        r = constants["drop_r"]
        u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
        x = r * np.cos(u) * np.sin(v)
        y = r * np.sin(u) * np.sin(v)
        z = r * np.cos(v)
        ax.plot_surface(x, y, z, color='pink', alpha=0.2)
    if shape == "spot":
        r = constants["spot_r"]
        rb= constants["spot_bottom_r"]
        rh = constants["spot_bottom_height"]
        u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
        x = r * np.cos(u) * np.sin(v)
        y = r * np.sin(u) * np.sin(v)
        z = r * np.cos(v)
        ax.plot_surface(x, y, z, color='pink', alpha=0.2)
    if shape == "drop"
    "":
        r = constants["drop_r"]
        u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
        x = r * np.cos(u) * np.sin(v)
        y = r * np.sin(u) * np.sin(v)
        z = r * np.cos(v)
        ax.plot_surface(x, y, z, color='pink', alpha=0.2)
    # 他の shape（cube, spotなど）を描画したい場合はここに追記
    
def draw_egg_2d(ax, egg_center, egg_radius, plane='xy'):
    from matplotlib.patches import Circle 
    """
    2D平面上に卵子（円）を描画する

    Parameters:
        ax: matplotlib.axes.Axes
        egg_center: 卵子中心の3D座標 [x, y, z]
        egg_radius: 半径（mm単位）
        plane: 'xy', 'xz', 'yz' のいずれか（描画平面）
    """
    if plane == 'xy':
        cx, cy = egg_center[0], egg_center[1]
    elif plane == 'xz':
        cx, cy = egg_center[0], egg_center[2]
    elif plane == 'yz':
        cx, cy = egg_center[1], egg_center[2]
    else:
        raise ValueError(f"Unsupported plane: {plane}. Use 'xy', 'xz', or 'yz'.")

    egg_patch = Circle((cx, cy), egg_radius, facecolor='yellow', edgecolor='gray', alpha=0.6)
    ax.add_patch(egg_patch)
    

def draw_egg_3d(ax, egg_center, radius):
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = radius * np.cos(u) * np.sin(v) + egg_center[0]
    y = radius * np.sin(u) * np.sin(v) + egg_center[1]
    z = radius * np.cos(v) + egg_center[2]
    ax.plot_surface(x, y, z, color='gold', alpha=0.8)
