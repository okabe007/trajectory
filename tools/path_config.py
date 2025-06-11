import os

# プロジェクトルート（tools/ から 1つ上 = trajectory/）
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 保存用ディレクトリ（figs_&_movies）
SAVE_DIR = os.path.join(PROJECT_ROOT, "figs_&_movies")

def ensure_save_dir() -> str:
    """
    保存先ディレクトリが存在しなければ作成し、そのパスを返す。
    """
    os.makedirs(SAVE_DIR, exist_ok=True)
    return SAVE_DIR


def open_file(path: str) -> None:
    """Open a file with the default application if possible."""
    import subprocess
    import platform

    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.run(["open", path], check=False)
        elif system == "Windows":
            os.startfile(path)  # type: ignore[attr-defined]
        else:  # Linux and others
            subprocess.run(["xdg-open", path], check=False)
    except Exception as e:  # pragma: no cover - OS dependent
        print(f"[WARN] Could not open file {path}: {e}")
