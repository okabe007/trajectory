from tools.config_loader import load_config
from tools.derived_constants import calculate_derived_constants
from simulation_driver import run_simulation_from_ini
from spermsim.initialization import SimApp  # GUI用

if __name__ == "__main__":
    mode = "cli"  # "cli" または "gui" に切り替え可能

    if mode == "cli":
        print("[ENTRY] CLIモードで実行します")
        run_simulation_from_ini("sperm_config.ini")

    elif mode == "gui":
        print("[ENTRY] GUIモードで起動します")
        app = SimApp()
        app.run_gui()
