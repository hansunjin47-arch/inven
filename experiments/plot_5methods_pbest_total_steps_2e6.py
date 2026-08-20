import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _sub in ("baselines", "envs", "models", "utils", "experiments"):
    _p = _os.path.join(_ROOT, _sub)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
_os.chdir(_ROOT)

"""3D Pareto front plots for the current 5-method comparison (MOPSO pbest-
fixed, Original PCN, PCN + buffer init, Guided PCN -- all at max_size=750
-- and MOPSO-AMS at the 2x matched budget), one figure per (setting, run),
all 10 runs, across all 6 settings. Same per-run/per-setting output style as
the older plot_5methods_fixed.py (CMODE-era), but using the method set and
data paths from compare_five_methods_pbest.py.

    conda activate pcn
    python plot_5methods_pbest_total_steps_2e6.py
"""

import os

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import pandas as pd

from compare_five_methods_pbest import METHODS, METHOD_LABELS, SETTINGS, N_RUNS, paths_for_setting

OUT_DIR = "results/pareto_5methods"

SETTING_LABELS = {
    "GBM": "GBM", "stat_low": "Stationary (low)", "stat_medium": "Stationary (medium)",
    "stat_high": "Stationary (high)", "seasonal": "Seasonal", "gs": "GS retail",
}
# Matches the dashboard's make_scatter_b64 style (generate_five_method_dashboard.py):
# uniform filled circles, color-only differentiation, alpha blending -- this
# reads far better than distinct hollow shapes once points overlap heavily.
SERIES_STYLE = {
    "MOPSO": dict(color="#000000"),
    "MOPSO_AMS": dict(color="#1d4ed8"),
    "CAPQL": dict(color="#9333ea"),
    "PCN_original": dict(color="#16a34a"),
    "PCN_variation1": dict(color="#eab308"),
    "PCN_variation2": dict(color="#dc2626"),
}
MARKER_SIZE = 28
MARKER_ALPHA = 0.85


def load_front(path):
    df = pd.read_csv(path)
    return df[["TC", "Frequency", "Amount"]].to_numpy(dtype=float)


def plot_setting(setting, run_id):
    paths = paths_for_setting(setting)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    for method, style in SERIES_STYLE.items():
        front = load_front(paths[method].format(run=run_id))
        ax.scatter(
            front[:, 1], front[:, 2], front[:, 0],
            label=METHOD_LABELS[method], color=style["color"],
            s=MARKER_SIZE, alpha=MARKER_ALPHA, depthshade=False,
        )  # Freq, Amount, TC

    ax.set_xlabel("Stockout Frequency")
    ax.set_ylabel("Stockout Amount")
    ax.set_zlabel("Total Cost")
    ax.legend()

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = f"{OUT_DIR}/{setting}_run{run_id}.pdf"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    for setting in SETTINGS:
        for run_id in range(1, N_RUNS + 1):
            path = plot_setting(setting, run_id)
            print(f"[{setting} | run {run_id}] saved {path}")
