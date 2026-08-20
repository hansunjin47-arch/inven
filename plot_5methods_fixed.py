"""3D Pareto front plots for the core 5-method comparison (MOPSO, CMODE,
Original PCN, PCN with buffer init., Guided PCN -- all MOPSO-seeded, fixed
code), one figure per (setting, run), all 10 runs, across all 6 settings
including gs.

    conda activate gpils
    python plot_5methods_fixed.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compute_metrics import load_front, N_RUNS
from compare_5methods_fixed import SETTINGS, SETTING_LABELS, METHODS, METHOD_LABELS, path_for

OUT_DIR = "./result/pareto_5methods"
RUN_ID = 1

SERIES_STYLE = {
    "MOPSO": dict(marker="x", color="black"),
    "CMODE": dict(marker="D", facecolors="none", edgecolor="purple"),
    "PCN_original": dict(marker="s", facecolors="none", edgecolor="blue"),
    "PCN_variation1": dict(marker="o", facecolors="none", edgecolor="red"),
    "PCN_variation2": dict(marker="^", facecolors="none", edgecolor="green"),
}


def plot_setting(setting, run_id=RUN_ID):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    for method, style in SERIES_STYLE.items():
        front = load_front(path_for(setting, method, run_id))
        kwargs = dict(marker=style["marker"], s=60, label=METHOD_LABELS[method])
        if "color" in style:
            kwargs["color"] = style["color"]
        if "facecolors" in style:
            kwargs["facecolors"] = style["facecolors"]
        if "edgecolor" in style:
            kwargs["edgecolor"] = style["edgecolor"]
        ax.scatter(front[:, 1], front[:, 2], front[:, 0], **kwargs)  # Freq, Amount, TC

    ax.set_xlabel("Stockout Frequency")
    ax.set_ylabel("Stockout Amount")
    ax.set_zlabel("Total Cost (1e3)")
    ax.set_title(f"{SETTING_LABELS[setting]} Pareto Front Comparison (Run {run_id})")
    ax.legend()

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = f"{OUT_DIR}/{setting}_run{run_id}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    for setting in SETTINGS:
        for run_id in range(1, N_RUNS + 1):
            path = plot_setting(setting, run_id)
            print(f"[{setting} | run {run_id}] saved {path}")
