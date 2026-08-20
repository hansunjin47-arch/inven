"""3D Pareto front comparison plots (MOPSO, Original PCN, PCN with buffer
init., Guided PCN, CMODE), one figure per (setting, run) -- matches the
existing pareto_comparison_run_{setting}_{run}.png convention in ./result
(all 10 runs, not just a single representative run). Reuses main.py's
existing marker/color scheme and adds CMODE as a 5th series.

    conda activate gpils
    python plot_pareto_with_cmode.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compute_metrics import SETTINGS, SETTING_LABELS, N_RUNS, path_for, load_front

OUT_DIR = "./result"

SERIES_STYLE = {
    "MOPSO": dict(marker="x", color="black", facecolors=None, edgecolor=None),
    "PCN_original": dict(marker="s", color=None, facecolors="none", edgecolor="blue"),
    "PCN_variation1": dict(marker="o", color=None, facecolors="none", edgecolor="red"),
    "PCN_variation2": dict(marker="^", color=None, facecolors="none", edgecolor="green"),
    "CMODE": dict(marker="D", color=None, facecolors="none", edgecolor="purple"),
}
SERIES_LABEL = {
    "MOPSO": "MOPSO",
    "PCN_original": "Original PCN",
    "PCN_variation1": "PCN with buffer init.",
    "PCN_variation2": "Guided PCN",
    "CMODE": "CMODE",
}


def plot_setting_run(setting, run_id):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    for method, style in SERIES_STYLE.items():
        front = load_front(path_for(setting, method, run_id))
        kwargs = dict(marker=style["marker"], s=60, label=SERIES_LABEL[method])
        if style["color"] is not None:
            kwargs["color"] = style["color"]
        if style["facecolors"] is not None:
            kwargs["facecolors"] = style["facecolors"]
        if style["edgecolor"] is not None:
            kwargs["edgecolor"] = style["edgecolor"]
        ax.scatter(front[:, 1], front[:, 2], front[:, 0], **kwargs)  # Freq, Amount, TC

    ax.set_xlabel("Stockout Frequency")
    ax.set_ylabel("Stockout Amount")
    ax.set_zlabel("Total Cost (1e3)")
    ax.set_title(f"{SETTING_LABELS[setting]} Pareto Front Comparison (Run {run_id})")
    ax.legend()

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = f"{OUT_DIR}/pareto_comparison_with_cmode_{setting}_run{run_id}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    for setting in SETTINGS:
        for run_id in range(1, N_RUNS + 1):
            path = plot_setting_run(setting, run_id)
            print(f"[{setting} | run {run_id}] saved {path}")
