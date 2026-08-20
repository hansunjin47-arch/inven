"""3D Pareto front comparison plots across all 7 methods (MOPSO, CMODE,
Original PCN, PCN+buffer init. MOPSO-seed, Guided PCN MOPSO-seed,
PCN+buffer init. CMODE-seed, Guided PCN CMODE-seed), one figure per setting
(run 1, representative). Reuses compare_cmode_pcn_vs_all_full.py's path
resolution (fixed-code results where ready, pre-fix fallback otherwise).

    conda activate gpils
    python plot_cmode_pcn_full.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compute_metrics import load_front, N_RUNS
from compare_cmode_pcn_vs_all_full import SETTINGS, SETTING_LABELS, METHODS, METHOD_LABELS, path_for

OUT_DIR = "./result/pareto_7methods"
RUN_ID = 1

SERIES_STYLE = {
    "MOPSO": dict(marker="x", color="black"),
    "CMODE": dict(marker="D", facecolors="none", edgecolor="purple"),
    "PCN_original": dict(marker="s", facecolors="none", edgecolor="blue"),
    "PCN_variation1_mopso": dict(marker="o", facecolors="none", edgecolor="red"),
    "PCN_variation2_mopso": dict(marker="^", facecolors="none", edgecolor="green"),
    "PCN_variation1_cmode": dict(marker="P", facecolors="none", edgecolor="orange"),
    "PCN_variation2_cmode": dict(marker="*", facecolors="none", edgecolor="deeppink"),
}


def plot_setting(setting, run_id=RUN_ID):
    fig = plt.figure(figsize=(11, 8))
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
    ax.legend(fontsize=8)

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
