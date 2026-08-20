"""3D Pareto front scatter plots (TC, Frequency, Amount) for all 5 methods
(MOPSO, Original PCN, PCN + buffer init, Guided PCN, MOPSO-AMS), one subplot
per setting (run 1) -- the same 5-method / color scheme already used in
generate_five_method_dashboard.py's per-setting scatter tab, just laid out
as a single static 2x3 grid figure for all 6 settings at once.

Run in the `pcn` conda env:

    conda activate pcn
    python plot_pareto_fronts_five_methods.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from compare_five_methods_pbest import METHODS, METHOD_LABELS, SETTINGS, paths_for_setting

SETTING_LABELS = {
    "GBM": "GBM", "stat_low": "Stationary (low)", "stat_medium": "Stationary (medium)",
    "stat_high": "Stationary (high)", "seasonal": "Seasonal", "gs": "GS retail",
}
COLORS = {
    "MOPSO": "#4b5563",
    "Original PCN": "#94a3b8",
    "PCN + buffer init": "#a855f7",
    "Guided PCN": "#22c55e",
    "MOPSO-AMS": "#f59e0b",
}


def load_front(path):
    df = pd.read_csv(path)
    return df[["TC", "Frequency", "Amount"]].to_numpy(dtype=float)


if __name__ == "__main__":
    fig = plt.figure(figsize=(19, 11))

    for i, setting in enumerate(SETTINGS):
        ax = fig.add_subplot(2, 3, i + 1, projection="3d")
        paths = paths_for_setting(setting)
        for m in METHODS:
            label = METHOD_LABELS[m]
            F = load_front(paths[m].format(run=1))
            ax.scatter(F[:, 1], F[:, 2], F[:, 0], label=label, color=COLORS[label], s=22, alpha=0.75)
        ax.set_xlabel("Stockout Frequency", fontsize=8)
        ax.set_ylabel("Stockout Amount", fontsize=8)
        ax.set_zlabel("Total Cost", fontsize=8)
        ax.set_title(f"{SETTING_LABELS[setting]}", fontsize=11)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=8)

    fig.suptitle("Pareto fronts, all 5 methods (run 1) — total_steps=2e6, MOPSO-AMS 2x budget", fontsize=14)
    fig.tight_layout()
    fig.savefig("pareto_fronts_five_methods.png", dpi=150, bbox_inches="tight")
    print("Saved -> pareto_fronts_five_methods.png")
