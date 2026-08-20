"""3D Pareto front scatter plots (TC, Frequency, Amount) for the 3 PCN
variants (Original, +buffer init, Guided PCN) at the adopted total_steps=2e6
config, one subplot per setting (run 1, consistent with the existing
generate_five_method_dashboard.py convention).

Run in the `pcn` conda env:

    conda activate pcn
    python plot_pareto_fronts_total_steps_2e6.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SETTINGS = ["GBM", "stat_low", "stat_medium", "stat_high", "seasonal", "gs"]
SETTING_LABELS = {
    "GBM": "GBM", "stat_low": "Stationary (low)", "stat_medium": "Stationary (medium)",
    "stat_high": "Stationary (high)", "seasonal": "Seasonal", "gs": "GS retail",
}
VERSIONS = ["original", "variation1", "variation2"]
VERSION_LABELS = {
    "original": "Original PCN",
    "variation1": "PCN + buffer init",
    "variation2": "Guided PCN",
}
COLORS = {
    "Original PCN": "#d62728",
    "PCN + buffer init": "#1f77b4",
    "Guided PCN": "#2ca02c",
}


def load_front(path):
    df = pd.read_csv(path)
    return df[["TC", "Frequency", "Amount"]].to_numpy(dtype=float)


def path_for(setting, version):
    return (f"./PCN_pbest_grid_costly/PCN_{setting}_{version}_total_steps_2e6/"
            f"PCN_{setting}_{version}_run_1.csv")


if __name__ == "__main__":
    fig = plt.figure(figsize=(18, 11))

    for i, setting in enumerate(SETTINGS):
        ax = fig.add_subplot(2, 3, i + 1, projection="3d")
        for version in VERSIONS:
            label = VERSION_LABELS[version]
            F = load_front(path_for(setting, version))
            ax.scatter(F[:, 1], F[:, 2], F[:, 0], label=label, color=COLORS[label], s=20, alpha=0.75)
        ax.set_xlabel("Stockout Frequency", fontsize=8)
        ax.set_ylabel("Stockout Amount", fontsize=8)
        ax.set_zlabel("Total Cost", fontsize=8)
        ax.set_title(f"{SETTING_LABELS[setting]}", fontsize=11)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=8)

    fig.suptitle("Pareto fronts at total_steps=2e6 (run 1) — all 6 settings", fontsize=14)
    fig.tight_layout()
    fig.savefig("pareto_fronts_total_steps_2e6.png", dpi=150, bbox_inches="tight")
    print("Saved -> pareto_fronts_total_steps_2e6.png")
