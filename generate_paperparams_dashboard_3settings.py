"""Builds a tabbed dashboard payload (GBM, stat_low, stat_medium -- the 3
settings with MOPSO-AMS paper-params data completed so far) comparing
MOPSO, MOPSO-AMS (matched budget), MOPSO-AMS (paper params I=150,T=1000),
and Guided PCN (softmax obj. selection).

Run in the `pcn` or `gpils` conda env:

    conda activate pcn
    python generate_paperparams_dashboard_3settings.py
"""

import base64
import io
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compare_mopso_ams_paperparams_multisetting import (
    METHODS, METHOD_LABELS, paths_for_setting, load_front, run_for_setting,
)

SETTINGS = ["GBM", "stat_low", "stat_medium"]
SETTING_LABELS = {
    "GBM": "GBM",
    "stat_low": "Stationary (low variance)",
    "stat_medium": "Stationary (medium variance)",
}

COLORS = {
    "MOPSO": "#4b5563",
    "MOPSO-AMS (matched budget)": "#a855f7",
    "MOPSO-AMS (paper params)": "#f59e0b",
    "Guided PCN (softmax obj. selection)": "#22c55e",
}


def make_scatter_b64(setting):
    paths = paths_for_setting(setting)
    fig = plt.figure(figsize=(8.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    for m in METHODS:
        F = load_front(paths[m].format(run=1))
        label = METHOD_LABELS[m]
        ax.scatter(F[:, 1], F[:, 2], F[:, 0], label=label, color=COLORS[label], s=25, alpha=0.75)
    ax.set_xlabel("Stockout Frequency")
    ax.set_ylabel("Stockout Amount")
    ax.set_zlabel("Total Cost")
    ax.set_title(f"{SETTING_LABELS[setting]} — Non-dominated front (run 1)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=8)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


if __name__ == "__main__":
    method_labels = [METHOD_LABELS[m] for m in METHODS]

    payload = {
        "data": {},
        "images": {},
        "settings": SETTINGS,
        "methods": method_labels,
        "colors": COLORS,
    }

    for setting in SETTINGS:
        print(f"=== {setting} ===")
        metrics_table, coverage_table = run_for_setting(setting)

        coverage_rows = []
        for method_row in coverage_table.index:
            row = {"Method": method_row}
            for method_col in coverage_table.columns:
                row[method_col] = coverage_table.loc[method_row, method_col]
            coverage_rows.append(row)

        metrics_rows = []
        for metric_name in metrics_table.index:
            row = {"Metric": metric_name}
            for method_col in metrics_table.columns:
                row[method_col] = metrics_table.loc[metric_name, method_col]
            metrics_rows.append(row)

        payload["data"][setting] = {
            "label": SETTING_LABELS[setting],
            "coverage": coverage_rows,
            "coverage_cols": list(coverage_table.columns),
            "metrics": metrics_rows,
            "metrics_cols": list(metrics_table.columns),
        }
        payload["images"][setting] = make_scatter_b64(setting)

    with open("dashboard_payload_paperparams_3settings.json", "w", encoding="utf-8") as f:
        json.dump(payload, f)

    print("done -> dashboard_payload_paperparams_3settings.json")
