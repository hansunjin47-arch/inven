"""Builds the JSON payload (coverage tables, metrics tables, run-1 Pareto
front scatter images) consumed by the MOPSO-AMS comparison dashboard, for
MOPSO / MOPSO-AMS / Original PCN / PCN with buffer init. / Guided PCN
(softmax obj. selection) across all 6 settings. Mirrors the schema used by
the earlier "CMODE-Seeded PCN vs. All Baselines" dashboard.

Run in the `pcn` or `gpils` conda env:

    conda activate pcn
    python generate_mopso_ams_dashboard_payload.py
"""

import base64
import io
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compare_mopso_ams_all_settings import (
    METHODS, METHOD_LABELS, SETTINGS, N_RUNS, paths_for_setting, run_for_setting, load_front,
)

SETTING_LABELS = {
    "GBM": "GBM",
    "stat_low": "Stationary (low variance)",
    "stat_medium": "Stationary (medium variance)",
    "stat_high": "Stationary (high variance)",
    "seasonal": "Seasonal",
    "gs": "Case study (gs, test period)",
}

COLORS = {
    "MOPSO": "#4b5563",
    "MOPSO-AMS": "#a855f7",
    "Original PCN": "#3b82f6",
    "PCN with buffer init.": "#ef4444",
    "Guided PCN (softmax obj. selection)": "#22c55e",
}


def make_scatter_b64(setting):
    paths = paths_for_setting(setting)
    fig = plt.figure(figsize=(8.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    for m in METHODS:
        try:
            F = load_front(paths[m].format(run=1))
        except FileNotFoundError:
            continue
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


def df_to_rows(df, key_col):
    rows = []
    for idx in df.columns:
        row = {key_col: idx}
        for col in df.index:
            row[col] = df.loc[col, idx]
        rows.append(row)
    return rows


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

        # metrics_table has methods as columns, metric names as index (built that way in compare_*.py)
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

    with open("dashboard_payload_mopso_ams.json", "w", encoding="utf-8") as f:
        json.dump(payload, f)

    print("done -> dashboard_payload_mopso_ams.json")
