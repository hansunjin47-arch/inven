"""Builds the tabbed dashboard payload (same format as
generate_paperparams_dashboard_3settings.py / dashboard_template.html) for the
five-method comparison: MOPSO, Original PCN, PCN + buffer init, Guided PCN
(all pbest-fixed MOPSO seed where applicable), and MOPSO-AMS (paper params).
All 6 settings.

Run in the `pcn` or `gpils` conda env:

    conda activate pcn
    python generate_five_method_dashboard.py
"""

import base64
import io
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from compare_five_methods_pbest import (
    METHODS, METHOD_LABELS, paths_for_setting, load_front, non_dominated_front,
    compute_igd, compute_hv_mc, dominates, SETTINGS, N_RUNS,
)

SETTING_LABELS = {
    "GBM": "GBM",
    "stat_low": "Stationary (low variance)",
    "stat_medium": "Stationary (medium variance)",
    "stat_high": "Stationary (high variance)",
    "seasonal": "Seasonal",
    "gs": "gs",
}

COLORS = {
    "MOPSO": "#4b5563",
    "Original PCN": "#94a3b8",
    "PCN + buffer init": "#a855f7",
    "Guided PCN": "#22c55e",
    "MOPSO-AMS": "#f59e0b",
    "CAPQL": "#dc2626",
}


def set_coverage(A, B):
    if len(B) == 0:
        return np.nan
    count = sum(1 for b in B if any(dominates(a, b) for a in A))
    return count / len(B)


def max_spread(solutions):
    """Maximum spread = distance between the two farthest-apart points (raw scale, matches main.py)."""
    if len(solutions) < 2:
        return 0.0
    dists = np.linalg.norm(solutions[:, None, :] - solutions[None, :, :], axis=-1)
    return np.max(dists)


def spacing(solutions):
    """Spacing = std. dev. of each solution's nearest-neighbor distance (raw scale, matches main.py)."""
    if len(solutions) < 2:
        return 0.0
    distances = []
    for i, s in enumerate(solutions):
        other = np.delete(solutions, i, axis=0)
        d = np.min(np.linalg.norm(other - s, axis=1))
        distances.append(d)
    return float(np.std(distances))


def run_for_setting(setting):
    paths = paths_for_setting(setting)
    rng = np.random.default_rng(0)
    per_method = {m: {k: [] for k in ["hv", "igd", "n_solutions", "spacing", "max_spread"]} for m in METHODS}
    coverage = {(a, b): [] for a in METHODS for b in METHODS if a != b}

    for run in range(1, N_RUNS + 1):
        fronts = {m: load_front(paths[m].format(run=run)) for m in METHODS}
        combined = np.vstack(list(fronts.values()))
        lo, hi = combined.min(axis=0), combined.max(axis=0)
        denom = np.where(hi - lo == 0, 1.0, hi - lo)
        norm = {m: (fronts[m] - lo) / denom for m in METHODS}
        ref_point = np.array([1.1, 1.1, 1.1])
        ref_front_norm = (non_dominated_front(combined) - lo) / denom

        for m in METHODS:
            F = fronts[m]
            per_method[m]["hv"].append(compute_hv_mc(norm[m], ref_point, rng=rng))
            per_method[m]["igd"].append(compute_igd(norm[m], ref_front_norm))
            per_method[m]["n_solutions"].append(len(F))
            per_method[m]["spacing"].append(spacing(F))
            per_method[m]["max_spread"].append(max_spread(F))

        for a in METHODS:
            for b in METHODS:
                if a != b:
                    coverage[(a, b)].append(set_coverage(fronts[a], fronts[b]))

    metrics_table = pd.DataFrame(
        {
            METHOD_LABELS[m]: {
                "N solutions": f"{np.mean(per_method[m]['n_solutions']):.1f} ± {np.std(per_method[m]['n_solutions']):.1f}",
                "Spacing": f"{np.mean(per_method[m]['spacing']):.4f} ± {np.std(per_method[m]['spacing']):.4f}",
                "Max Spread": f"{np.mean(per_method[m]['max_spread']):.4f} ± {np.std(per_method[m]['max_spread']):.4f}",
                "HV (normalized)": f"{np.mean(per_method[m]['hv']):.4f} ± {np.std(per_method[m]['hv']):.4f}",
                "IGD (normalized)": f"{np.mean(per_method[m]['igd']):.4f} ± {np.std(per_method[m]['igd']):.4f}",
            }
            for m in METHODS
        }
    )

    coverage_table = pd.DataFrame(
        index=[METHOD_LABELS[m] for m in METHODS],
        columns=[METHOD_LABELS[m] for m in METHODS],
        dtype=object,
    )
    for a in METHODS:
        for b in METHODS:
            if a == b:
                coverage_table.loc[METHOD_LABELS[a], METHOD_LABELS[b]] = "-"
            else:
                vals = coverage[(a, b)]
                coverage_table.loc[METHOD_LABELS[a], METHOD_LABELS[b]] = f"{np.mean(vals):.2f} ± {np.std(vals):.2f}"

    return metrics_table, coverage_table


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

    with open("dashboard_payload_five_method.json", "w", encoding="utf-8") as f:
        json.dump(payload, f)

    print("done -> dashboard_payload_five_method.json")
