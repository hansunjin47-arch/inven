"""Compares CMODE-seeded PCN (variation1/variation2) against all existing
baselines -- MOPSO, CMODE, Original PCN, and MOPSO-seeded PCN -- using the
FIXED pcn_continuous.py (Algorithm 2 curriculum restored, Algorithm 3 min/max
bug fixed) results wherever available.

Only settings where every method's fixed-code results are ready are included.
As of this run that is stat_low/stat_medium/stat_high (seasonal/GBM/gs still
finishing the MOPSO-seed rerun; CMODE-seed is done for all 6).

    conda activate gpils
    python compare_cmode_pcn_vs_all.py
"""

import os
import numpy as np
import pandas as pd

from compute_metrics import (
    N_RUNS,
    load_front,
    set_coverage,
    spacing,
    max_spread,
    non_dominated_front,
    compute_igd,
    compute_hv_mc,
)

SETTINGS = ["stat_low", "stat_medium", "stat_high"]
SETTING_LABELS = {
    "stat_low": "Stationary (low variance)",
    "stat_medium": "Stationary (medium variance)",
    "stat_high": "Stationary (high variance)",
}
MOPSO_SUFFIX = {
    "stat_low": "_lognormal_3.29,0.23",
    "stat_medium": "",
    "stat_high": "_lognormal_3.0,0.8",
}

METHODS = [
    "MOPSO",
    "CMODE",
    "PCN_original",
    "PCN_variation1_mopso",
    "PCN_variation2_mopso",
    "PCN_variation1_cmode",
    "PCN_variation2_cmode",
]
METHOD_LABELS = {
    "MOPSO": "MOPSO",
    "CMODE": "CMODE",
    "PCN_original": "Original PCN",
    "PCN_variation1_mopso": "PCN+buffer init. (MOPSO-seed)",
    "PCN_variation2_mopso": "Guided PCN (MOPSO-seed)",
    "PCN_variation1_cmode": "PCN+buffer init. (CMODE-seed)",
    "PCN_variation2_cmode": "Guided PCN (CMODE-seed)",
}


def path_for(setting, method, run):
    if method == "MOPSO":
        return f"./MOPSO/MOPSO_stationary{MOPSO_SUFFIX[setting]}/MOPSO_run_stationary_{run}.csv"
    if method == "CMODE":
        return f"./CMODE/CMODE_{setting}/CMODE_run_{setting}_{run}.csv"
    if method == "PCN_original":
        return f"./PCN_fixed/PCN_{setting}_original_fixed/PCN_{setting}_original_fixed_run_{run}.csv"
    if method == "PCN_variation1_mopso":
        return f"./PCN_fixed/PCN_{setting}_variation1_fixed/PCN_{setting}_variation1_fixed_run_{run}.csv"
    if method == "PCN_variation2_mopso":
        return f"./PCN_fixed/PCN_{setting}_variation2_fixed/PCN_{setting}_variation2_fixed_run_{run}.csv"
    if method == "PCN_variation1_cmode":
        return f"./PCN/PCN_{setting}_variation1_cmode_seed/PCN_{setting}_variation1_cmode_seed_run_{run}.csv"
    if method == "PCN_variation2_cmode":
        return f"./PCN/PCN_{setting}_variation2_cmode_seed/PCN_{setting}_variation2_cmode_seed_run_{run}.csv"
    raise ValueError(method)


def main():
    rng = np.random.default_rng(0)
    all_results = {}

    for setting in SETTINGS:
        per_method = {m: {k: [] for k in ["spacing", "max_spread", "mean_tc", "mean_freq", "mean_amt", "hv", "igd"]} for m in METHODS}
        coverage = {(a, b): [] for a in METHODS for b in METHODS if a != b}

        for run in range(1, N_RUNS + 1):
            fronts = {m: load_front(path_for(setting, m, run)) for m in METHODS}

            combined = np.vstack(list(fronts.values()))
            lo, hi = combined.min(axis=0), combined.max(axis=0)
            denom = np.where(hi - lo == 0, 1.0, hi - lo)
            norm = {m: (fronts[m] - lo) / denom for m in METHODS}
            ref_point = np.array([1.1, 1.1, 1.1])
            ref_front_norm = (non_dominated_front(combined) - lo) / denom

            for m in METHODS:
                F = fronts[m]
                per_method[m]["spacing"].append(spacing(F))
                per_method[m]["max_spread"].append(max_spread(F))
                per_method[m]["mean_tc"].append(F[:, 0].mean())
                per_method[m]["mean_freq"].append(F[:, 1].mean())
                per_method[m]["mean_amt"].append(F[:, 2].mean())
                per_method[m]["hv"].append(compute_hv_mc(norm[m], ref_point, rng=rng))
                per_method[m]["igd"].append(compute_igd(norm[m], ref_front_norm))

            for a in METHODS:
                for b in METHODS:
                    if a != b:
                        coverage[(a, b)].append(set_coverage(fronts[a], fronts[b]))

        metrics_table = pd.DataFrame(
            {
                METHOD_LABELS[m]: {
                    "Spacing": f"{np.mean(per_method[m]['spacing']):.2f} ± {np.std(per_method[m]['spacing']):.2f}",
                    "Maximum spread": f"{np.mean(per_method[m]['max_spread']):.2f} ± {np.std(per_method[m]['max_spread']):.2f}",
                    "Mean total cost (1e3)": f"{np.mean(per_method[m]['mean_tc']):.2f} ± {np.std(per_method[m]['mean_tc']):.2f}",
                    "Mean stockout frequency": f"{np.mean(per_method[m]['mean_freq']):.2f} ± {np.std(per_method[m]['mean_freq']):.2f}",
                    "Mean stockout amount": f"{np.mean(per_method[m]['mean_amt']):.2f} ± {np.std(per_method[m]['mean_amt']):.2f}",
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
                    coverage_table.loc[METHOD_LABELS[a], METHOD_LABELS[b]] = f"{np.mean(vals):.2f}"

        all_results[setting] = {"metrics": metrics_table, "coverage": coverage_table}

    return all_results


if __name__ == "__main__":
    results = main()
    for setting in SETTINGS:
        print(f"\n{'=' * 20} {SETTING_LABELS[setting]} {'=' * 20}")
        print("\n--- Set Coverage C(A, B): rows=A, cols=B ---")
        print(results[setting]["coverage"].to_string())
        print("\n--- Metrics ---")
        print(results[setting]["metrics"].to_string())
