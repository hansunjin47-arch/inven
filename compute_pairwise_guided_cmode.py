"""Focused head-to-head comparison: Guided PCN (the paper's proposed
algorithm) vs. CMODE, per demand setting.

Set Coverage / Spacing / Maximum Spread are unchanged from compute_metrics.py
(they're computed per-method and don't depend on which other methods are in
the comparison). HV and IGD are RE-normalized/re-referenced using only these
two methods' pooled fronts (rather than all 5), so the other baselines'
extreme corner solutions don't distort the reference box / reference front
for this specific pairwise comparison.

    conda activate gpils
    python compute_pairwise_guided_cmode.py
"""

import numpy as np
import pandas as pd

from compute_metrics import (
    SETTINGS,
    SETTING_LABELS,
    N_RUNS,
    path_for,
    load_front,
    set_coverage,
    spacing,
    max_spread,
    non_dominated_front,
    compute_igd,
    compute_hv_mc,
)

METHODS = ["PCN_variation2", "CMODE"]
METHOD_LABELS = {"PCN_variation2": "Guided PCN (proposed)", "CMODE": "CMODE"}


def main():
    rng = np.random.default_rng(0)
    tables = {}

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
                    "HV (pairwise-normalized)": f"{np.mean(per_method[m]['hv']):.4f} ± {np.std(per_method[m]['hv']):.4f}",
                    "IGD (pairwise-normalized)": f"{np.mean(per_method[m]['igd']):.4f} ± {np.std(per_method[m]['igd']):.4f}",
                }
                for m in METHODS
            }
        )

        cov_gp_cmode = coverage[("PCN_variation2", "CMODE")]
        cov_cmode_gp = coverage[("CMODE", "PCN_variation2")]
        coverage_line = (
            f"C(Guided PCN, CMODE) = {np.mean(cov_gp_cmode):.2f} ± {np.std(cov_gp_cmode):.2f}  |  "
            f"C(CMODE, Guided PCN) = {np.mean(cov_cmode_gp):.2f} ± {np.std(cov_cmode_gp):.2f}"
        )

        tables[setting] = {"metrics": metrics_table, "coverage_line": coverage_line}

    return tables


if __name__ == "__main__":
    results = main()
    for setting in SETTINGS:
        print(f"\n{'=' * 20} {SETTING_LABELS[setting]} {'=' * 20}")
        print(results[setting]["coverage_line"])
        print(results[setting]["metrics"].to_string())
