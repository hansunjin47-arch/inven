"""Compares the pre-fix (buggy Algorithm 2/3) results against the post-fix
results, for whichever (setting, version) pairs have all 10 fixed runs done
so far. Uses the same metric definitions as compute_metrics.py (paper-exact
Set Coverage / Spacing / Maximum Spread, plus HV/IGD with pairwise pooled
normalization, matching compute_pairwise_guided_cmode.py's approach).

    conda activate gpils
    python compare_fixed_vs_original.py
"""

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

SETTINGS = ["stat_low", "stat_medium"]
VERSIONS = ["original", "variation1", "variation2"]
VERSION_LABELS = {
    "original": "Original PCN",
    "variation1": "PCN with buffer init.",
    "variation2": "Guided PCN",
}

OLD_SUFFIX = {
    "stat_low": "_lognormal_3.29,0.23",
    "stat_medium": "",
}


def old_path(setting, version, run):
    suffix = OLD_SUFFIX[setting]
    return f"./PCN/PCN_stationary_{version}{suffix}/PCN_stationary_{version}_run_{run}.csv"


def fixed_path(setting, version, run):
    return f"./PCN_fixed/PCN_{setting}_{version}_fixed/PCN_{setting}_{version}_fixed_run_{run}.csv"


def main():
    rng = np.random.default_rng(0)

    for setting in SETTINGS:
        print(f"\n{'=' * 25} {setting} {'=' * 25}")

        for version in VERSIONS:
            per = {k: {"old": [], "fixed": []} for k in ["spacing", "max_spread", "mean_tc", "mean_freq", "mean_amt", "hv", "igd"]}
            cov_old_fixed, cov_fixed_old = [], []

            for run in range(1, N_RUNS + 1):
                F_old = load_front(old_path(setting, version, run))
                F_fixed = load_front(fixed_path(setting, version, run))

                combined = np.vstack([F_old, F_fixed])
                lo, hi = combined.min(axis=0), combined.max(axis=0)
                denom = np.where(hi - lo == 0, 1.0, hi - lo)
                norm_old = (F_old - lo) / denom
                norm_fixed = (F_fixed - lo) / denom
                ref_point = np.array([1.1, 1.1, 1.1])
                ref_front_norm = (non_dominated_front(combined) - lo) / denom

                for tag, F, norm in [("old", F_old, norm_old), ("fixed", F_fixed, norm_fixed)]:
                    per["spacing"][tag].append(spacing(F))
                    per["max_spread"][tag].append(max_spread(F))
                    per["mean_tc"][tag].append(F[:, 0].mean())
                    per["mean_freq"][tag].append(F[:, 1].mean())
                    per["mean_amt"][tag].append(F[:, 2].mean())
                    per["hv"][tag].append(compute_hv_mc(norm, ref_point, rng=rng))
                    per["igd"][tag].append(compute_igd(norm, ref_front_norm))

                cov_old_fixed.append(set_coverage(F_old, F_fixed))
                cov_fixed_old.append(set_coverage(F_fixed, F_old))

            table = pd.DataFrame(
                {
                    "Before fix": {
                        "Spacing": f"{np.mean(per['spacing']['old']):.2f} ± {np.std(per['spacing']['old']):.2f}",
                        "Maximum spread": f"{np.mean(per['max_spread']['old']):.2f} ± {np.std(per['max_spread']['old']):.2f}",
                        "Mean total cost (1e3)": f"{np.mean(per['mean_tc']['old']):.2f} ± {np.std(per['mean_tc']['old']):.2f}",
                        "Mean stockout frequency": f"{np.mean(per['mean_freq']['old']):.2f} ± {np.std(per['mean_freq']['old']):.2f}",
                        "Mean stockout amount": f"{np.mean(per['mean_amt']['old']):.2f} ± {np.std(per['mean_amt']['old']):.2f}",
                        "HV (pairwise-normalized)": f"{np.mean(per['hv']['old']):.4f} ± {np.std(per['hv']['old']):.4f}",
                        "IGD (pairwise-normalized)": f"{np.mean(per['igd']['old']):.4f} ± {np.std(per['igd']['old']):.4f}",
                    },
                    "After fix": {
                        "Spacing": f"{np.mean(per['spacing']['fixed']):.2f} ± {np.std(per['spacing']['fixed']):.2f}",
                        "Maximum spread": f"{np.mean(per['max_spread']['fixed']):.2f} ± {np.std(per['max_spread']['fixed']):.2f}",
                        "Mean total cost (1e3)": f"{np.mean(per['mean_tc']['fixed']):.2f} ± {np.std(per['mean_tc']['fixed']):.2f}",
                        "Mean stockout frequency": f"{np.mean(per['mean_freq']['fixed']):.2f} ± {np.std(per['mean_freq']['fixed']):.2f}",
                        "Mean stockout amount": f"{np.mean(per['mean_amt']['fixed']):.2f} ± {np.std(per['mean_amt']['fixed']):.2f}",
                        "HV (pairwise-normalized)": f"{np.mean(per['hv']['fixed']):.4f} ± {np.std(per['hv']['fixed']):.4f}",
                        "IGD (pairwise-normalized)": f"{np.mean(per['igd']['fixed']):.4f} ± {np.std(per['igd']['fixed']):.4f}",
                    },
                }
            )

            print(f"\n--- {VERSION_LABELS[version]} ---")
            print(f"C(before, after) = {np.mean(cov_old_fixed):.2f} ± {np.std(cov_old_fixed):.2f}  |  "
                  f"C(after, before) = {np.mean(cov_fixed_old):.2f} ± {np.std(cov_fixed_old):.2f}")
            print(table.to_string())


if __name__ == "__main__":
    main()
