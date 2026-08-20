"""Before/after (pre-fix vs fixed Algorithm 2/3) comparison for seasonal and
GBM, for Original PCN / PCN+buffer init. / Guided PCN (all MOPSO-seeded),
plus CMODE (raw, unaffected by either bug) for context.

    conda activate gpils
    python compare_fixed_vs_original_seasonal_gbm.py
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

SETTINGS = ["seasonal", "GBM"]
VERSIONS = ["original", "variation1", "variation2"]
VERSION_LABELS = {
    "original": "Original PCN",
    "variation1": "PCN with buffer init.",
    "variation2": "Guided PCN",
}


def old_path(setting, version, run):
    return f"./PCN/PCN_{setting}_{version}/PCN_{setting}_{version}_run_{run}.csv"


def fixed_path(setting, version, run):
    return f"./PCN_fixed/PCN_{setting}_{version}_fixed/PCN_{setting}_{version}_fixed_run_{run}.csv"


def cmode_path(setting, run):
    return f"./CMODE/CMODE_{setting}/CMODE_run_{setting}_{run}.csv"


def main():
    rng = np.random.default_rng(0)

    for setting in SETTINGS:
        print(f"\n{'=' * 25} {setting} {'=' * 25}")

        for version in VERSIONS:
            per = {k: {"old": [], "fixed": [], "cmode": []} for k in ["spacing", "max_spread", "mean_tc", "mean_freq", "mean_amt", "hv", "igd"]}
            cov = {("old", "fixed"): [], ("fixed", "old"): [], ("cmode", "fixed"): [], ("fixed", "cmode"): [], ("cmode", "old"): [], ("old", "cmode"): []}

            for run in range(1, N_RUNS + 1):
                F_old = load_front(old_path(setting, version, run))
                F_fixed = load_front(fixed_path(setting, version, run))
                F_cmode = load_front(cmode_path(setting, run))

                combined = np.vstack([F_old, F_fixed, F_cmode])
                lo, hi = combined.min(axis=0), combined.max(axis=0)
                denom = np.where(hi - lo == 0, 1.0, hi - lo)
                norms = {"old": (F_old - lo) / denom, "fixed": (F_fixed - lo) / denom, "cmode": (F_cmode - lo) / denom}
                ref_point = np.array([1.1, 1.1, 1.1])
                ref_front_norm = (non_dominated_front(combined) - lo) / denom

                fronts = {"old": F_old, "fixed": F_fixed, "cmode": F_cmode}
                for tag, F in fronts.items():
                    per["spacing"][tag].append(spacing(F))
                    per["max_spread"][tag].append(max_spread(F))
                    per["mean_tc"][tag].append(F[:, 0].mean())
                    per["mean_freq"][tag].append(F[:, 1].mean())
                    per["mean_amt"][tag].append(F[:, 2].mean())
                    per["hv"][tag].append(compute_hv_mc(norms[tag], ref_point, rng=rng))
                    per["igd"][tag].append(compute_igd(norms[tag], ref_front_norm))

                cov[("old", "fixed")].append(set_coverage(F_old, F_fixed))
                cov[("fixed", "old")].append(set_coverage(F_fixed, F_old))
                cov[("cmode", "fixed")].append(set_coverage(F_cmode, F_fixed))
                cov[("fixed", "cmode")].append(set_coverage(F_fixed, F_cmode))
                cov[("cmode", "old")].append(set_coverage(F_cmode, F_old))
                cov[("old", "cmode")].append(set_coverage(F_old, F_cmode))

            table = pd.DataFrame(
                {
                    "Before fix": {
                        "Spacing": f"{np.mean(per['spacing']['old']):.2f} ± {np.std(per['spacing']['old']):.2f}",
                        "Maximum spread": f"{np.mean(per['max_spread']['old']):.2f} ± {np.std(per['max_spread']['old']):.2f}",
                        "Mean total cost (1e3)": f"{np.mean(per['mean_tc']['old']):.2f} ± {np.std(per['mean_tc']['old']):.2f}",
                        "Mean stockout frequency": f"{np.mean(per['mean_freq']['old']):.2f} ± {np.std(per['mean_freq']['old']):.2f}",
                        "Mean stockout amount": f"{np.mean(per['mean_amt']['old']):.2f} ± {np.std(per['mean_amt']['old']):.2f}",
                        "HV": f"{np.mean(per['hv']['old']):.4f} ± {np.std(per['hv']['old']):.4f}",
                        "IGD": f"{np.mean(per['igd']['old']):.4f} ± {np.std(per['igd']['old']):.4f}",
                    },
                    "After fix": {
                        "Spacing": f"{np.mean(per['spacing']['fixed']):.2f} ± {np.std(per['spacing']['fixed']):.2f}",
                        "Maximum spread": f"{np.mean(per['max_spread']['fixed']):.2f} ± {np.std(per['max_spread']['fixed']):.2f}",
                        "Mean total cost (1e3)": f"{np.mean(per['mean_tc']['fixed']):.2f} ± {np.std(per['mean_tc']['fixed']):.2f}",
                        "Mean stockout frequency": f"{np.mean(per['mean_freq']['fixed']):.2f} ± {np.std(per['mean_freq']['fixed']):.2f}",
                        "Mean stockout amount": f"{np.mean(per['mean_amt']['fixed']):.2f} ± {np.std(per['mean_amt']['fixed']):.2f}",
                        "HV": f"{np.mean(per['hv']['fixed']):.4f} ± {np.std(per['hv']['fixed']):.4f}",
                        "IGD": f"{np.mean(per['igd']['fixed']):.4f} ± {np.std(per['igd']['fixed']):.4f}",
                    },
                    "CMODE": {
                        "Spacing": f"{np.mean(per['spacing']['cmode']):.2f} ± {np.std(per['spacing']['cmode']):.2f}",
                        "Maximum spread": f"{np.mean(per['max_spread']['cmode']):.2f} ± {np.std(per['max_spread']['cmode']):.2f}",
                        "Mean total cost (1e3)": f"{np.mean(per['mean_tc']['cmode']):.2f} ± {np.std(per['mean_tc']['cmode']):.2f}",
                        "Mean stockout frequency": f"{np.mean(per['mean_freq']['cmode']):.2f} ± {np.std(per['mean_freq']['cmode']):.2f}",
                        "Mean stockout amount": f"{np.mean(per['mean_amt']['cmode']):.2f} ± {np.std(per['mean_amt']['cmode']):.2f}",
                        "HV": f"{np.mean(per['hv']['cmode']):.4f} ± {np.std(per['hv']['cmode']):.4f}",
                        "IGD": f"{np.mean(per['igd']['cmode']):.4f} ± {np.std(per['igd']['cmode']):.4f}",
                    },
                }
            )

            print(f"\n--- {VERSION_LABELS[version]} ---")
            print(f"C(before, after)={np.mean(cov[('old','fixed')]):.2f}  C(after, before)={np.mean(cov[('fixed','old')]):.2f}  |  "
                  f"C(CMODE, after)={np.mean(cov[('cmode','fixed')]):.2f}  C(after, CMODE)={np.mean(cov[('fixed','cmode')]):.2f}  |  "
                  f"C(CMODE, before)={np.mean(cov[('cmode','old')]):.2f}  C(before, CMODE)={np.mean(cov[('old','cmode')]):.2f}")
            print(table.to_string())


if __name__ == "__main__":
    main()
