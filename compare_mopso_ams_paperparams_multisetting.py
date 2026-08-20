"""Compares MOPSO, MOPSO-AMS (matched budget), MOPSO-AMS (paper params
I=150,T=1000), and Guided PCN across whichever settings have paper-params
data available yet (checks and skips missing).

Run in the `pcn` or `gpils` conda env:

    conda activate pcn
    python compare_mopso_ams_paperparams_multisetting.py
"""

import numpy as np
import pandas as pd

METHODS = ["MOPSO", "MOPSO_AMS_matched", "MOPSO_AMS_paperparams", "PCN_variation2"]
METHOD_LABELS = {
    "MOPSO": "MOPSO",
    "MOPSO_AMS_matched": "MOPSO-AMS (matched budget)",
    "MOPSO_AMS_paperparams": "MOPSO-AMS (paper params)",
    "PCN_variation2": "Guided PCN (softmax obj. selection)",
}
N_RUNS = 10
SETTINGS = ["GBM", "stat_low", "stat_medium", "stat_high", "seasonal", "gs"]

MOPSO_PATH_FOR_SETTING = {
    "GBM": "./MOPSO/MOPSO_GBM/MOPSO_run_GBM_{run}.csv",
    "stat_low": "./MOPSO/MOPSO_stationary_lognormal_3.29,0.23/MOPSO_run_stationary_{run}.csv",
    "stat_medium": "./MOPSO/MOPSO_stationary/MOPSO_run_stationary_{run}.csv",
    "stat_high": "./MOPSO/MOPSO_stationary_lognormal_3.0,0.8/MOPSO_run_stationary_{run}.csv",
    "seasonal": "./MOPSO/MOPSO_seasonal/MOPSO_run_seasonal_{run}.csv",
    "gs": "./MOPSO/MOPSO_run_gs_test1_{run}.csv",
}
MOPSO_AMS_MATCHED_PATH = {
    "GBM": "./MOPSO_AMS/MOPSO_AMS_GBM/MOPSO_AMS_run_GBM_{run}.csv",
    "stat_low": "./MOPSO_AMS/MOPSO_AMS_stat_low/MOPSO_AMS_run_stat_low_{run}.csv",
    "stat_medium": "./MOPSO_AMS/MOPSO_AMS_stat_medium/MOPSO_AMS_run_stat_medium_{run}.csv",
    "stat_high": "./MOPSO_AMS/MOPSO_AMS_stat_high/MOPSO_AMS_run_stat_high_{run}.csv",
    "seasonal": "./MOPSO_AMS/MOPSO_AMS_seasonal/MOPSO_AMS_run_seasonal_{run}.csv",
    "gs": "./MOPSO_AMS/MOPSO_AMS_gs/MOPSO_AMS_run_gs_test1_{run}.csv",
}
MOPSO_AMS_PAPERPARAMS_PATH = {
    "GBM": "./MOPSO_AMS_paperparams/MOPSO_AMS_GBM/MOPSO_AMS_run_GBM_{run}.csv",
    "stat_low": "./MOPSO_AMS_paperparams/MOPSO_AMS_stat_low/MOPSO_AMS_run_stat_low_{run}.csv",
    "stat_medium": "./MOPSO_AMS_paperparams/MOPSO_AMS_stat_medium/MOPSO_AMS_run_stat_medium_{run}.csv",
    "stat_high": "./MOPSO_AMS_paperparams/MOPSO_AMS_stat_high/MOPSO_AMS_run_stat_high_{run}.csv",
    "seasonal": "./MOPSO_AMS_paperparams/MOPSO_AMS_seasonal/MOPSO_AMS_run_seasonal_{run}.csv",
    "gs": "./MOPSO_AMS_paperparams/MOPSO_AMS_gs/MOPSO_AMS_run_gs_test1_{run}.csv",
}


def paths_for_setting(setting):
    return {
        "MOPSO": MOPSO_PATH_FOR_SETTING[setting],
        "MOPSO_AMS_matched": MOPSO_AMS_MATCHED_PATH[setting],
        "MOPSO_AMS_paperparams": MOPSO_AMS_PAPERPARAMS_PATH[setting],
        "PCN_variation2": f"./PCN_stochastic_objsel_test/PCN_{setting}_variation2_stochobjsel/PCN_{setting}_variation2_stochobjsel_run_{{run}}.csv",
    }


def load_front(path):
    df = pd.read_csv(path)
    return df[["TC", "Frequency", "Amount"]].to_numpy(dtype=float)


def dominates(p, q):
    return bool(np.all(p <= q) and np.any(p < q))


def set_coverage(A, B):
    if len(B) == 0:
        return np.nan
    count = sum(1 for b in B if any(dominates(a, b) for a in A))
    return count / len(B)


def non_dominated_front(points):
    points = np.asarray(points)
    keep = np.ones(len(points), dtype=bool)
    for i, p in enumerate(points):
        for j, q in enumerate(points):
            if i != j and dominates(q, p):
                keep[i] = False
                break
    return points[keep]


def compute_igd(front, reference_front):
    if len(front) == 0 or len(reference_front) == 0:
        return np.nan
    dists = np.linalg.norm(reference_front[:, None, :] - front[None, :, :], axis=-1)
    return float(np.mean(np.min(dists, axis=1)))


def compute_hv_mc(front, ref_point, n_samples=300_000, rng=None):
    rng = rng or np.random.default_rng(0)
    if len(front) == 0:
        return 0.0
    box_volume = float(np.prod(ref_point))
    dominated = np.zeros(n_samples, dtype=bool)
    chunk = 20_000
    for start in range(0, n_samples, chunk):
        s = rng.uniform(low=0.0, high=ref_point, size=(min(chunk, n_samples - start), len(ref_point)))
        le = np.all(front[:, None, :] <= s[None, :, :], axis=2)
        dominated[start:start + s.shape[0]] = np.any(le, axis=0)
    return float(dominated.mean() * box_volume)


def run_for_setting(setting):
    paths = paths_for_setting(setting)
    rng = np.random.default_rng(0)
    per_method = {m: {k: [] for k in ["hv", "igd", "n_solutions"]} for m in METHODS}
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

        for a in METHODS:
            for b in METHODS:
                if a != b:
                    coverage[(a, b)].append(set_coverage(fronts[a], fronts[b]))

    metrics_table = pd.DataFrame(
        {
            METHOD_LABELS[m]: {
                "N solutions": f"{np.mean(per_method[m]['n_solutions']):.1f} ± {np.std(per_method[m]['n_solutions']):.1f}",
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


if __name__ == "__main__":
    for setting in SETTINGS:
        print(f"\n\n=============== {setting} ===============")
        try:
            metrics_table, coverage_table = run_for_setting(setting)
        except FileNotFoundError as e:
            print(f"[skipped: missing file] {e}")
            continue
        print("\n--- Set Coverage C(A, B): rows=A, cols=B ---")
        print(coverage_table.to_string())
        print("\n--- Metrics ---")
        print(metrics_table.to_string())
