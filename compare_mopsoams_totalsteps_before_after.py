"""Three-way comparison: MOPSO-AMS (paper params) vs Guided PCN baseline
(total_steps=1e6) vs Guided PCN with total_steps=2e6 (grid search finding),
all with the ORIGINAL (non-pbest) MOPSO seed, single normalization pass per
setting. Answers: does doubling total_steps narrow the gap to MOPSO-AMS?

Run in the `pcn` or `gpils` conda env:

    conda activate pcn
    python compare_mopsoams_totalsteps_before_after.py
"""

import numpy as np
import pandas as pd

SETTINGS = ["GBM", "stat_low", "stat_medium", "stat_high", "seasonal", "gs"]
N_RUNS = 10

METHODS = ["MOPSO_AMS_paperparams", "PCN_baseline_1e6", "PCN_total_steps_2e6"]
METHOD_LABELS = {
    "MOPSO_AMS_paperparams": "MOPSO-AMS (paper params)",
    "PCN_baseline_1e6": "Guided PCN (total_steps=1e6, before)",
    "PCN_total_steps_2e6": "Guided PCN (total_steps=2e6, after)",
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
        "MOPSO_AMS_paperparams": MOPSO_AMS_PAPERPARAMS_PATH[setting],
        "PCN_baseline_1e6": f"./PCN_stochastic_objsel_test/PCN_{setting}_variation2_stochobjsel/PCN_{setting}_variation2_stochobjsel_run_{{run}}.csv",
        "PCN_total_steps_2e6": f"./PCN_grid_search/PCN_{setting}_variation2_total_steps_2000000.0/PCN_{setting}_variation2_run_{{run}}.csv",
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

    print(f"\n=============== {setting} ===============")
    for m in METHODS:
        print(f"  {METHOD_LABELS[m]:38s}: N={np.mean(per_method[m]['n_solutions']):6.1f}  HV={np.mean(per_method[m]['hv']):.4f}  IGD={np.mean(per_method[m]['igd']):.4f}")

    hv_gap_before = np.mean(per_method["MOPSO_AMS_paperparams"]["hv"]) - np.mean(per_method["PCN_baseline_1e6"]["hv"])
    hv_gap_after = np.mean(per_method["MOPSO_AMS_paperparams"]["hv"]) - np.mean(per_method["PCN_total_steps_2e6"]["hv"])
    cov_before = np.nanmean(coverage[("MOPSO_AMS_paperparams", "PCN_baseline_1e6")])
    cov_after = np.nanmean(coverage[("MOPSO_AMS_paperparams", "PCN_total_steps_2e6")])
    print(f"  HV gap (MOPSO-AMS - PCN): before={hv_gap_before:+.4f}  after={hv_gap_after:+.4f}  ({'narrowed' if abs(hv_gap_after) < abs(hv_gap_before) else 'widened'})")
    print(f"  Coverage (MOPSO-AMS dominates PCN): before={cov_before:.2f}  after={cov_after:.2f}")


if __name__ == "__main__":
    for setting in SETTINGS:
        try:
            run_for_setting(setting)
        except FileNotFoundError as e:
            print(f"\n=============== {setting} ===============\n  [skipped: missing file] {e}")
