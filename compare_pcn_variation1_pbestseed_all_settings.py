"""Compares PCN with buffer init. (variation1) seeded from the original
MOPSO archive vs seeded from the pbest-tie-break-fixed MOPSO archive,
across all 6 settings -- tests whether a better MOPSO seed changes
variation1's own final training outcome (mirrors
compare_pcn_pbestseed_all_settings.py, which did this for variation2).

Run in the `pcn` or `gpils` conda env:

    conda activate pcn
    python compare_pcn_variation1_pbestseed_all_settings.py
"""

import numpy as np
import pandas as pd

METHODS = ["PCN_original_seed", "PCN_pbest_seed"]
METHOD_LABELS = {
    "PCN_original_seed": "variation1 (original MOPSO seed)",
    "PCN_pbest_seed": "variation1 (pbest-fixed MOPSO seed)",
}
N_RUNS = 10
SETTINGS = ["GBM", "stat_low", "stat_medium", "stat_high", "seasonal", "gs"]

PATHS = {
    "PCN_original_seed": "./PCN_fixed/PCN_{setting}_variation1_fixed/PCN_{setting}_variation1_fixed_run_{run}.csv",
    "PCN_pbest_seed": "./PCN_pbest_seed/PCN_{setting}_variation1_pbestseed/PCN_{setting}_variation1_pbestseed_run_{run}.csv",
}


def path_for(method, setting, run):
    return PATHS[method].format(setting=setting, run=run)


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
    rng = np.random.default_rng(0)
    per_method = {m: {k: [] for k in ["hv", "igd", "n_solutions"]} for m in METHODS}
    coverage = {(a, b): [] for a in METHODS for b in METHODS if a != b}

    for run in range(1, N_RUNS + 1):
        fronts = {m: load_front(path_for(m, setting, run)) for m in METHODS}

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
        print(f"  {METHOD_LABELS[m]:38s}: N={np.mean(per_method[m]['n_solutions']):5.1f}  HV={np.mean(per_method[m]['hv']):.4f}  IGD={np.mean(per_method[m]['igd']):.4f}")
    c1 = np.nanmean(coverage[("PCN_pbest_seed", "PCN_original_seed")])
    c2 = np.nanmean(coverage[("PCN_original_seed", "PCN_pbest_seed")])
    print(f"  Set coverage: pbest_seed→original={c1:.2f}   original→pbest_seed={c2:.2f}")


if __name__ == "__main__":
    for setting in SETTINGS:
        try:
            run_for_setting(setting)
        except FileNotFoundError as e:
            print(f"\n=============== {setting} ===============\n  [skipped: missing file] {e}")
