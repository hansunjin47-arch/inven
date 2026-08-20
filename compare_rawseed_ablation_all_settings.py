"""Compares Guided PCN (variation2, pbest-fixed MOPSO seed) WITH vs WITHOUT
raw_seed_episode (one noise-free episode per MOPSO seed solution, prepended
to Algorithm 2's usual n_repeat noisy episodes). Both conditions use the same
pbest-fixed MOPSO archive and training hyperparameters -- isolates the effect
of raw_seed_episode alone, across all 6 settings, full N_RUNS=10.

Run in the `pcn` or `gpils` conda env:

    conda activate pcn
    python compare_rawseed_ablation_all_settings.py
"""

import numpy as np
import pandas as pd

SETTINGS = ["GBM", "stat_low", "stat_medium", "stat_high", "seasonal", "gs"]
N_RUNS = 10

WITHOUT_PATH = "./PCN_pbest_seed/PCN_{setting}_variation2_pbestseed/PCN_{setting}_variation2_pbestseed_run_{run}.csv"
WITH_PATH = "./PCN_pbest_seed_rawseed/PCN_{setting}_variation2_pbestseed_rawseed/PCN_{setting}_variation2_pbestseed_rawseed_run_{run}.csv"


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


def compare_one(setting):
    without_fronts = [load_front(WITHOUT_PATH.format(setting=setting, run=r)) for r in range(1, N_RUNS + 1)]
    with_fronts = [load_front(WITH_PATH.format(setting=setting, run=r)) for r in range(1, N_RUNS + 1)]

    rng = np.random.default_rng(0)
    cov_with_over_without, cov_without_over_with = [], []
    hv_without, hv_with, igd_without, igd_with, n_without, n_with = [], [], [], [], [], []

    for i in range(N_RUNS):
        F_without, F_with = without_fronts[i], with_fronts[i]
        combined = np.vstack([F_without, F_with])
        lo, hi = combined.min(axis=0), combined.max(axis=0)
        denom = np.where(hi - lo == 0, 1.0, hi - lo)
        norm_without, norm_with = (F_without - lo) / denom, (F_with - lo) / denom
        ref_point = np.array([1.1, 1.1, 1.1])
        ref_front_norm = (non_dominated_front(combined) - lo) / denom

        cov_with_over_without.append(set_coverage(F_with, F_without))
        cov_without_over_with.append(set_coverage(F_without, F_with))
        hv_without.append(compute_hv_mc(norm_without, ref_point, rng=rng))
        hv_with.append(compute_hv_mc(norm_with, ref_point, rng=rng))
        igd_without.append(compute_igd(norm_without, ref_front_norm))
        igd_with.append(compute_igd(norm_with, ref_front_norm))
        n_without.append(len(F_without))
        n_with.append(len(F_with))

    return {
        "hv_without": np.mean(hv_without), "hv_with": np.mean(hv_with),
        "igd_without": np.mean(igd_without), "igd_with": np.mean(igd_with),
        "n_without": np.mean(n_without), "n_with": np.mean(n_with),
        "cov_with_over_without": np.nanmean(cov_with_over_without),
        "cov_without_over_with": np.nanmean(cov_without_over_with),
    }


if __name__ == "__main__":
    rows = []
    for setting in SETTINGS:
        try:
            r = compare_one(setting)
        except FileNotFoundError as e:
            print(f"[skip] {setting}: {e}")
            continue
        verdict = "better" if r["hv_with"] > r["hv_without"] and r["igd_with"] < r["igd_without"] else (
            "worse" if r["hv_with"] < r["hv_without"] and r["igd_with"] > r["igd_without"] else "mixed")
        rows.append({
            "setting": setting,
            "N w/o": f"{r['n_without']:.1f}", "N w/": f"{r['n_with']:.1f}",
            "HV w/o": f"{r['hv_without']:.3f}", "HV w/": f"{r['hv_with']:.3f}",
            "IGD w/o": f"{r['igd_without']:.3f}", "IGD w/": f"{r['igd_with']:.3f}",
            "cov with>w/o": f"{r['cov_with_over_without']:.2f}", "cov w/o>with": f"{r['cov_without_over_with']:.2f}",
            "verdict": verdict,
        })

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)
    print("raw_seed_episode=True vs False (both pbest-fixed MOPSO seed, variation2), N_RUNS=10")
    print(df.to_string(index=False))
