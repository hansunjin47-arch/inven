"""Compares Guided PCN baseline against the combined-winners configuration
(total_steps=2e6, n_model_updates=200, batch_size=512 -- the 3 grid-search
parameters with the clearest net-positive effect), full N_RUNS=10, across all
6 settings.

Run in the `pcn` or `gpils` conda env:

    conda activate pcn
    python compare_combined_winners_full10.py
"""

import numpy as np
import pandas as pd

SETTINGS = ["GBM", "stat_low", "stat_medium", "stat_high", "seasonal", "gs"]
N_RUNS = 10

BASELINE_PATH = "./PCN_stochastic_objsel_test/PCN_{setting}_variation2_stochobjsel/PCN_{setting}_variation2_stochobjsel_run_{run}.csv"
COMBINED_PATH = "./PCN_combined_winners/PCN_{setting}_variation2_combined/PCN_{setting}_variation2_run_{run}.csv"


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
    baseline_fronts = [load_front(BASELINE_PATH.format(setting=setting, run=r)) for r in range(1, N_RUNS + 1)]
    combined_fronts = [load_front(COMBINED_PATH.format(setting=setting, run=r)) for r in range(1, N_RUNS + 1)]

    rng = np.random.default_rng(0)
    cov_comb_over_base, cov_base_over_comb = [], []
    hv_base, hv_comb, igd_base, igd_comb, n_base, n_comb = [], [], [], [], [], []

    for i in range(N_RUNS):
        F_base, F_comb = baseline_fronts[i], combined_fronts[i]
        combined = np.vstack([F_base, F_comb])
        lo, hi = combined.min(axis=0), combined.max(axis=0)
        denom = np.where(hi - lo == 0, 1.0, hi - lo)
        norm_base, norm_comb = (F_base - lo) / denom, (F_comb - lo) / denom
        ref_point = np.array([1.1, 1.1, 1.1])
        ref_front_norm = (non_dominated_front(combined) - lo) / denom

        cov_comb_over_base.append(set_coverage(F_comb, F_base))
        cov_base_over_comb.append(set_coverage(F_base, F_comb))
        hv_base.append(compute_hv_mc(norm_base, ref_point, rng=rng))
        hv_comb.append(compute_hv_mc(norm_comb, ref_point, rng=rng))
        igd_base.append(compute_igd(norm_base, ref_front_norm))
        igd_comb.append(compute_igd(norm_comb, ref_front_norm))
        n_base.append(len(F_base))
        n_comb.append(len(F_comb))

    return {
        "hv_base": np.mean(hv_base), "hv_comb": np.mean(hv_comb),
        "igd_base": np.mean(igd_base), "igd_comb": np.mean(igd_comb),
        "n_base": np.mean(n_base), "n_comb": np.mean(n_comb),
        "cov_comb_over_base": np.nanmean(cov_comb_over_base),
        "cov_base_over_comb": np.nanmean(cov_base_over_comb),
    }


if __name__ == "__main__":
    rows = []
    for setting in SETTINGS:
        try:
            r = compare_one(setting)
        except FileNotFoundError as e:
            print(f"[skip] {setting}: {e}")
            continue
        verdict = "better" if r["hv_comb"] > r["hv_base"] and r["igd_comb"] < r["igd_base"] else (
            "worse" if r["hv_comb"] < r["hv_base"] and r["igd_comb"] > r["igd_base"] else "mixed")
        rows.append({
            "setting": setting,
            "HV base": f"{r['hv_base']:.3f}", "HV combined": f"{r['hv_comb']:.3f}",
            "IGD base": f"{r['igd_base']:.3f}", "IGD combined": f"{r['igd_comb']:.3f}",
            "n base": f"{r['n_base']:.1f}", "n combined": f"{r['n_comb']:.1f}",
            "cov comb>base": f"{r['cov_comb_over_base']:.2f}", "cov base>comb": f"{r['cov_base_over_comb']:.2f}",
            "verdict": verdict,
        })

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)
    print(df.to_string(index=False))
