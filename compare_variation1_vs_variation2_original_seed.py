"""Checks whether variation1 already beat variation2 in some settings BEFORE
the MOPSO pbest-tiebreak fix -- i.e. using the ORIGINAL (non-pbest) MOPSO
seed for both, at their original lr=1e-4 baseline configs. This tells us
whether the variation1-sometimes-beats-variation2 pattern is a pre-existing
property of the guided-exploration mechanism (Algorithm 3) itself, or
something introduced/exposed specifically by the pbest seed change.

Run in the `pcn` or `gpils` conda env:

    conda activate pcn
    python compare_variation1_vs_variation2_original_seed.py
"""

import numpy as np
import pandas as pd

SETTINGS = ["GBM", "stat_low", "stat_medium", "stat_high", "seasonal", "gs"]
N_RUNS = 10

PATHS = {
    "variation1": "./PCN_fixed/PCN_{setting}_variation1_fixed/PCN_{setting}_variation1_fixed_run_{run}.csv",
    "variation2": "./PCN_stochastic_objsel_test/PCN_{setting}_variation2_stochobjsel/PCN_{setting}_variation2_stochobjsel_run_{run}.csv",
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


def compare_one(setting):
    f1 = [load_front(PATHS["variation1"].format(setting=setting, run=r)) for r in range(1, N_RUNS + 1)]
    f2 = [load_front(PATHS["variation2"].format(setting=setting, run=r)) for r in range(1, N_RUNS + 1)]

    rng = np.random.default_rng(0)
    cov_v1_over_v2, cov_v2_over_v1 = [], []
    hv1, hv2, igd1, igd2, n1, n2 = [], [], [], [], [], []

    for i in range(N_RUNS):
        F1, F2 = f1[i], f2[i]
        combined = np.vstack([F1, F2])
        lo, hi = combined.min(axis=0), combined.max(axis=0)
        denom = np.where(hi - lo == 0, 1.0, hi - lo)
        norm1, norm2 = (F1 - lo) / denom, (F2 - lo) / denom
        ref_point = np.array([1.1, 1.1, 1.1])
        ref_front_norm = (non_dominated_front(combined) - lo) / denom

        cov_v1_over_v2.append(set_coverage(F1, F2))
        cov_v2_over_v1.append(set_coverage(F2, F1))
        hv1.append(compute_hv_mc(norm1, ref_point, rng=rng))
        hv2.append(compute_hv_mc(norm2, ref_point, rng=rng))
        igd1.append(compute_igd(norm1, ref_front_norm))
        igd2.append(compute_igd(norm2, ref_front_norm))
        n1.append(len(F1))
        n2.append(len(F2))

    return {
        "hv1": np.mean(hv1), "hv2": np.mean(hv2),
        "igd1": np.mean(igd1), "igd2": np.mean(igd2),
        "n1": np.mean(n1), "n2": np.mean(n2),
        "cov_v1_over_v2": np.nanmean(cov_v1_over_v2),
        "cov_v2_over_v1": np.nanmean(cov_v2_over_v1),
    }


if __name__ == "__main__":
    rows = []
    for setting in SETTINGS:
        try:
            r = compare_one(setting)
        except FileNotFoundError as e:
            print(f"[skip] {setting}: {e}")
            continue
        verdict = "v2 wins" if r["hv2"] > r["hv1"] and r["igd2"] < r["igd1"] else (
            "v1 wins" if r["hv1"] > r["hv2"] and r["igd1"] < r["igd2"] else "mixed")
        rows.append({
            "setting": setting,
            "N v1": f"{r['n1']:.1f}", "N v2": f"{r['n2']:.1f}",
            "HV v1": f"{r['hv1']:.3f}", "HV v2": f"{r['hv2']:.3f}",
            "IGD v1": f"{r['igd1']:.3f}", "IGD v2": f"{r['igd2']:.3f}",
            "cov v1>v2": f"{r['cov_v1_over_v2']:.2f}", "cov v2>v1": f"{r['cov_v2_over_v1']:.2f}",
            "verdict": verdict,
        })

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
