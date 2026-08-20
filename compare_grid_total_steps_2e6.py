"""Compares Guided PCN (variation2, softmax obj. selection) baseline
(max_size=500, N_RUNS_BASELINE=10) against the total_steps=2e6 grid-search
screening cell (N_RUNS_SCREEN=3), across all 6 settings.

Run in the `pcn` or `gpils` conda env:

    conda activate pcn
    python compare_grid_total_steps_2000000.0.py
"""

import numpy as np
import pandas as pd

SETTINGS = ["GBM", "stat_low", "stat_medium", "stat_high", "seasonal", "gs"]
N_RUNS_BASELINE = 10
N_RUNS_SCREEN = 3

BASELINE_PATH = "./PCN_stochastic_objsel_test/PCN_{setting}_variation2_stochobjsel/PCN_{setting}_variation2_stochobjsel_run_{run}.csv"
GRID_PATH = "./PCN_grid_search/PCN_{setting}_variation2_total_steps_2000000.0/PCN_{setting}_variation2_run_{run}.csv"


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


def main():
    global SETTINGS
    SETTINGS = ["GBM"]
    for setting in SETTINGS:
        print(f"\n=============== {setting} ===============")
        baseline_fronts = [load_front(BASELINE_PATH.format(setting=setting, run=r)) for r in range(1, N_RUNS_BASELINE + 1)]
        grid_fronts = [load_front(GRID_PATH.format(setting=setting, run=r)) for r in range(1, N_RUNS_SCREEN + 1)]

        # use only the first N_RUNS_SCREEN baseline runs (matched seeds 1..3) for apples-to-apples,
        # plus report full-10 baseline stats too for context
        baseline_matched = baseline_fronts[:N_RUNS_SCREEN]

        rng = np.random.default_rng(0)
        cov_grid_over_baseline = []
        cov_baseline_over_grid = []
        n_baseline, n_grid = [], []
        hv_baseline, hv_grid = [], []
        igd_baseline, igd_grid = [], []

        for i in range(N_RUNS_SCREEN):
            F_base = baseline_matched[i]
            F_grid = grid_fronts[i]
            combined = np.vstack([F_base, F_grid])
            lo, hi = combined.min(axis=0), combined.max(axis=0)
            denom = np.where(hi - lo == 0, 1.0, hi - lo)
            norm_base = (F_base - lo) / denom
            norm_grid = (F_grid - lo) / denom
            ref_point = np.array([1.1, 1.1, 1.1])
            ref_front_norm = (non_dominated_front(combined) - lo) / denom

            cov_grid_over_baseline.append(set_coverage(F_grid, F_base))
            cov_baseline_over_grid.append(set_coverage(F_base, F_grid))
            n_baseline.append(len(F_base))
            n_grid.append(len(F_grid))
            hv_baseline.append(compute_hv_mc(norm_base, ref_point, rng=rng))
            hv_grid.append(compute_hv_mc(norm_grid, ref_point, rng=rng))
            igd_baseline.append(compute_igd(norm_base, ref_front_norm))
            igd_grid.append(compute_igd(norm_grid, ref_front_norm))

        print(f"  N solutions      : baseline(max_size=500, seeds1-3)={np.mean(n_baseline):.1f}±{np.std(n_baseline):.1f}   total_steps=2e6={np.mean(n_grid):.1f}±{np.std(n_grid):.1f}")
        print(f"  HV (normalized)  : baseline={np.mean(hv_baseline):.4f}±{np.std(hv_baseline):.4f}   total_steps=2e6={np.mean(hv_grid):.4f}±{np.std(hv_grid):.4f}")
        print(f"  IGD (normalized) : baseline={np.mean(igd_baseline):.4f}±{np.std(igd_baseline):.4f}   total_steps=2e6={np.mean(igd_grid):.4f}±{np.std(igd_grid):.4f}")
        print(f"  Set coverage     : total_steps=2e6→baseline={np.nanmean(cov_grid_over_baseline):.2f}   baseline→total_steps=2e6={np.nanmean(cov_baseline_over_grid):.2f}")

        # full-10 baseline context
        n_baseline_full = [len(f) for f in baseline_fronts]
        print(f"  (baseline full-10 N solutions: {np.mean(n_baseline_full):.1f}±{np.std(n_baseline_full):.1f})")


if __name__ == "__main__":
    main()
