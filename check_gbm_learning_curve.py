"""Uses the per-run log.h5 (already saved during training -- 224 snapshots of
the non-dominated front, logged every ~3650 steps from step ~185k to ~999k)
to check whether Guided PCN's Pareto front on GBM is still improving near the
end of training (1e6 steps) or has already plateaued. No retraining needed --
this reads existing logs.

    conda activate pcn
    python check_gbm_learning_curve.py <path-to-run-folder>
"""

import sys

import h5py
import numpy as np


def non_dominated(cost):
    """cost: (N,3) array, minimization. Returns boolean mask of non-dominated rows."""
    n = len(cost)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j and np.all(cost[j] <= cost[i]) and np.any(cost[j] < cost[i]):
                keep[i] = False
                break
    return keep


def hv_mc(front, ref, n_samples=20_000, rng=None):
    rng = rng or np.random.default_rng(0)
    if len(front) == 0:
        return 0.0
    samples = rng.uniform(low=0.0, high=ref, size=(n_samples, len(ref)))
    dominated = np.any(np.all(front[:, None, :] <= samples[None, :, :], axis=2), axis=0)
    return float(dominated.mean() * np.prod(ref))


def main(run_dir):
    with h5py.File(f"{run_dir}/log.h5", "r") as f:
        steps = f["train/leaves/step"][:]
        arr = f["train/leaves/ndarray"][:]

    # convert reward (-cost,-freq,-amt) to cost-space
    all_cost = -arr.reshape(-1, 3)
    all_cost = all_cost[~np.isnan(all_cost).any(axis=1)]
    lo = all_cost.min(axis=0)
    hi = all_cost.max(axis=0)
    denom = np.where(hi - lo == 0, 1.0, hi - lo)
    ref = np.array([1.1, 1.1, 1.1])

    rng = np.random.default_rng(0)
    print(f"{'step':>10} {'n_leaves':>9} {'n_nondom':>9} {'mean_cost':>10} {'HV':>8}")
    for i in range(0, len(steps), 8):  # subsample snapshots for a readable printout
        snap = -arr[i]
        snap = snap[~np.isnan(snap).any(axis=1)]
        if len(snap) == 0:
            continue
        nd_mask = non_dominated(snap)
        nd = snap[nd_mask]
        norm_nd = (nd - lo) / denom
        hv = hv_mc(norm_nd, ref, rng=rng)
        print(f"{steps[i]:>10} {len(snap):>9} {nd_mask.sum():>9} {nd[:,0].mean():>10.3f} {hv:>8.4f}")


if __name__ == "__main__":
    run_dir = sys.argv[1] if len(sys.argv) > 1 else "PCN/PCN_GBM_variation2_fixed/2026-07-15_12-43-22_e260"
    main(run_dir)
