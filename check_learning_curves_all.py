"""Compact learning-curve check across multiple (setting, version) run folders,
using the existing log.h5 files (no retraining). Reports HV at ~25/50/75/100%
of the logged step range to see whether training is still improving late or
has already plateaued.

    conda activate pcn
    python check_learning_curves_all.py
"""

import sys

import h5py
import numpy as np


def non_dominated(cost):
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


def check_run(run_dir, label):
    with h5py.File(f"{run_dir}/log.h5", "r") as f:
        steps = f["train/leaves/step"][:]
        arr = f["train/leaves/ndarray"][:]

    all_cost = -arr.reshape(-1, 3)
    all_cost = all_cost[~np.isnan(all_cost).any(axis=1)]
    lo, hi = all_cost.min(axis=0), all_cost.max(axis=0)
    denom = np.where(hi - lo == 0, 1.0, hi - lo)
    ref = np.array([1.1, 1.1, 1.1])
    rng = np.random.default_rng(0)

    n = len(steps)
    checkpoints = [0, n // 4, n // 2, (3 * n) // 4, n - 1]
    hvs = []
    for i in checkpoints:
        snap = -arr[i]
        snap = snap[~np.isnan(snap).any(axis=1)]
        nd = snap[non_dominated(snap)]
        norm_nd = (nd - lo) / denom
        hvs.append(hv_mc(norm_nd, ref, rng=rng))

    pct_labels = ["0%", "25%", "50%", "75%", "100%"]
    hv_str = "  ".join(f"{p}={h:.4f}" for p, h in zip(pct_labels, hvs))
    step_str = f"[{steps[checkpoints[0]]}..{steps[checkpoints[-1]]}]"
    improvement = hvs[-1] - hvs[0]
    print(f"{label:35s} steps{step_str:18s} {hv_str}   Δ(100%-0%)={improvement:+.4f}")


RUNS = [
    ("stat_low | variation2", "PCN/PCN_stat_low_variation2_fixed/2026-07-14_19-43-00_83a3"),
    ("stat_high | variation2", "PCN/PCN_stat_high_variation2_fixed/2026-07-14_23-29-35_748d"),
    ("seasonal | variation2", "PCN/PCN_seasonal_variation2_fixed/2026-07-15_11-41-26_cae3"),
    ("GBM | variation1", "PCN/PCN_GBM_variation1_fixed/2026-07-15_12-40-20_0e5c"),
    ("GBM | variation2", "PCN/PCN_GBM_variation2_fixed/2026-07-15_12-43-22_e260"),
    ("gs | variation2", "PCN/PCN_gs_variation2_fixed/2026-07-15_14-53-47_9bb6"),
]

if __name__ == "__main__":
    for label, run_dir in RUNS:
        try:
            check_run(run_dir, label)
        except Exception as e:
            print(f"{label}: FAILED ({e})")
