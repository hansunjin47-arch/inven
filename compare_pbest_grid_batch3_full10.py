"""Compares Guided PCN pbest-seed baseline against batch-3 compute-neutral
variants (action_noise_scale=0.01/0.05/0.1, max_size=1000),
all at full N_RUNS=10, across all 6 settings.

Run in the `pcn` or `gpils` conda env:

    conda activate pcn
    python compare_pbest_grid_batch3_full10.py
"""

import numpy as np
import pandas as pd

SETTINGS = ["GBM", "stat_low", "stat_medium", "stat_high", "seasonal", "gs"]
N_RUNS = 10

PARAMS = {
    "action_noise_scale=0.01": "action_noise_scale_0.01",
    "action_noise_scale=0.05": "action_noise_scale_0.05",
    "action_noise_scale=0.1": "action_noise_scale_0.1",
    "max_size=1000": "max_size_1000",
}

BASELINE_PATH = "./PCN_pbest_seed/PCN_{setting}_variation2_pbestseed/PCN_{setting}_variation2_pbestseed_run_{run}.csv"
GRID_PATH = "./PCN_pbest_grid_free/PCN_{setting}_variation2_{variant}/PCN_{setting}_variation2_run_{run}.csv"


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


def compare_one(setting, variant):
    baseline_fronts = [load_front(BASELINE_PATH.format(setting=setting, run=r)) for r in range(1, N_RUNS + 1)]
    grid_fronts = [load_front(GRID_PATH.format(setting=setting, variant=variant, run=r)) for r in range(1, N_RUNS + 1)]

    rng = np.random.default_rng(0)
    cov_grid_over_base, cov_base_over_grid = [], []
    hv_base, hv_grid, igd_base, igd_grid, n_base, n_grid = [], [], [], [], [], []

    for i in range(N_RUNS):
        F_base, F_grid = baseline_fronts[i], grid_fronts[i]
        combined = np.vstack([F_base, F_grid])
        lo, hi = combined.min(axis=0), combined.max(axis=0)
        denom = np.where(hi - lo == 0, 1.0, hi - lo)
        norm_base, norm_grid = (F_base - lo) / denom, (F_grid - lo) / denom
        ref_point = np.array([1.1, 1.1, 1.1])
        ref_front_norm = (non_dominated_front(combined) - lo) / denom

        cov_grid_over_base.append(set_coverage(F_grid, F_base))
        cov_base_over_grid.append(set_coverage(F_base, F_grid))
        hv_base.append(compute_hv_mc(norm_base, ref_point, rng=rng))
        hv_grid.append(compute_hv_mc(norm_grid, ref_point, rng=rng))
        igd_base.append(compute_igd(norm_base, ref_front_norm))
        igd_grid.append(compute_igd(norm_grid, ref_front_norm))
        n_base.append(len(F_base))
        n_grid.append(len(F_grid))

    return {
        "hv_base": np.mean(hv_base), "hv_grid": np.mean(hv_grid),
        "igd_base": np.mean(igd_base), "igd_grid": np.mean(igd_grid),
        "n_base": np.mean(n_base), "n_grid": np.mean(n_grid),
        "cov_grid_over_base": np.nanmean(cov_grid_over_base),
        "cov_base_over_grid": np.nanmean(cov_base_over_grid),
    }


if __name__ == "__main__":
    rows = []
    for param_label, variant in PARAMS.items():
        better = mixed = worse = 0
        for setting in SETTINGS:
            try:
                r = compare_one(setting, variant)
            except FileNotFoundError as e:
                print(f"[skip] {param_label} | {setting}: {e}")
                continue
            verdict = "better" if r["hv_grid"] > r["hv_base"] and r["igd_grid"] < r["igd_base"] else (
                "worse" if r["hv_grid"] < r["hv_base"] and r["igd_grid"] > r["igd_base"] else "mixed")
            if verdict == "better":
                better += 1
            elif verdict == "worse":
                worse += 1
            else:
                mixed += 1
            rows.append({
                "param": param_label, "setting": setting,
                "N base": f"{r['n_base']:.1f}", "N grid": f"{r['n_grid']:.1f}",
                "HV base": f"{r['hv_base']:.3f}", "HV grid": f"{r['hv_grid']:.3f}",
                "IGD base": f"{r['igd_base']:.3f}", "IGD grid": f"{r['igd_grid']:.3f}",
                "cov grid>base": f"{r['cov_grid_over_base']:.2f}", "cov base>grid": f"{r['cov_base_over_grid']:.2f}",
                "verdict": verdict,
            })
        print(f"=== {param_label}: better={better} mixed={mixed} worse={worse} ===")

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)
    print()
    print(df.to_string(index=False))
