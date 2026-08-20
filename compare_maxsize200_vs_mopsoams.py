"""Checks whether Guided PCN (variation2) at max_size=200 still beats
MOPSO-AMS (2x budget) the way the max_size=500 baseline did.

Run in the `pcn` or `gpils` conda env:

    conda activate pcn
    python compare_maxsize200_vs_mopsoams.py
"""

import numpy as np
import pandas as pd

SETTINGS = ["GBM", "stat_low", "stat_medium", "stat_high", "seasonal", "gs"]
N_RUNS = 10

MOPSO_AMS_PATH = {
    "GBM": "./MOPSO_AMS_2x/MOPSO_AMS_GBM/MOPSO_AMS_run_GBM_{run}.csv",
    "stat_low": "./MOPSO_AMS_2x/MOPSO_AMS_stat_low/MOPSO_AMS_run_stat_low_{run}.csv",
    "stat_medium": "./MOPSO_AMS_2x/MOPSO_AMS_stat_medium/MOPSO_AMS_run_stat_medium_{run}.csv",
    "stat_high": "./MOPSO_AMS_2x/MOPSO_AMS_stat_high/MOPSO_AMS_run_stat_high_{run}.csv",
    "seasonal": "./MOPSO_AMS_2x/MOPSO_AMS_seasonal/MOPSO_AMS_run_seasonal_{run}.csv",
    "gs": "./MOPSO_AMS_2x/MOPSO_AMS_gs/MOPSO_AMS_run_gs_test1_{run}.csv",
}
PCN_MAXSIZE200_PATH = "./PCN_pbest_grid_free/PCN_{setting}_variation2_max_size_200/PCN_{setting}_variation2_run_{run}.csv"


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
    f_pcn = [load_front(PCN_MAXSIZE200_PATH.format(setting=setting, run=r)) for r in range(1, N_RUNS + 1)]
    f_ams = [load_front(MOPSO_AMS_PATH[setting].format(run=r)) for r in range(1, N_RUNS + 1)]

    rng = np.random.default_rng(0)
    hv_pcn, hv_ams, igd_pcn, igd_ams, n_pcn, n_ams = [], [], [], [], [], []
    cov_pcn_over_ams, cov_ams_over_pcn = [], []

    for i in range(N_RUNS):
        F1, F2 = f_pcn[i], f_ams[i]
        combined = np.vstack([F1, F2])
        lo, hi = combined.min(axis=0), combined.max(axis=0)
        denom = np.where(hi - lo == 0, 1.0, hi - lo)
        norm1, norm2 = (F1 - lo) / denom, (F2 - lo) / denom
        ref_point = np.array([1.1, 1.1, 1.1])
        ref_front_norm = (non_dominated_front(combined) - lo) / denom

        cov_pcn_over_ams.append(set_coverage(F1, F2))
        cov_ams_over_pcn.append(set_coverage(F2, F1))
        hv_pcn.append(compute_hv_mc(norm1, ref_point, rng=rng))
        hv_ams.append(compute_hv_mc(norm2, ref_point, rng=rng))
        igd_pcn.append(compute_igd(norm1, ref_front_norm))
        igd_ams.append(compute_igd(norm2, ref_front_norm))
        n_pcn.append(len(F1))
        n_ams.append(len(F2))

    return {
        "hv_pcn": np.mean(hv_pcn), "hv_ams": np.mean(hv_ams),
        "igd_pcn": np.mean(igd_pcn), "igd_ams": np.mean(igd_ams),
        "n_pcn": np.mean(n_pcn), "n_ams": np.mean(n_ams),
        "cov_pcn_over_ams": np.nanmean(cov_pcn_over_ams),
        "cov_ams_over_pcn": np.nanmean(cov_ams_over_pcn),
    }


if __name__ == "__main__":
    rows = []
    for setting in SETTINGS:
        r = compare_one(setting)
        verdict = "PCN wins" if r["hv_pcn"] > r["hv_ams"] and r["igd_pcn"] < r["igd_ams"] else (
            "MOPSO-AMS wins" if r["hv_ams"] > r["hv_pcn"] and r["igd_ams"] < r["igd_pcn"] else "mixed")
        rows.append({
            "setting": setting,
            "N PCN(200)": f"{r['n_pcn']:.1f}", "N AMS": f"{r['n_ams']:.1f}",
            "HV PCN(200)": f"{r['hv_pcn']:.3f}", "HV AMS": f"{r['hv_ams']:.3f}",
            "IGD PCN(200)": f"{r['igd_pcn']:.3f}", "IGD AMS": f"{r['igd_ams']:.3f}",
            "cov PCN>AMS": f"{r['cov_pcn_over_ams']:.2f}", "cov AMS>PCN": f"{r['cov_ams_over_pcn']:.2f}",
            "verdict": verdict,
        })

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
