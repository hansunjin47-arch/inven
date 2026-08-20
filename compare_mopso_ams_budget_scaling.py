"""Compares MOPSO-AMS at 3 evaluation budgets -- matched (1x, 200x100=20k),
2x (200x200=40k), paper params (7.5x, 150x1000=150k) -- against Guided PCN
(pbest-fixed seed), across all 6 settings. Checks whether 2x performance
falls between 1x and 7.5x (budget monotonicity), and where Guided PCN
stands relative to each.

Run in the `pcn` or `gpils` conda env:

    conda activate pcn
    python compare_mopso_ams_budget_scaling.py
"""

import numpy as np
import pandas as pd

SETTINGS = ["GBM", "stat_low", "stat_medium", "stat_high", "seasonal", "gs"]
N_RUNS = 10

METHODS = ["MOPSO_AMS_1x", "MOPSO_AMS_2x", "MOPSO_AMS_7.5x", "Guided_PCN"]
METHOD_LABELS = {
    "MOPSO_AMS_1x": "MOPSO-AMS (1x, matched)",
    "MOPSO_AMS_2x": "MOPSO-AMS (2x)",
    "MOPSO_AMS_7.5x": "MOPSO-AMS (7.5x, paper)",
    "Guided_PCN": "Guided PCN (pbest seed)",
}

MOPSO_AMS_1X_PATH = {
    "GBM": "./MOPSO_AMS/MOPSO_AMS_GBM/MOPSO_AMS_run_GBM_{run}.csv",
    "stat_low": "./MOPSO_AMS/MOPSO_AMS_stat_low/MOPSO_AMS_run_stat_low_{run}.csv",
    "stat_medium": "./MOPSO_AMS/MOPSO_AMS_stat_medium/MOPSO_AMS_run_stat_medium_{run}.csv",
    "stat_high": "./MOPSO_AMS/MOPSO_AMS_stat_high/MOPSO_AMS_run_stat_high_{run}.csv",
    "seasonal": "./MOPSO_AMS/MOPSO_AMS_seasonal/MOPSO_AMS_run_seasonal_{run}.csv",
    "gs": "./MOPSO_AMS/MOPSO_AMS_gs/MOPSO_AMS_run_gs_test1_{run}.csv",
}
MOPSO_AMS_2X_PATH = {
    "GBM": "./MOPSO_AMS_2x/MOPSO_AMS_GBM/MOPSO_AMS_run_GBM_{run}.csv",
    "stat_low": "./MOPSO_AMS_2x/MOPSO_AMS_stat_low/MOPSO_AMS_run_stat_low_{run}.csv",
    "stat_medium": "./MOPSO_AMS_2x/MOPSO_AMS_stat_medium/MOPSO_AMS_run_stat_medium_{run}.csv",
    "stat_high": "./MOPSO_AMS_2x/MOPSO_AMS_stat_high/MOPSO_AMS_run_stat_high_{run}.csv",
    "seasonal": "./MOPSO_AMS_2x/MOPSO_AMS_seasonal/MOPSO_AMS_run_seasonal_{run}.csv",
    "gs": "./MOPSO_AMS_2x/MOPSO_AMS_gs/MOPSO_AMS_run_gs_test1_{run}.csv",
}
MOPSO_AMS_7_5X_PATH = {
    "GBM": "./MOPSO_AMS_paperparams/MOPSO_AMS_GBM/MOPSO_AMS_run_GBM_{run}.csv",
    "stat_low": "./MOPSO_AMS_paperparams/MOPSO_AMS_stat_low/MOPSO_AMS_run_stat_low_{run}.csv",
    "stat_medium": "./MOPSO_AMS_paperparams/MOPSO_AMS_stat_medium/MOPSO_AMS_run_stat_medium_{run}.csv",
    "stat_high": "./MOPSO_AMS_paperparams/MOPSO_AMS_stat_high/MOPSO_AMS_run_stat_high_{run}.csv",
    "seasonal": "./MOPSO_AMS_paperparams/MOPSO_AMS_seasonal/MOPSO_AMS_run_seasonal_{run}.csv",
    "gs": "./MOPSO_AMS_paperparams/MOPSO_AMS_gs/MOPSO_AMS_run_gs_test1_{run}.csv",
}


def paths_for_setting(setting):
    return {
        "MOPSO_AMS_1x": MOPSO_AMS_1X_PATH[setting],
        "MOPSO_AMS_2x": MOPSO_AMS_2X_PATH[setting],
        "MOPSO_AMS_7.5x": MOPSO_AMS_7_5X_PATH[setting],
        "Guided_PCN": f"./PCN_pbest_seed/PCN_{setting}_variation2_pbestseed/PCN_{setting}_variation2_pbestseed_run_{{run}}.csv",
    }


def load_front(path):
    df = pd.read_csv(path)
    return df[["TC", "Frequency", "Amount"]].to_numpy(dtype=float)


def dominates(p, q):
    return bool(np.all(p <= q) and np.any(p < q))


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
    per_method = {m: {"hv": [], "igd": [], "n": []} for m in METHODS}

    for run in range(1, N_RUNS + 1):
        fronts = {m: load_front(paths[m].format(run=run)) for m in METHODS}
        combined = np.vstack(list(fronts.values()))
        lo, hi = combined.min(axis=0), combined.max(axis=0)
        denom = np.where(hi - lo == 0, 1.0, hi - lo)
        norm = {m: (fronts[m] - lo) / denom for m in METHODS}
        ref_point = np.array([1.1, 1.1, 1.1])
        ref_front_norm = (non_dominated_front(combined) - lo) / denom

        for m in METHODS:
            per_method[m]["hv"].append(compute_hv_mc(norm[m], ref_point, rng=rng))
            per_method[m]["igd"].append(compute_igd(norm[m], ref_front_norm))
            per_method[m]["n"].append(len(fronts[m]))

    return {
        m: {
            "HV": np.mean(per_method[m]["hv"]),
            "IGD": np.mean(per_method[m]["igd"]),
            "N": np.mean(per_method[m]["n"]),
        }
        for m in METHODS
    }


if __name__ == "__main__":
    rows = []
    for setting in SETTINGS:
        try:
            res = run_for_setting(setting)
        except FileNotFoundError as e:
            print(f"[skip] {setting}: {e}")
            continue
        for m in METHODS:
            rows.append({
                "setting": setting, "method": METHOD_LABELS[m],
                "N": f"{res[m]['N']:.1f}", "HV": f"{res[m]['HV']:.3f}", "IGD": f"{res[m]['IGD']:.3f}",
            })
        # monotonicity check
        hv1, hv2, hv75 = res["MOPSO_AMS_1x"]["HV"], res["MOPSO_AMS_2x"]["HV"], res["MOPSO_AMS_7.5x"]["HV"]
        igd1, igd2, igd75 = res["MOPSO_AMS_1x"]["IGD"], res["MOPSO_AMS_2x"]["IGD"], res["MOPSO_AMS_7.5x"]["IGD"]
        hv_between = min(hv1, hv75) <= hv2 <= max(hv1, hv75)
        igd_between = min(igd1, igd75) <= igd2 <= max(igd1, igd75)
        print(f"[{setting}] HV monotonic (1x/2x/7.5x = {hv1:.3f}/{hv2:.3f}/{hv75:.3f}): {'YES' if hv_between else 'NO'}   "
              f"IGD monotonic ({igd1:.3f}/{igd2:.3f}/{igd75:.3f}): {'YES' if igd_between else 'NO'}")

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)
    print()
    print(df.to_string(index=False))
