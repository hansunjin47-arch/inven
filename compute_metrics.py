"""Computes the paper's reported metrics (Set Coverage, Spacing, Maximum
Spread, mean objective values) plus Hypervolume (HV) and IGD, across all five
methods -- MOPSO, Original PCN, PCN with buffer init. (variation1), Guided PCN
(variation2), and CMODE -- for every demand setting for which all five have
completed 10-run results (stat_low/stat_medium/stat_high/seasonal/GBM; the
real-world case study is not included since CMODE hasn't been run for it).

Formulas follow the paper ("Guided Multi-objective Reinforcement Learning for
Inventory Management") exactly:
  - Set coverage C(A, B): Eq. (14)
  - Spacing S: Eq. (15) (L1 nearest-neighbor distance, NOT the L2 version in
    main.py's own spacing() -- kept separate here to match the paper exactly)
  - Maximum spread D: Eq. (16) (diagonal of the per-objective bounding box,
    NOT main.py's max_spread(), which is the max pairwise distance -- these
    are genuinely different metrics despite the similar name)
  - Mean total cost / stockout frequency / stockout amount: per-run means of
    the returned front, averaged over 10 runs (matches paper's Tables 4-12)

HV and IGD are not in the paper; added here per request. Both are computed in
a per-run, min-max normalized objective space (combining all 5 methods'
fronts for that run to get ideal/nadir), consistent with the normalization
approach already used for GPI-LS's HV/IGD in main.py. HV uses Monte Carlo
integration (pygmo is unavailable in this env -- see gpils_baseline.py's
docstring) with ref_point=[1.1, 1.1, 1.1] in the normalized space. IGD's
reference set is the non-dominated union of all 5 methods' fronts for that
run (no analytical true Pareto front exists for this simulation-based
problem).

Run in the `gpils` conda env (only needs numpy/pandas, but that's the env
this project's other scripts use):

    conda activate gpils
    python compute_metrics.py
"""

import numpy as np
import pandas as pd

METHODS = ["MOPSO", "PCN_original", "PCN_variation1", "PCN_variation2", "CMODE"]
METHOD_LABELS = {
    "MOPSO": "MOPSO",
    "PCN_original": "Original PCN",
    "PCN_variation1": "PCN with buffer init.",
    "PCN_variation2": "Guided PCN",
    "CMODE": "CMODE",
}
SETTINGS = ["stat_low", "stat_medium", "stat_high", "seasonal", "GBM"]
SETTING_LABELS = {
    "stat_low": "Stationary (low variance)",
    "stat_medium": "Stationary (medium variance)",
    "stat_high": "Stationary (high variance)",
    "seasonal": "Seasonal",
    "GBM": "GBM",
}
N_RUNS = 10


def path_for(setting, method, run):
    if method == "CMODE":
        return f"./CMODE/CMODE_{setting}/CMODE_run_{setting}_{run}.csv"

    demand_env, suffix = {
        "stat_low": ("stationary", "_lognormal_3.29,0.23"),
        "stat_medium": ("stationary", ""),
        "stat_high": ("stationary", "_lognormal_3.0,0.8"),
        "seasonal": ("seasonal", ""),
        "GBM": ("GBM", ""),
    }[setting]

    if method == "MOPSO":
        return f"./MOPSO/MOPSO_{demand_env}{suffix}/MOPSO_run_{demand_env}_{run}.csv"
    else:  # PCN_original / PCN_variation1 / PCN_variation2
        variant = method.replace("PCN_", "")  # original / variation1 / variation2
        return f"./PCN/PCN_{demand_env}_{variant}{suffix}/PCN_{demand_env}_{variant}_run_{run}.csv"


def load_front(path):
    df = pd.read_csv(path)
    return df[["TC", "Frequency", "Amount"]].to_numpy(dtype=float)


def dominates(p, q):
    return bool(np.all(p <= q) and np.any(p < q))


def set_coverage(A, B):
    """Eq. (14): fraction of B dominated by at least one solution in A."""
    if len(B) == 0:
        return np.nan
    count = sum(1 for b in B if any(dominates(a, b) for a in A))
    return count / len(B)


def spacing(F):
    """Eq. (15): L1 nearest-neighbor distance uniformity."""
    n = len(F)
    if n < 2:
        return 0.0
    d = np.empty(n)
    for i in range(n):
        dists = np.sum(np.abs(F - F[i]), axis=1)
        dists[i] = np.inf
        d[i] = dists.min()
    dbar = d.mean()
    return float(np.sqrt(np.sum((d - dbar) ** 2) / (n - 1)))


def max_spread(F):
    """Eq. (16): diagonal of the per-objective bounding box."""
    if len(F) == 0:
        return 0.0
    fmax, fmin = F.max(axis=0), F.min(axis=0)
    return float(np.sqrt(np.sum((fmax - fmin) ** 2)))


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
    """Monte Carlo hypervolume (minimization) in [0, ref_point]^3."""
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
    rng = np.random.default_rng(0)
    all_results = {}  # setting -> dict of tables

    for setting in SETTINGS:
        per_method = {m: {k: [] for k in ["spacing", "max_spread", "mean_tc", "mean_freq", "mean_amt", "hv", "igd"]} for m in METHODS}
        coverage = {(a, b): [] for a in METHODS for b in METHODS if a != b}

        for run in range(1, N_RUNS + 1):
            fronts = {m: load_front(path_for(setting, m, run)) for m in METHODS}

            combined = np.vstack(list(fronts.values()))
            lo, hi = combined.min(axis=0), combined.max(axis=0)
            denom = np.where(hi - lo == 0, 1.0, hi - lo)
            norm = {m: (fronts[m] - lo) / denom for m in METHODS}
            ref_point = np.array([1.1, 1.1, 1.1])
            ref_front_norm = (non_dominated_front(combined) - lo) / denom

            for m in METHODS:
                F = fronts[m]
                per_method[m]["spacing"].append(spacing(F))
                per_method[m]["max_spread"].append(max_spread(F))
                per_method[m]["mean_tc"].append(F[:, 0].mean())
                per_method[m]["mean_freq"].append(F[:, 1].mean())
                per_method[m]["mean_amt"].append(F[:, 2].mean())
                per_method[m]["hv"].append(compute_hv_mc(norm[m], ref_point, rng=rng))
                per_method[m]["igd"].append(compute_igd(norm[m], ref_front_norm))

            for a in METHODS:
                for b in METHODS:
                    if a != b:
                        coverage[(a, b)].append(set_coverage(fronts[a], fronts[b]))

        metrics_table = pd.DataFrame(
            {
                METHOD_LABELS[m]: {
                    "Spacing": f"{np.mean(per_method[m]['spacing']):.2f} ± {np.std(per_method[m]['spacing']):.2f}",
                    "Maximum spread": f"{np.mean(per_method[m]['max_spread']):.2f} ± {np.std(per_method[m]['max_spread']):.2f}",
                    "Mean total cost (1e3)": f"{np.mean(per_method[m]['mean_tc']):.2f} ± {np.std(per_method[m]['mean_tc']):.2f}",
                    "Mean stockout frequency": f"{np.mean(per_method[m]['mean_freq']):.2f} ± {np.std(per_method[m]['mean_freq']):.2f}",
                    "Mean stockout amount": f"{np.mean(per_method[m]['mean_amt']):.2f} ± {np.std(per_method[m]['mean_amt']):.2f}",
                    "HV (normalized)": f"{np.mean(per_method[m]['hv']):.4f} ± {np.std(per_method[m]['hv']):.4f}",
                    "IGD (normalized)": f"{np.mean(per_method[m]['igd']):.4f} ± {np.std(per_method[m]['igd']):.4f}",
                }
                for m in METHODS
            }
        )

        coverage_table = pd.DataFrame(
            index=[METHOD_LABELS[m] for m in METHODS],
            columns=[METHOD_LABELS[m] for m in METHODS],
            dtype=object,
        )
        for a in METHODS:
            for b in METHODS:
                if a == b:
                    coverage_table.loc[METHOD_LABELS[a], METHOD_LABELS[b]] = "-"
                else:
                    vals = coverage[(a, b)]
                    coverage_table.loc[METHOD_LABELS[a], METHOD_LABELS[b]] = f"{np.mean(vals):.2f} ± {np.std(vals):.2f}"

        all_results[setting] = {"metrics": metrics_table, "coverage": coverage_table}

    return all_results


if __name__ == "__main__":
    results = main()
    for setting in SETTINGS:
        print(f"\n{'=' * 20} {SETTING_LABELS[setting]} {'=' * 20}")
        print("\n--- Set Coverage C(A, B): rows=A, cols=B ---")
        print(results[setting]["coverage"].to_string())
        print("\n--- Metrics ---")
        print(results[setting]["metrics"].to_string())
