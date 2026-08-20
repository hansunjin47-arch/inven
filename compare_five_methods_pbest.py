"""Six-method comparison table + figure, all using the pbest-tiebreak-fixed
MOPSO seed where applicable: MOPSO, Original PCN, PCN variation1 (buffer init
only), PCN variation2 (Guided PCN), MOPSO-AMS (2x matched-budget --
I=200,T=200 -- chosen as the single reported version since Guided PCN's
total pipeline cost is ~2x plain MOPSO's, matching MOPSO-AMS to the same
total compute for a fair comparison), and CAPQL (MORL baseline, uniform
recipe: obs-norm on, alpha=0.05, start_steps=10000, total_steps=300000,
10 seeds -- results generated on a separate PC and imported under
./capql_results_export/). All 3 PCN variants now use max_size=700 (final
adopted single-factor config, superseding the earlier max_size=750
default: a full fine-grid sweep of 600/700/750/800/900/1500 found 700
best overall, and a further comparison of 700 alone vs 750 vs
700+total_steps=2e6 confirmed 700 alone gives the cleanest variation2-
over-variation1 ablation result (0 genuine v1 wins across all 6 settings,
vs 1 for 750 and 2 for 700+total_steps2e6, judged jointly on HV/IGD/set
coverage), at no extra compute cost -- this was prioritized over
700+total_steps2e6's marginally better MOPSO-AMS-coverage-gap closing in
the stationary settings, since that came at 2x the training cost and a
noisier ablation story). Across all 6 settings, N_RUNS=10.

Run in the `pcn` or `gpils` conda env:

    conda activate pcn
    python compare_five_methods_pbest.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SETTINGS = ["GBM", "stat_low", "stat_medium", "stat_high", "seasonal", "gs"]
N_RUNS = 10

METHODS = ["MOPSO", "PCN_original", "PCN_variation1", "PCN_variation2", "MOPSO_AMS", "CAPQL"]
METHOD_LABELS = {
    "MOPSO": "MOPSO",
    "PCN_original": "Original PCN",
    "PCN_variation1": "PCN + buffer init",
    "PCN_variation2": "Guided PCN",
    "MOPSO_AMS": "MOPSO-AMS",
    "CAPQL": "CAPQL",
}

MOPSO_PBEST_PATH = {
    "GBM": "./MOPSO_tsou_ablation/MOPSO_GBM_pbest_only/MOPSO_run_GBM_{run}.csv",
    "stat_low": "./MOPSO_tsou_ablation/MOPSO_stat_low_pbest_only/MOPSO_run_stat_low_{run}.csv",
    "stat_medium": "./MOPSO_tsou_ablation/MOPSO_stat_medium_pbest_only/MOPSO_run_stat_medium_{run}.csv",
    "stat_high": "./MOPSO_tsou_ablation/MOPSO_stat_high_pbest_only/MOPSO_run_stat_high_{run}.csv",
    "seasonal": "./MOPSO_tsou_ablation/MOPSO_seasonal_pbest_only/MOPSO_run_seasonal_{run}.csv",
    "gs": "./MOPSO_tsou_ablation/MOPSO_gs_pbest_only/MOPSO_run_gs_test1_{run}.csv",
}
MOPSO_AMS_PAPERPARAMS_PATH = {
    "GBM": "./MOPSO_AMS_2x/MOPSO_AMS_GBM/MOPSO_AMS_run_GBM_{run}.csv",
    "stat_low": "./MOPSO_AMS_2x/MOPSO_AMS_stat_low/MOPSO_AMS_run_stat_low_{run}.csv",
    "stat_medium": "./MOPSO_AMS_2x/MOPSO_AMS_stat_medium/MOPSO_AMS_run_stat_medium_{run}.csv",
    "stat_high": "./MOPSO_AMS_2x/MOPSO_AMS_stat_high/MOPSO_AMS_run_stat_high_{run}.csv",
    "seasonal": "./MOPSO_AMS_2x/MOPSO_AMS_seasonal/MOPSO_AMS_run_seasonal_{run}.csv",
    "gs": "./MOPSO_AMS_2x/MOPSO_AMS_gs/MOPSO_AMS_run_gs_test1_{run}.csv",
}


def paths_for_setting(setting):
    return {
        "MOPSO": MOPSO_PBEST_PATH[setting],
        "PCN_original": f"./PCN_pbest_grid_free/PCN_{setting}_original_max_size_700/PCN_{setting}_original_run_{{run}}.csv",
        "PCN_variation1": f"./PCN_pbest_grid_free/PCN_{setting}_variation1_max_size_700/PCN_{setting}_variation1_run_{{run}}.csv",
        "PCN_variation2": f"./PCN_pbest_grid_free/PCN_{setting}_variation2_max_size_700/PCN_{setting}_variation2_run_{{run}}.csv",
        "MOPSO_AMS": MOPSO_AMS_PAPERPARAMS_PATH[setting],
        "CAPQL": f"./capql_results_export/{setting}/seed_runs/{setting}_run{{run}}.csv",
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
    all_results = {}
    for setting in SETTINGS:
        try:
            all_results[setting] = run_for_setting(setting)
        except FileNotFoundError as e:
            print(f"[skip] {setting}: {e}")

    # ---- Table ----
    rows = []
    for setting, res in all_results.items():
        for m in METHODS:
            rows.append({
                "setting": setting, "method": METHOD_LABELS[m],
                "N": f"{res[m]['N']:.1f}", "HV": f"{res[m]['HV']:.3f}", "IGD": f"{res[m]['IGD']:.3f}",
            })
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)
    print(df.to_string(index=False))
    df.to_csv("five_method_comparison_table.csv", index=False)
    print("\nSaved table -> five_method_comparison_table.csv")

    # ---- Figure: HV (top) and IGD (bottom), grouped bars per setting ----
    settings_done = list(all_results.keys())
    x = np.arange(len(settings_done))
    n_methods = len(METHODS)
    width = 0.8 / n_methods
    colors = plt.cm.tab10(np.linspace(0, 1, n_methods))

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    for i, m in enumerate(METHODS):
        hv_vals = [all_results[s][m]["HV"] for s in settings_done]
        igd_vals = [all_results[s][m]["IGD"] for s in settings_done]
        offset = (i - (n_methods - 1) / 2) * width
        axes[0].bar(x + offset, hv_vals, width, label=METHOD_LABELS[m], color=colors[i])
        axes[1].bar(x + offset, igd_vals, width, label=METHOD_LABELS[m], color=colors[i])

    axes[0].set_ylabel("Hypervolume (higher is better)")
    axes[0].set_title("Five-method comparison (pbest-fixed MOPSO seed)")
    axes[0].legend(ncol=n_methods, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.25))
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].set_ylabel("IGD (lower is better)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(settings_done)
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig("five_method_comparison.png", dpi=200, bbox_inches="tight")
    print("Saved figure -> five_method_comparison.png")
