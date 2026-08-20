import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _sub in ("baselines", "envs", "models", "utils", "experiments"):
    _p = _os.path.join(_ROOT, _sub)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
_os.chdir(_ROOT)

"""Regenerates all 12 paper result tables (6 set-coverage + 6 metrics) now
that PCN's adopted config has switched from max_size=750 to max_size=700
(see compare_five_methods_pbest.py's updated docstring/paths_for_setting()).
Methodology mirrors compare_five_methods_pbest.py exactly (pooled 6-method
min-max normalization, ref point (1.1,1.1,1.1)) plus adds set coverage,
spacing, maximum spread, and per-objective means, matching the paper's
Section 5.1 metric definitions.

Run in the `pcn` conda env:
    conda run -n pcn python _gen_paper_tables_700.py
"""

import numpy as np
import pandas as pd

from compare_five_methods_pbest import (
    METHODS, METHOD_LABELS, SETTINGS, N_RUNS,
    paths_for_setting, load_front, non_dominated_front,
    dominates, compute_igd, compute_hv_mc,
)

ORDER = ["MOPSO", "MOPSO_AMS", "CAPQL", "PCN_original", "PCN_variation1", "PCN_variation2"]
ORDER_LABELS = {
    "MOPSO": "MOPSO",
    "MOPSO_AMS": "MOPSO-AMS",
    "CAPQL": "CAPQL",
    "PCN_original": "Original PCN",
    "PCN_variation1": "PCN with buffer init.",
    "PCN_variation2": "Guided PCN",
}


def set_coverage(A, B):
    """C(A,B): fraction of B's points dominated by at least one point in A."""
    if len(B) == 0:
        return np.nan
    if len(A) == 0:
        return 0.0
    count = 0
    for b in B:
        if np.any([dominates(a, b) for a in A]):
            count += 1
    return count / len(B)


def spacing(front):
    n = len(front)
    if n < 2:
        return 0.0
    d = np.zeros(n)
    for i in range(n):
        dists = np.sum(np.abs(front - front[i]), axis=1)
        dists[i] = np.inf
        d[i] = dists.min()
    return float(np.std(d, ddof=1)) if n > 1 else 0.0


def max_spread(front):
    if len(front) == 0:
        return 0.0
    return float(np.linalg.norm(front.max(axis=0) - front.min(axis=0)))


def fmt(mean, std, dec=2):
    return f"{mean:.{dec}f} \\pm {std:.{dec}f}"


def run_for_setting(setting):
    paths = paths_for_setting(setting)
    rng = np.random.default_rng(0)

    sc_runs = {a: {b: [] for b in METHODS if b != a} for a in METHODS}
    spacing_runs = {m: [] for m in METHODS}
    spread_runs = {m: [] for m in METHODS}
    n_runs = {m: [] for m in METHODS}
    tc_runs = {m: [] for m in METHODS}
    freq_runs = {m: [] for m in METHODS}
    amt_runs = {m: [] for m in METHODS}
    hv_runs = {m: [] for m in METHODS}
    igd_runs = {m: [] for m in METHODS}

    for run in range(1, N_RUNS + 1):
        fronts = {m: load_front(paths[m].format(run=run)) for m in METHODS}
        combined = np.vstack(list(fronts.values()))
        lo, hi = combined.min(axis=0), combined.max(axis=0)
        denom = np.where(hi - lo == 0, 1.0, hi - lo)
        norm = {m: (fronts[m] - lo) / denom for m in METHODS}
        ref_point = np.array([1.1, 1.1, 1.1])
        ref_front_norm = (non_dominated_front(combined) - lo) / denom

        for a in METHODS:
            for b in METHODS:
                if a == b:
                    continue
                sc_runs[a][b].append(set_coverage(fronts[a], fronts[b]))

        for m in METHODS:
            f = fronts[m]
            n_runs[m].append(len(f))
            tc_runs[m].append(f[:, 0].mean() if len(f) else 0.0)
            freq_runs[m].append(f[:, 1].mean() if len(f) else 0.0)
            amt_runs[m].append(f[:, 2].mean() if len(f) else 0.0)
            spacing_runs[m].append(spacing(f))
            spread_runs[m].append(max_spread(f))
            hv_runs[m].append(compute_hv_mc(norm[m], ref_point, rng=rng))
            igd_runs[m].append(compute_igd(norm[m], ref_front_norm))

    sc_summary = {a: {b: (np.mean(sc_runs[a][b]), np.std(sc_runs[a][b])) for b in METHODS if b != a} for a in METHODS}
    metric_summary = {
        m: {
            "N": (np.mean(n_runs[m]), np.std(n_runs[m])),
            "Spacing": (np.mean(spacing_runs[m]), np.std(spacing_runs[m])),
            "Spread": (np.mean(spread_runs[m]), np.std(spread_runs[m])),
            "TC": (np.mean(tc_runs[m]), np.std(tc_runs[m])),
            "Freq": (np.mean(freq_runs[m]), np.std(freq_runs[m])),
            "Amt": (np.mean(amt_runs[m]), np.std(amt_runs[m])),
            "HV": (np.mean(hv_runs[m]), np.std(hv_runs[m])),
            "IGD": (np.mean(igd_runs[m]), np.std(igd_runs[m])),
        }
        for m in METHODS
    }
    return sc_summary, metric_summary


def sc_table_latex(setting, sc_summary, table_num, label_suffix, caption, tabcolsep=None):
    lines = []
    lines.append(f"%table{table_num}")
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    if tabcolsep is not None:
        lines.append(f"\\setlength{{\\tabcolsep}}{{{tabcolsep}}}")
    lines.append("\\scriptsize")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{tab:SC_{label_suffix}}}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{c|c c c c c c}")
    lines.append("\\hline")
    header = " & ".join(ORDER_LABELS[m] for m in ORDER)
    lines.append("\\diagbox[width=11em,height=2em]{$A$}{$B$} & " + header + " \\\\")
    lines.append("\\hline")
    for a in ORDER:
        row = [ORDER_LABELS[a]]
        for b in ORDER:
            if a == b:
                row.append("--")
            else:
                mean, std = sc_summary[a][b]
                row.append(f"${fmt(mean, std)}$")
        lines.append(" & ".join(row) + " \\\\")
        lines.append("")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def metric_table_latex(setting, metric_summary, table_num, label_suffix, caption, tc_scale_1e3=True, tabcolsep="1pt"):
    lines = []
    lines.append(f"%table{table_num}")
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append(f"\\setlength{{\\tabcolsep}}{{{tabcolsep}}}")
    lines.append("\\scriptsize")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{tab:metric_{label_suffix}}}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{c|c c c c c c}")
    lines.append("\\hline")
    header = " & ".join(ORDER_LABELS[m] for m in ORDER)
    lines.append("Metric & " + header + " \\\\")
    lines.append("\\hline")

    def row(name, key, dec=2, scale=1.0):
        vals = []
        for m in ORDER:
            mean, std = metric_summary[m][key]
            vals.append(f"${fmt(mean * scale, std * scale, dec)}$")
        return name + "\n    & " + "\n    & ".join(vals) + " \\\\"

    lines.append(row("Spacing", "Spacing"))
    lines.append("")
    lines.append(row("Maximum spread", "Spread"))
    lines.append("")
    tc_label = "Mean total cost (1e3)" if tc_scale_1e3 else "Mean total cost"
    lines.append(row(tc_label, "TC", dec=2, scale=1.0))
    lines.append("")
    lines.append(row("Mean stockout frequency", "Freq"))
    lines.append("")
    lines.append(row("Mean stockout amount", "Amt"))
    lines.append("")
    lines.append(row("HV (normalized)", "HV", dec=4))
    lines.append("")
    lines.append(row("IGD (normalized)", "IGD", dec=4))
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append("\\end{table}")
    return "\n".join(lines)


TABLE_META = {
    "stat_low": {
        "sc_num": 1, "sc_label": "stationary_1", "sc_caption": "Results of the set coverage $C(A,B)$ metric for the stationary setting (low variance)",
        "m_num": 4, "m_label": "stationary_1", "m_caption": "Metrics of the six algorithms for stationary setting (low variance)",
    },
    "stat_medium": {
        "sc_num": 2, "sc_label": "stationary_2", "sc_caption": "Results of the set coverage $C(A,B)$ metric for stationary setting (medium variance)",
        "m_num": 5, "m_label": "stationary_2", "m_caption": "Metrics of the six algorithms for stationary setting (medium variance)",
    },
    "stat_high": {
        "sc_num": 3, "sc_label": "stationary_3", "sc_caption": "Results of the set coverage $C(A,B)$ metric for stationary setting (high variance)",
        "m_num": 6, "m_label": "stationary_3", "m_caption": "Metrics of the six algorithms for stationary setting (high variance)",
    },
    "seasonal": {
        "sc_num": 7, "sc_label": "seasonal", "sc_caption": "Results of the set coverage $C(A,B)$ metric for the seasonal setting",
        "m_num": 8, "m_label": "seasonal", "m_caption": "Metrics of the six algorithms for the seasonal setting",
    },
    "GBM": {
        "sc_num": 9, "sc_label": "GBM", "sc_caption": "Results of the set coverage $C(A,B)$ metric for the geometric Brownian motion (GBM) setting",
        "m_num": 10, "m_label": "GBM", "m_caption": "Metrics of the six algorithms for the GBM setting",
    },
    "gs": {
        "sc_num": 11, "sc_label": "gs", "sc_caption": "Results of the set coverage $C(A,B)$ metric for the case-study setting",
        "m_num": 12, "m_label": "gs", "m_caption": "Metrics of the six algorithms for the case-study setting",
    },
}

if __name__ == "__main__":
    out_lines = []
    all_summaries = {}
    for setting in SETTINGS:
        print(f"=== {setting} ===", flush=True)
        sc_summary, metric_summary = run_for_setting(setting)
        all_summaries[setting] = (sc_summary, metric_summary)
        meta = TABLE_META[setting]
        out_lines.append(sc_table_latex(setting, sc_summary, meta["sc_num"], meta["sc_label"], meta["sc_caption"], tabcolsep=("2pt" if setting == "gs" else None)))
        out_lines.append("")
        out_lines.append(metric_table_latex(setting, metric_summary, meta["m_num"], meta["m_label"], meta["m_caption"], tc_scale_1e3=(setting != "gs"), tabcolsep=("2pt" if setting == "gs" else "1pt")))
        out_lines.append("")

    with open("_paper_tables_700_out.txt", "w") as f:
        f.write("\n".join(out_lines))
    print("\nSaved -> _paper_tables_700_out.txt")

    # Also dump a compact numeric CSV for quick reference / sanity checks
    rows = []
    for setting in SETTINGS:
        _, metric_summary = all_summaries[setting]
        for m in ORDER:
            r = {"setting": setting, "method": ORDER_LABELS[m]}
            for k in ["N", "Spacing", "Spread", "TC", "Freq", "Amt", "HV", "IGD"]:
                mean, std = metric_summary[m][k]
                r[f"{k}_mean"] = mean
                r[f"{k}_std"] = std
            rows.append(r)
    pd.DataFrame(rows).to_csv("_paper_tables_700_metrics.csv", index=False)
    print("Saved -> _paper_tables_700_metrics.csv")
