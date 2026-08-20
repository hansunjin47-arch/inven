"""One-off: before archiving PCN_pbest_grid_free/ and PCN_pbest_grid_costly/
(all the single-factor and combo hyperparameter sweeps explored during this
project's tuning phase), extract a compact per-combo summary (N solutions,
mean objectives) into a single permanent CSV so the raw per-run CSVs and
model checkpoints can be safely archived without losing the sweep results.

Run in the `pcn` conda env:
    conda run -n pcn python _summarize_grid_search_before_archive.py
"""

import glob
import os

import numpy as np
import pandas as pd

ROOT_DIRS = ["PCN_pbest_grid_free", "PCN_pbest_grid_costly"]

rows = []
for root in ROOT_DIRS:
    if not os.path.isdir(root):
        continue
    for combo_dir in sorted(os.listdir(root)):
        full_dir = os.path.join(root, combo_dir)
        if not os.path.isdir(full_dir):
            continue
        csvs = sorted(glob.glob(os.path.join(full_dir, "*.csv")))
        if not csvs:
            continue

        n_list, tc_list, freq_list, amt_list = [], [], [], []
        for csv_path in csvs:
            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                print(f"[skip] {csv_path}: {e}")
                continue
            n_list.append(len(df))
            if len(df) > 0 and "TC" in df.columns:
                tc_list.append(df["TC"].mean())
                freq_list.append(df["Frequency"].mean())
                amt_list.append(df["Amount"].mean())

        rows.append({
            "grid_root": root,
            "combo_dir": combo_dir,
            "n_runs_found": len(csvs),
            "N_mean": np.mean(n_list) if n_list else np.nan,
            "N_std": np.std(n_list) if n_list else np.nan,
            "TC_mean": np.mean(tc_list) if tc_list else np.nan,
            "TC_std": np.std(tc_list) if tc_list else np.nan,
            "Freq_mean": np.mean(freq_list) if freq_list else np.nan,
            "Freq_std": np.std(freq_list) if freq_list else np.nan,
            "Amt_mean": np.mean(amt_list) if amt_list else np.nan,
            "Amt_std": np.std(amt_list) if amt_list else np.nan,
        })
        print(f"[{root}] {combo_dir}: {len(csvs)} runs", flush=True)

out = pd.DataFrame(rows)
out.to_csv("grid_search_sweep_summary.csv", index=False)
print(f"\nSaved {len(out)} combo summaries -> grid_search_sweep_summary.csv")
