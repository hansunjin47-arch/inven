"""Runs the combined-winners configuration (total_steps=2e6, n_model_updates=200,
batch_size=512) for a single setting x 10 runs. Lets all 6 settings run as
separate parallel processes instead of one long sequential script.

Usage:
    conda activate pcn
    python run_combined_winners_single_setting.py <setting>
"""

import os
import sys
import time

from grid_search_training_params import BASELINE_TRAIN_KWARGS, N_RUNS_SCREEN, run_one

COMBINED_KWARGS = dict(BASELINE_TRAIN_KWARGS)
COMBINED_KWARGS["total_steps"] = 2e6
COMBINED_KWARGS["n_model_updates"] = 200
COMBINED_KWARGS["batch_size"] = 512

if __name__ == "__main__":
    setting = sys.argv[1]
    out_dir = f"./PCN_combined_winners/PCN_{setting}_variation2_combined"
    os.makedirs(out_dir, exist_ok=True)

    total_start = time.time()
    for run_id in range(1, N_RUNS_SCREEN + 1):
        print(f"=== combined | {setting} | run {run_id} ===")
        start = time.time()
        n_solutions = run_one(setting, run_id, COMBINED_KWARGS, out_dir)
        elapsed = time.time() - start
        print(f"[{setting} | run {run_id}] {elapsed:.1f}s, {n_solutions} solutions")

    print(f"\n{setting} all done in {time.time() - total_start:.1f}s")
