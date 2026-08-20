"""Combines the 3 grid-search parameters with the clearest net-positive
effect (total_steps=2e6: 6/6 settings better; n_model_updates=200: 5/6
better; batch_size=512: 4/6 better, 1 mixed, 1 worse) into a single
configuration, and reruns Guided PCN (variation2) across all 6 settings x 10
runs against this combination. Uses the original (non-pbest) MOPSO seed to
isolate the effect of these 3 hyperparameters alone, matching the same
baseline used throughout compare_grid_search_*_full10.py.

Run in the `pcn` conda env:

    conda activate pcn
    python run_combined_winners_all_settings.py
"""

import os
import time

from grid_search_training_params import SETTINGS, BASELINE_TRAIN_KWARGS, N_RUNS_SCREEN, run_one

COMBINED_KWARGS = dict(BASELINE_TRAIN_KWARGS)
COMBINED_KWARGS["total_steps"] = 2e6
COMBINED_KWARGS["n_model_updates"] = 200
COMBINED_KWARGS["batch_size"] = 512

if __name__ == "__main__":
    total_start = time.time()
    for setting in SETTINGS:
        out_dir = f"./PCN_combined_winners/PCN_{setting}_variation2_combined"
        os.makedirs(out_dir, exist_ok=True)

        for run_id in range(1, N_RUNS_SCREEN + 1):
            print(f"=== combined(total_steps=2e6,n_model_updates=200,batch_size=512) | {setting} | run {run_id} ===")
            start = time.time()
            n_solutions = run_one(setting, run_id, COMBINED_KWARGS, out_dir)
            elapsed = time.time() - start
            print(f"[{setting} | run {run_id}] {elapsed:.1f}s, {n_solutions} solutions")

    print(f"\nAll done in {time.time() - total_start:.1f}s")
