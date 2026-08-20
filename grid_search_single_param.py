"""Runs a single grid-search parameter (all 6 settings x 10 runs), reusing
run_one/SETTINGS/BASELINE_TRAIN_KWARGS from grid_search_training_params.py.
Lets the remaining grid-search parameters be launched as separate parallel
processes instead of running sequentially in one process.

Usage:
    conda activate pcn
    python grid_search_single_param.py <param_name> <param_value>

Example:
    python grid_search_single_param.py n_model_updates 200
    python grid_search_single_param.py clip_grad_norm 1.0
    python grid_search_single_param.py learning_rate 5e-5
    python grid_search_single_param.py batch_size 512
"""

import os
import sys
import time

from grid_search_training_params import (
    SETTINGS, BASELINE_TRAIN_KWARGS, N_RUNS_SCREEN, run_one,
)

if __name__ == "__main__":
    param_name = sys.argv[1]
    param_value = sys.argv[2]
    # cast to the right type based on the baseline's existing value type
    baseline_val = BASELINE_TRAIN_KWARGS[param_name]
    if isinstance(baseline_val, int) and param_name != "clip_grad_norm":
        param_value = int(float(param_value))
    elif param_name == "clip_grad_norm":
        param_value = float(param_value)
    else:
        param_value = float(param_value)

    train_kwargs = dict(BASELINE_TRAIN_KWARGS)
    train_kwargs[param_name] = param_value

    total_start = time.time()
    for setting in SETTINGS:
        out_dir = f"./PCN_grid_search/PCN_{setting}_variation2_{param_name}_{param_value}"
        os.makedirs(out_dir, exist_ok=True)

        for run_id in range(1, N_RUNS_SCREEN + 1):
            print(f"=== {param_name}={param_value} | {setting} | run {run_id} ===")
            start = time.time()
            n_solutions = run_one(setting, run_id, train_kwargs, out_dir)
            elapsed = time.time() - start
            print(f"[{param_name}={param_value} | {setting} | run {run_id}] {elapsed:.1f}s, {n_solutions} solutions")

    print(f"\n{param_name}={param_value} all done in {time.time() - total_start:.1f}s")
