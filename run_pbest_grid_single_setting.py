"""Runs one hyperparameter variant against the pbest-fixed MOPSO seed, for a
single setting x 10 runs. Baseline itself is NOT rerun here -- it already
exists at ./PCN_pbest_seed/PCN_{setting}_variation2_pbestseed/. Compute-
neutral variants are saved under PCN_pbest_grid_free/; the compute-costly
ones (total_steps/n_model_updates/batch_size, all meaningfully more
expensive than baseline) are saved under PCN_pbest_grid_costly/ to keep
the two kinds of results visually separated.

Usage:
    conda activate pcn
    python run_pbest_grid_single_setting.py <setting> <variant> [version]

<variant> one of: learning_rate_2e-4, learning_rate_3e-4, learning_rate_5e-5, learning_rate_1.5e-4,
                   clip_grad_norm_0.5, clip_grad_norm_2.0,
                   noise_0.1, noise_0.3, noise_0.5,
                   action_noise_scale_0.01, action_noise_scale_0.05, action_noise_scale_0.1,
                   max_size_1000, max_size_200, max_size_750, max_size_1500, max_size_600,
                   max_size_700, max_size_800, max_size_900,
                   total_steps_2e6, total_steps_1.5e6, total_steps_3e6,
                   n_model_updates_200, batch_size_512, batch_size_128
[version] one of: original, variation1, variation2 (default: variation2)
"""

import os
import sys
import time

from grid_search_pbest_training_params import BASELINE_TRAIN_KWARGS, N_RUNS, run_one

VARIANTS = {
    "learning_rate_2e-4": ("learning_rate", 2e-4),
    "learning_rate_3e-4": ("learning_rate", 3e-4),
    "learning_rate_5e-5": ("learning_rate", 5e-5),
    "learning_rate_1.5e-4": ("learning_rate", 1.5e-4),
    "clip_grad_norm_0.5": ("clip_grad_norm", 0.5),
    "clip_grad_norm_2.0": ("clip_grad_norm", 2.0),
    "noise_0.1": ("noise", 0.1),
    "noise_0.3": ("noise", 0.3),
    "noise_0.5": ("noise", 0.5),
    "action_noise_scale_0.01": ("action_noise_scale", 0.01),
    "action_noise_scale_0.05": ("action_noise_scale", 0.05),
    "action_noise_scale_0.1": ("action_noise_scale", 0.1),
    "max_size_1000": ("max_size", 1000),
    "max_size_200": ("max_size", 200),
    "max_size_750": ("max_size", 750),
    "max_size_1500": ("max_size", 1500),
    "max_size_600": ("max_size", 600),
    "max_size_700": ("max_size", 700),
    "max_size_800": ("max_size", 800),
    "max_size_900": ("max_size", 900),
    "total_steps_2e6": ("total_steps", 2e6),
    "total_steps_1.5e6": ("total_steps", 1.5e6),
    "total_steps_3e6": ("total_steps", 3e6),
    "n_model_updates_200": ("n_model_updates", 200),
    "batch_size_512": ("batch_size", 512),
    "batch_size_128": ("batch_size", 128),
}
COSTLY_VARIANTS = {"total_steps_2e6", "total_steps_1.5e6", "total_steps_3e6", "n_model_updates_200", "batch_size_512"}

if __name__ == "__main__":
    setting = sys.argv[1]
    variant = sys.argv[2]
    version = sys.argv[3] if len(sys.argv) > 3 else "variation2"
    param_name, param_value = VARIANTS[variant]

    kwargs = dict(BASELINE_TRAIN_KWARGS)
    kwargs[param_name] = param_value

    grid_folder = "PCN_pbest_grid_costly" if variant in COSTLY_VARIANTS else "PCN_pbest_grid_free"
    out_dir = f"./{grid_folder}/PCN_{setting}_{version}_{variant}"
    os.makedirs(out_dir, exist_ok=True)

    total_start = time.time()
    for run_id in range(1, N_RUNS + 1):
        print(f"=== {version} | {variant} | {setting} | run {run_id} ===")
        start = time.time()
        n_solutions = run_one(setting, run_id, kwargs, out_dir, version=version)
        elapsed = time.time() - start
        print(f"[{version} | {variant} | {setting} | run {run_id}] {elapsed:.1f}s, {n_solutions} solutions")

    print(f"\n{version} | {variant} | {setting} all done in {time.time() - total_start:.1f}s")
