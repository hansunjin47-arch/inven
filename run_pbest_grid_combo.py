"""Runs a named combination of >1 compute-neutral overrides at once (e.g.
learning_rate=2e-4 + max_size=1000 together), against the pbest-fixed MOPSO
seed, for a single setting x 10 runs.

Usage:
    conda activate pcn
    python run_pbest_grid_combo.py <setting> <combo> [version]

<combo> one of: lr2e4_maxsize1000, total_steps2e6_lr2e4, total_steps2e6_maxsize1000,
                 total_steps2e6_lr2e4_maxsize1000, maxsize750_lr2e4, total_steps2e6_maxsize750,
                 maxsize750_nmodelupdates200, maxsize750_batchsize512
[version] one of: original, variation1, variation2 (default: variation2)
"""

import os
import sys
import time

from grid_search_pbest_training_params import BASELINE_TRAIN_KWARGS, N_RUNS, run_one

COMBOS = {
    "lr2e4_maxsize1000": {"learning_rate": 2e-4, "max_size": 1000},
    "total_steps2e6_lr2e4": {"total_steps": 2e6, "learning_rate": 2e-4},
    "total_steps2e6_maxsize1000": {"total_steps": 2e6, "max_size": 1000},
    "total_steps2e6_lr2e4_maxsize1000": {"total_steps": 2e6, "learning_rate": 2e-4, "max_size": 1000},
    "maxsize750_lr2e4": {"max_size": 750, "learning_rate": 2e-4},
    "total_steps2e6_maxsize750": {"total_steps": 2e6, "max_size": 750},
    "maxsize750_nmodelupdates200": {"max_size": 750, "n_model_updates": 200},
    "maxsize750_batchsize512": {"max_size": 750, "batch_size": 512},
    "maxsize700_lr2e4": {"max_size": 700, "learning_rate": 2e-4},
    "total_steps2e6_maxsize700": {"total_steps": 2e6, "max_size": 700},
    "maxsize700_nmodelupdates200": {"max_size": 700, "n_model_updates": 200},
    "maxsize700_batchsize512": {"max_size": 700, "batch_size": 512},
    "maxsize700_noise05": {"max_size": 700, "noise": 0.5},
    "maxsize700_clipgradnorm05": {"max_size": 700, "clip_grad_norm": 0.5},
}
COSTLY_COMBOS = {
    "total_steps2e6_lr2e4", "total_steps2e6_maxsize1000", "total_steps2e6_lr2e4_maxsize1000",
    "total_steps2e6_maxsize750", "maxsize750_nmodelupdates200", "maxsize750_batchsize512",
    "total_steps2e6_maxsize700", "maxsize700_nmodelupdates200", "maxsize700_batchsize512",
}

if __name__ == "__main__":
    setting = sys.argv[1]
    combo = sys.argv[2]
    version = sys.argv[3] if len(sys.argv) > 3 else "variation2"

    kwargs = dict(BASELINE_TRAIN_KWARGS)
    kwargs.update(COMBOS[combo])

    grid_folder = "PCN_pbest_grid_costly" if combo in COSTLY_COMBOS else "PCN_pbest_grid_free"
    out_dir = f"./{grid_folder}/PCN_{setting}_{version}_{combo}"
    os.makedirs(out_dir, exist_ok=True)

    total_start = time.time()
    for run_id in range(1, N_RUNS + 1):
        print(f"=== {version} | {combo} | {setting} | run {run_id} ===")
        start = time.time()
        n_solutions = run_one(setting, run_id, kwargs, out_dir, version=version)
        elapsed = time.time() - start
        print(f"[{version} | {combo} | {setting} | run {run_id}] {elapsed:.1f}s, {n_solutions} solutions")

    print(f"\n{version} | {combo} | {setting} all done in {time.time() - total_start:.1f}s")
