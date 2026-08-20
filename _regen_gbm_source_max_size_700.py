import os
from grid_search_pbest_training_params import BASELINE_TRAIN_KWARGS, run_one

train_kwargs = dict(BASELINE_TRAIN_KWARGS)
train_kwargs["max_size"] = 700
out_dir = "./result/gbm_trajectory_source_verified"
os.makedirs(out_dir, exist_ok=True)

for version in ["original", "variation1", "variation2"]:
    n = run_one("GBM", 1, train_kwargs, out_dir, version=version)
    print(f"[{version}] {n} solutions -> {out_dir}", flush=True)
