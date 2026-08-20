import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _sub in ("baselines", "envs", "models", "utils", "experiments"):
    _p = _os.path.join(_ROOT, _sub)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
_os.chdir(_ROOT)

import os
from grid_search_pbest_training_params import BASELINE_TRAIN_KWARGS, run_one

train_kwargs = dict(BASELINE_TRAIN_KWARGS)
train_kwargs["max_size"] = 700
out_dir = "results/gbm_trajectory_source_verified"
os.makedirs(out_dir, exist_ok=True)

for version in ["original", "variation1", "variation2"]:
    n = run_one("GBM", 1, train_kwargs, out_dir, version=version)
    print(f"[{version}] {n} solutions -> {out_dir}", flush=True)
