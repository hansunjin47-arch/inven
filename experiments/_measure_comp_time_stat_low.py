import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _sub in ("baselines", "envs", "models", "utils", "experiments"):
    _p = _os.path.join(_ROOT, _sub)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
_os.chdir(_ROOT)

"""One-off: time a single run (run_id=1) of MOPSO, MOPSO-AMS (2x matched
budget), Original PCN, PCN+buffer init, and Guided PCN on stat_low, using
the exact same configs that produced the paper's reported numbers (MOPSO:
tsou_pbest_update=True, 200x100; MOPSO-AMS: I=200,T=200,4 subswarms,
archive cap 200; PCN variants: grid_search_pbest_training_params baseline
kwargs + max_size=750). CAPQL is timed separately via capql_uniform_recipe.py
in the gpils env since it needs CUDA.

Run in the `pcn` conda env:
    conda run -n pcn python _measure_comp_time_stat_low.py
"""

import json
import os
import random
import time
import types

import gym
import numpy as np
import pandas as pd
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from main_original import Inventory
from MOPSO import MOPSO, get_non_dominated
from MOPSO_AMS import MOPSO_AMS
from grid_search_pbest_training_params import BASELINE_TRAIN_KWARGS, run_one

MU, SIGMA = 3.29, 0.23
SCALING_FACTOR_VALS = [0.0001, 0.1, 0.1, 0.1]
RUN_ID = 1

results = {}


def _make_stationary_demand_fn(mu, sigma):
    def _get_demand_stationary(self, _mu=mu, _sigma=sigma):
        return np.random.lognormal(mean=_mu, sigma=_sigma)
    return _get_demand_stationary


def build_env():
    scaling_factor = torch.tensor([SCALING_FACTOR_VALS])
    env = Inventory(scaling_factor, "stationary")
    D = np.exp(MU + SIGMA ** 2 / 2) * 365
    env.D = D
    env.action_space = gym.spaces.Box(
        low=np.array([0.0, 0.0], dtype=np.float32),
        high=np.array([D, D], dtype=np.float32),
        dtype=np.float32,
    )
    env.get_demand_stationary = types.MethodType(_make_stationary_demand_fn(MU, SIGMA), env)
    return env, scaling_factor


# ---- MOPSO (pbest tie-break, 200 particles x 100 iterations) ----
random.seed(RUN_ID); np.random.seed(RUN_ID); torch.manual_seed(RUN_ID)
env, scaling_factor = build_env()
bounds = np.array([[0, env.D], [0, env.D]])
particle_positions = np.array([np.random.uniform(low, high, size=200) for (low, high) in bounds]).T
start = time.time()
mopso = MOPSO(num_particles=200, dim=2, bounds=bounds, num_iterations=100,
              initial_positions=particle_positions, tsou_pbest_update=True)
solution, _, _ = mopso.optimize_variation(env, "stationary")
results["MOPSO"] = time.time() - start
print(f"[MOPSO] {results['MOPSO']:.2f}s, {len(solution)} solutions", flush=True)

# ---- MOPSO-AMS (2x matched budget: 200 particles x 200 iterations, 4 subswarms) ----
random.seed(RUN_ID); np.random.seed(RUN_ID); torch.manual_seed(RUN_ID)
env, scaling_factor = build_env()
bounds = np.array([[0, env.D], [0, env.D]])
start = time.time()
mopso_ams = MOPSO_AMS(num_particles=200, dim=2, bounds=bounds, num_iterations=200,
                       num_subswarms=4, max_archive_size=200)
archive, _, _ = mopso_ams.optimize_variation(env, "stationary")
results["MOPSO_AMS"] = time.time() - start
print(f"[MOPSO-AMS] {results['MOPSO_AMS']:.2f}s, {len(archive)} solutions", flush=True)

# ---- Original PCN / PCN+buffer init / Guided PCN (max_size=750, run_id=1) ----
train_kwargs = dict(BASELINE_TRAIN_KWARGS)
train_kwargs["max_size"] = 750
out_dir = "./_comp_time_tmp"
os.makedirs(out_dir, exist_ok=True)

for version, label in [("original", "PCN_original"), ("variation1", "PCN_variation1"), ("variation2", "PCN_variation2")]:
    start = time.time()
    n_solutions = run_one("stat_low", RUN_ID, train_kwargs, out_dir, version=version)
    results[label] = time.time() - start
    print(f"[{label}] {results[label]:.2f}s, {n_solutions} solutions", flush=True)

with open("_comp_time_stat_low_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n=== Summary (stat_low, run_id=1) ===")
for k, v in results.items():
    print(f"{k}: {v:.2f}s")
