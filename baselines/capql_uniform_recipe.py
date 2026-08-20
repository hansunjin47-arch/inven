import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _sub in ("baselines", "envs", "models", "utils", "experiments"):
    _p = _os.path.join(_ROOT, _sub)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
_os.chdir(_ROOT)

"""Single uniform recipe applied identically across all 6 scenarios:
obs-norm ON (ObsNormalizer, default eps=1e-6), alpha=0.05, start_steps=10000,
lr=3e-4 (paper default), total_steps=300000. No per-scenario parameter
differences -- this replaces the earlier per-scenario patchwork
(tuning_log.md Rounds 0-6) with one consistent setting, validated with 10
seeds per scenario as requested.

total_steps raised from the original 100000 test budget to 300000 based on
the Round 11 convergence-curve check (results/capql/convergence_curves.csv):
4/5 scenarios already plateaued by 100k, but stat_high only reached its
plateau around 150k-300k, so 300k is a round number that covers all
scenarios' convergence points without the ~10-100x cost of matching the
reference repo's 10M-step MuJoCo convention (which this simpler, faster-
converging custom domain doesn't need).

Usage (run in the `pcn` conda env):

    conda activate pcn
    python capql_uniform_recipe.py <scenario> <run_id>

    <scenario>: one of GBM, stat_low, stat_medium, stat_high, seasonal, gs
    <run_id>: 1-10 (used as the random/np/torch seed)

Saves the non-dominated front to
./results/capql/uniform_300k/<scenario>_run<run_id>.csv
"""

import os
import pickle
import random
import sys
import time
import types

import numpy as np
import pandas as pd
import torch

import capql_baseline as cb

SCENARIO = sys.argv[1]
RUN_ID = int(sys.argv[2])

SCALING_FACTOR = torch.tensor([[0.0001, 0.1, 0.1, 0.1]])
ALPHA = 0.05
START_STEPS = 10000
TOTAL_STEPS = 300000


def _make_stationary_demand_fn(mu, sigma):
    def _get_demand_stationary(self, _mu=mu, _sigma=sigma):
        return np.random.lognormal(mean=_mu, sigma=_sigma)
    return _get_demand_stationary


def make_patched_inventory(mu, sigma):
    from main_original import Inventory as _RealInventory

    def patched(scaling_factor, demand_env, fixed_demand_seq=None):
        env = _RealInventory(scaling_factor, demand_env, fixed_demand_seq)
        D = np.exp(mu + sigma ** 2 / 2) * 365
        env.D = D
        import gym
        env.action_space = gym.spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([D, D], dtype=np.float32),
            dtype=np.float32,
        )
        env.get_demand_stationary = types.MethodType(_make_stationary_demand_fn(mu, sigma), env)
        return env
    return patched


STATIONARY_MU_SIGMA = {"stat_low": (3.29, 0.23), "stat_high": (3.0, 0.8)}

if __name__ == "__main__":
    random.seed(RUN_ID)
    np.random.seed(RUN_ID)
    torch.manual_seed(RUN_ID)

    if SCENARIO == "GBM":
        from pcn_continuous import _generate_demand_sequence
        fixed_seq = _generate_demand_sequence()
        env = cb.ObsNormalizer(cb.InventoryMOEnv(SCALING_FACTOR, "GBM", fixed_demand_seq=fixed_seq))
    elif SCENARIO == "stat_medium":
        env = cb.ObsNormalizer(cb.InventoryMOEnv(SCALING_FACTOR, "stationary"))
    elif SCENARIO in STATIONARY_MU_SIGMA:
        mu, sigma = STATIONARY_MU_SIGMA[SCENARIO]
        cb.Inventory = make_patched_inventory(mu, sigma)
        env = cb.ObsNormalizer(cb.InventoryMOEnv(SCALING_FACTOR, "stationary"))
    elif SCENARIO == "seasonal":
        env = cb.ObsNormalizer(cb.InventoryMOEnv(SCALING_FACTOR, "seasonal"))
    elif SCENARIO == "gs":
        with open("data/daily_demand_2015_good1.pkl", "rb") as f:
            fixed_seq = pickle.load(f)
        env = cb.ObsNormalizer(cb.InventoryMOEnv(SCALING_FACTOR, "gs", fixed_demand_seq=fixed_seq))
    else:
        raise ValueError(f"unknown scenario {SCENARIO}")

    print(f"=== CAPQL uniform recipe: {SCENARIO} run {RUN_ID} "
          f"(obs_norm=True, alpha={ALPHA}, start_steps={START_STEPS}, total_steps={TOTAL_STEPS}) ===", flush=True)

    train_start = time.time()
    agent = cb.train_capql(env, total_steps=TOTAL_STEPS, start_steps=START_STEPS,
                            alpha=ALPHA, device="cuda", verbose=False)
    train_elapsed = time.time() - train_start
    print(f"[{SCENARIO} run {RUN_ID}] training done in {train_elapsed:.1f}s", flush=True)

    if SCENARIO == "gs":
        # pcn_continuous.py's own reference pipeline trains on 2015 data and
        # evaluates on 2016 data ("Starting sequential evaluation with 2016
        # data") -- our earlier gs runs skipped this split and evaluated on
        # the same 2015 data used for training. Swap in the 2016 test
        # sequence now, keeping the same ObsNormalizer instance (and its
        # obs_rms learned during training) and the same InventoryMOEnv --
        # only the underlying demand data changes.
        with open("data/daily_demand_2016_good1.pkl", "rb") as f:
            test_seq = pickle.load(f)
        env.sim.fixed_demand_seq = test_seq

    eval_start = time.time()
    front = cb.evaluate_front(agent, env, n_eval_weights=200, n_episodes_per_weight=1)
    eval_elapsed = time.time() - eval_start

    os.makedirs("./results/capql/uniform_300k", exist_ok=True)
    df = pd.DataFrame({"TC": -front[:, 0], "Frequency": -front[:, 1], "Amount": -front[:, 2]})
    out_path = f"./results/capql/uniform_300k/{SCENARIO}_run{RUN_ID}.csv"
    df.to_csv(out_path, index=False)
    print(f"[{SCENARIO} run {RUN_ID}] {len(front)} non-dominated points -> saved {out_path}")
    print(f"[{SCENARIO} run {RUN_ID}] eval done in {eval_elapsed:.1f}s")

    # Representative action trajectory (equal-weight preference vector) for the
    # "representative action trajectories" figure -- CAPQL analogue of PCN's
    # Episode==0 trajectory dump, since CAPQL's action is state-conditioned
    # (unlike MOPSO/MOPSO-AMS's single static (Q, r) policy).
    traj_path = f"./results/capql/uniform_300k/{SCENARIO}_run{RUN_ID}_action_trajectory.csv"
    equal_w = np.full(env.rwd_dim, 1.0 / env.rwd_dim, dtype=np.float32)
    cb.rollout_action_trajectory(agent, env, equal_w, out_path=traj_path)
    print(f"[{SCENARIO} run {RUN_ID}] action trajectory (w={equal_w}) -> saved {traj_path}")

    total_elapsed = time.time() - train_start
    print(f"[{SCENARIO} run {RUN_ID}] total wall-clock time: {total_elapsed:.1f}s "
          f"(train={train_elapsed:.1f}s, eval={eval_elapsed:.1f}s)")
