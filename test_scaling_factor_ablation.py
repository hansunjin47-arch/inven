"""Ablation test: does using the user's originally-intended, per-variance
scaling_factor for stat_low/stat_high (instead of the medium-variance value
that has been used for all three stationary levels throughout this session
and in main_original.py's own __main__) actually change Guided PCN's
(softmax objective selection) performance?

  - stat_low:  scaling_factor = [0.0075, 0.1, 0.1, 0.1]   (per user's note)
  - stat_high: scaling_factor = [0.0001, 0.1, 0.1, 0.1]   (per user's note)
  (both differ from the medium value [0.000125, 0.1, 1.0, 0.1] that was
  actually used for all three levels so far)

Runs variation2 (softmax) for both settings, 10 seeds each, saved to a
separate folder for comparison against the existing (medium-scaling-factor)
PCN_stochastic_objsel_test results.

Run in the `pcn` conda env:

    conda activate pcn
    python test_scaling_factor_ablation.py
"""

import os
import types
import uuid
from datetime import datetime
import time

import gym
import numpy as np
import pandas as pd
import torch

import pcn_continuous
from main_original import ContinuousHead
from pcn_continuous import train

_RealInventory = pcn_continuous.Inventory

STATIONARY_MU_SIGMA = {
    "stat_low": (3.29, 0.23),
    "stat_high": (3.0, 0.8),
}
# per the user's own note (not yet reflected in any script this session)
SCALING_FACTOR_NOTED = {
    "stat_low": [0.0075, 0.1, 0.1, 0.1],
    "stat_high": [0.0001, 0.1, 0.1, 0.1],
}
MOPSO_SEED_PATH = {
    "stat_low": "./MOPSO/MOPSO_stationary_lognormal_3.29,0.23/MOPSO_run_stationary_{run_id}.csv",
    "stat_high": "./MOPSO/MOPSO_stationary_lognormal_3.0,0.8/MOPSO_run_stationary_{run_id}.csv",
}
N_RUNS = 10
VERSION = "variation2"


def _make_stationary_demand_fn(mu, sigma):
    def _get_demand_stationary(self, _mu=mu, _sigma=sigma):
        return np.random.lognormal(mean=_mu, sigma=_sigma)
    return _get_demand_stationary


def make_patched_inventory(mu, sigma):
    def patched(scaling_factor, demand_env, fixed_demand_seq=None):
        env = _RealInventory(scaling_factor, demand_env, fixed_demand_seq)
        D = np.exp(mu + sigma ** 2 / 2) * 365
        env.D = D
        env.action_space = gym.spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([D, D], dtype=np.float32),
            dtype=np.float32,
        )
        env.get_demand_stationary = types.MethodType(_make_stationary_demand_fn(mu, sigma), env)
        return env
    return patched


if __name__ == "__main__":
    import random

    total_start = time.time()
    for setting in ["stat_low", "stat_high"]:
        mu, sigma = STATIONARY_MU_SIGMA[setting]
        pcn_continuous.Inventory = make_patched_inventory(mu, sigma)

        out_dir = f"./PCN_scaling_ablation_test/PCN_{setting}_variation2_notedscaling"
        os.makedirs(out_dir, exist_ok=True)

        for run_id in range(1, N_RUNS + 1):
            random.seed(run_id)
            np.random.seed(run_id)
            torch.manual_seed(run_id)

            device = "cpu"
            scaling_factor = torch.tensor([SCALING_FACTOR_NOTED[setting]]).to(device)
            env = pcn_continuous.Inventory(scaling_factor, "stationary")
            env.nA = np.prod(env.action_space.shape)
            model = ContinuousHead(pcn_continuous.Inventory(scaling_factor, "stationary").to(device), "stationary")
            model.D = np.exp(mu + sigma ** 2 / 2) * 365

            seed_path = MOPSO_SEED_PATH[setting].format(run_id=run_id)
            logdir = f"/PCN/diag_scaling_ablation_{setting}_{run_id}/"
            logdir += datetime.now().strftime("%Y-%m-%d_%H-%M-%S_") + str(uuid.uuid4())[:4] + "/"

            print(f"=== {setting} | variation2 | noted scaling_factor={SCALING_FACTOR_NOTED[setting]} | run {run_id} ===")
            start = time.time()
            non_dom_rewards, mopso_time, pcn_time = train(
                VERSION, "stationary", env, model, run_id, scaling_factor,
                learning_rate=1e-4, batch_size=256, total_steps=1e6,
                n_model_updates=100, n_step_episodes=10, max_size=500,
                noise=0.0, logdir=logdir, clip_grad_norm=None,
                pareto_seed_path=seed_path,
                stochastic_objective_selection=True,
            )
            elapsed = time.time() - start

            out_path = f"{out_dir}/PCN_{setting}_variation2_notedscaling_run_{run_id}.csv"
            df = pd.DataFrame(
                {
                    "TC": [-1 * p[0] for p in non_dom_rewards],
                    "Frequency": [-1 * p[1] for p in non_dom_rewards],
                    "Amount": [-1 * p[2] for p in non_dom_rewards],
                }
            )
            df.to_csv(out_path, index=False)
            print(f"[{setting} | run {run_id}] {elapsed:.1f}s, {len(df)} solutions -> {out_path}")

    print(f"\nAll done in {time.time() - total_start:.1f}s")
