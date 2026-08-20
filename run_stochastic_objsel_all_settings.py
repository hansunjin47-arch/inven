"""Runs variation2 (Guided PCN) with softmax(|z|)-based stochastic objective
selection (stochastic_objective_selection=True) across all settings except
GBM (already done separately, see test_stochastic_objsel_variation2_gbm.py):
stat_low, stat_medium, stat_high, seasonal, gs.

Mirrors rerun_fixed_all_settings.py's monkeypatching/seeding conventions
exactly, so results are directly comparable to the existing PCN_fixed
variation2 baselines for each setting.

Run in the `pcn` conda env:

    conda activate pcn
    python run_stochastic_objsel_all_settings.py
"""

import os
import pickle
import random
import shutil
import time
import types
import uuid
from datetime import datetime

import gym
import numpy as np
import pandas as pd
import torch

import pcn_continuous
from main_original import ContinuousHead
from pcn_continuous import train

_RealInventory = pcn_continuous.Inventory

SETTINGS = ["stat_low", "stat_medium", "stat_high", "seasonal", "gs"]
DEMAND_ENV_FOR_SETTING = {
    "stat_low": "stationary",
    "stat_medium": "stationary",
    "stat_high": "stationary",
    "seasonal": "seasonal",
    "gs": "gs",
}
STATIONARY_MU_SIGMA = {
    "stat_low": (3.29, 0.23),
    "stat_medium": (3.22, 0.45),
    "stat_high": (3.0, 0.8),
}
SCALING_FACTOR_FOR_DEMAND_ENV = {
    "stationary": [0.000125, 0.1, 1.0, 0.1],
    "seasonal": [0.0001, 0.1, 0.1, 0.1],
    "gs": [0.0001, 0.1, 0.1, 0.1],
}
MOPSO_SEED_PATH = {
    "stat_low": "./MOPSO/MOPSO_stationary_lognormal_3.29,0.23/MOPSO_run_stationary_{run_id}.csv",
    "stat_medium": "./MOPSO/MOPSO_stationary/MOPSO_run_stationary_{run_id}.csv",
    "stat_high": "./MOPSO/MOPSO_stationary_lognormal_3.0,0.8/MOPSO_run_stationary_{run_id}.csv",
    "seasonal": "./MOPSO/MOPSO_seasonal/MOPSO_run_seasonal_{run_id}.csv",
    "gs": "./MOPSO/MOPSO_run_gs_{run_id}.csv",
}
N_RUNS = 10
RUN_TEST_OUTPUT_TEMPLATE = "./PCN/original_PCN_run_test1_variation2_{run_id}.csv"


def _make_stationary_demand_fn(mu, sigma):
    def _get_demand_stationary(self, _mu=mu, _sigma=sigma):
        return np.random.lognormal(mean=_mu, sigma=_sigma)
    return _get_demand_stationary


def make_patched_inventory(mu, sigma):
    def patched(scaling_factor, demand_env, fixed_demand_seq=None):
        env = _RealInventory(scaling_factor, demand_env, fixed_demand_seq)
        if demand_env == "stationary":
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


def build_model(setting, demand_env, scaling_factor, device):
    base = _RealInventory(scaling_factor, demand_env).to(device)
    model = ContinuousHead(base, demand_env)
    if setting in STATIONARY_MU_SIGMA:
        mu, sigma = STATIONARY_MU_SIGMA[setting]
        model.D = np.exp(mu + sigma ** 2 / 2) * 365
    return model


if __name__ == "__main__":
    total_start = time.time()
    for setting in SETTINGS:
        demand_env = DEMAND_ENV_FOR_SETTING[setting]
        scaling_factor_vals = SCALING_FACTOR_FOR_DEMAND_ENV[demand_env]

        if setting in STATIONARY_MU_SIGMA:
            mu, sigma = STATIONARY_MU_SIGMA[setting]
            pcn_continuous.Inventory = make_patched_inventory(mu, sigma)
        else:
            pcn_continuous.Inventory = _RealInventory

        out_dir = f"./PCN_stochastic_objsel_test/PCN_{setting}_variation2_stochobjsel"
        os.makedirs(out_dir, exist_ok=True)

        for run_id in range(1, N_RUNS + 1):
            random.seed(run_id)
            np.random.seed(run_id)
            torch.manual_seed(run_id)

            device = "cpu"
            scaling_factor = torch.tensor([scaling_factor_vals]).to(device)

            if setting == "gs":
                with open("./daily_demand_2015_good1.pkl", "rb") as f:
                    train_seq = pickle.load(f)
                env = pcn_continuous.Inventory(scaling_factor, demand_env, train_seq)
            else:
                env = pcn_continuous.Inventory(scaling_factor, demand_env)
            env.nA = np.prod(env.action_space.shape)
            model = build_model(setting, demand_env, scaling_factor, device)

            seed_path = MOPSO_SEED_PATH[setting].format(run_id=run_id)

            logdir = f"/PCN/PCN_{setting}_variation2_stochobjsel/"
            logdir += datetime.now().strftime("%Y-%m-%d_%H-%M-%S_") + str(uuid.uuid4())[:4] + "/"

            print(f"=== {setting} | variation2 | stochastic_objective_selection | run {run_id} ===")
            start = time.time()
            non_dom_rewards, mopso_time, pcn_time = train(
                "variation2", demand_env, env, model, run_id, scaling_factor,
                learning_rate=1e-4, batch_size=256, total_steps=1e6,
                n_model_updates=100, n_step_episodes=10, max_size=500,
                noise=0.0, logdir=logdir, clip_grad_norm=None,
                pareto_seed_path=seed_path,
                stochastic_objective_selection=True,
            )
            elapsed = time.time() - start

            out_path = f"{out_dir}/PCN_{setting}_variation2_stochobjsel_run_{run_id}.csv"

            if setting == "gs":
                src = RUN_TEST_OUTPUT_TEMPLATE.format(run_id=run_id)
                shutil.copyfile(src, out_path)
                n_solutions = len(pd.read_csv(out_path))
            else:
                df = pd.DataFrame(
                    {
                        "TC": [-1 * p[0] for p in non_dom_rewards],
                        "Frequency": [-1 * p[1] for p in non_dom_rewards],
                        "Amount": [-1 * p[2] for p in non_dom_rewards],
                    }
                )
                df.to_csv(out_path, index=False)
                n_solutions = len(df)

            print(f"[{setting} | run {run_id}] {elapsed:.1f}s, {n_solutions} solutions -> {out_path}")

    print(f"\nAll done in {time.time() - total_start:.1f}s")
