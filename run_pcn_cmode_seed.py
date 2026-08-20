"""Runs PCN's evolutionary-buffer-initialization variants (variation1 = "PCN
with buffer init.", variation2 = "Guided PCN") with the experience buffer
seeded from CMODE's already-computed Pareto front instead of MOPSO's, across
the 5 settings CMODE has results for (5 x 10 runs x 2 versions = 100 training
runs). CMODE itself is NOT re-run here -- its saved CSVs are reused as-is.

Uses pcn_continuous.py's train() via the new pareto_seed_path parameter
(added specifically for this experiment; default None preserves the
original MOPSO-seeded behavior for every existing caller untouched).

Note: train() reconstructs `env` internally for the stationary/seasonal/GBM
branches (Inventory(scaling_factor, demand_env, ...)), ignoring whatever
`env` is passed in as an argument -- so for stat_low/stat_high (which need
low/high-variance demand instead of main_original.py's hardcoded medium
variance), we monkeypatch the *name* `Inventory` inside pcn_continuous's own
module namespace to a wrapper that fixes D / action_space / demand sampling
after construction. This changes nothing in main_original.py or
pcn_continuous.py on disk -- it only affects name resolution within this
script's process.

Run in the `pcn` conda env (the tested environment for main_original.py /
pcn_continuous.py -- NOT `gpils`, which has newer/incompatible torch/numpy):

    conda activate pcn
    python run_pcn_cmode_seed.py
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

_RealInventory = pcn_continuous.Inventory  # the untouched, original class

SETTINGS = ["stat_low", "stat_medium", "stat_high", "seasonal", "GBM", "gs"]
DEMAND_ENV_FOR_SETTING = {
    "stat_low": "stationary",
    "stat_medium": "stationary",
    "stat_high": "stationary",
    "seasonal": "seasonal",
    "GBM": "GBM",
    "gs": "gs",
}
STATIONARY_MU_SIGMA = {
    "stat_low": (3.29, 0.23),
    "stat_medium": (3.22, 0.45),  # matches main_original.py's hardcoded default; patching is a no-op here
    "stat_high": (3.0, 0.8),
}
# matches main_original.py's __main__ exactly
SCALING_FACTOR_FOR_DEMAND_ENV = {
    "stationary": [0.000125, 0.1, 1.0, 0.1],
    "seasonal": [0.0001, 0.1, 0.1, 0.1],
    "GBM": [0.0001, 0.1, 0.1, 0.1],
    "gs": [0.0001, 0.1, 0.1, 0.1],
}
VERSIONS = ["variation1", "variation2"]
N_RUNS = 10

# demand_env == 'gs' is a train(2015)/test(2016) setting: train() internally calls
# run_test() as a side effect and hardcodes its save path (no per-experiment
# distinction), so we must copy that file out to our own CMODE-seed-labeled folder
# right after each run, rather than relying on train()'s return value (which for
# 'gs' reflects train-period, not the test-period numbers everyone else reports).
RUN_TEST_OUTPUT_TEMPLATE = "./PCN/original_PCN_run_test1_{version}_{run_id}.csv"


def _make_stationary_demand_fn(mu, sigma):
    def _get_demand_stationary(self, _mu=mu, _sigma=sigma):
        return np.random.lognormal(mean=_mu, sigma=_sigma)
    return _get_demand_stationary


def make_patched_inventory(mu, sigma):
    """Wraps the real Inventory class: same construction, then fixes D /
    action_space / demand sampling for the given (mu, sigma) stationary
    variance level. Non-stationary demand_env values pass through unchanged."""

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
        model.D = np.exp(mu + sigma ** 2 / 2) * 365  # ContinuousHead scales tanh output to [0, D]
    return model


if __name__ == "__main__":
    total_start = time.time()
    for setting in SETTINGS:
        demand_env = DEMAND_ENV_FOR_SETTING[setting]
        scaling_factor_vals = SCALING_FACTOR_FOR_DEMAND_ENV[demand_env]

        # Patch (or restore) the Inventory name inside pcn_continuous's namespace
        # for this setting's variance level.
        if setting in STATIONARY_MU_SIGMA:
            mu, sigma = STATIONARY_MU_SIGMA[setting]
            pcn_continuous.Inventory = make_patched_inventory(mu, sigma)
        else:
            pcn_continuous.Inventory = _RealInventory

        for version in VERSIONS:
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
                nA = np.prod(env.action_space.shape)
                env.nA = nA
                model = build_model(setting, demand_env, scaling_factor, device)

                cmode_csv = (
                    f"./CMODE/CMODE_gs/CMODE_run_gs_test1_{run_id}.csv"
                    if setting == "gs"
                    else f"./CMODE/CMODE_{setting}/CMODE_run_{setting}_{run_id}.csv"
                )

                logdir = f"/PCN/PCN_{setting}_{version}_cmode_seed/"
                logdir += datetime.now().strftime("%Y-%m-%d_%H-%M-%S_") + str(uuid.uuid4())[:4] + "/"

                start = time.time()
                non_dom_rewards, mopso_time, pcn_time = train(
                    version,
                    demand_env,
                    env,
                    model,
                    run_id,
                    scaling_factor,
                    learning_rate=1e-4,
                    batch_size=256,
                    total_steps=1e6,
                    n_model_updates=100,
                    n_step_episodes=10,
                    max_size=500,
                    noise=0.0,
                    logdir=logdir,
                    clip_grad_norm=None,
                    pareto_seed_path=cmode_csv,
                )
                elapsed = time.time() - start

                out_dir = f"./PCN/PCN_{setting}_{version}_cmode_seed"
                os.makedirs(out_dir, exist_ok=True)
                out_path = f"{out_dir}/PCN_{setting}_{version}_cmode_seed_run_{run_id}.csv"

                if setting == "gs":
                    # train()'s own return value is train-period (2015); the real
                    # test-period (2016) result was auto-saved by run_test() as a
                    # side effect of train() -- copy it out before it's overwritten
                    # by the next run.
                    src = RUN_TEST_OUTPUT_TEMPLATE.format(version=version, run_id=run_id)
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

                print(f"[{setting} | {version} | run {run_id}] {elapsed:.1f}s, {n_solutions} solutions -> {out_path}")

    print(f"\nAll done in {time.time() - total_start:.1f}s")
