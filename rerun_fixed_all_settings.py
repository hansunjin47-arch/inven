"""Reruns Original PCN, PCN with buffer init. (variation1), and Guided PCN
(variation2) across all 6 settings using the FIXED pcn_continuous.py:
  - Algorithm 2 (evolutionary buffer initialization): restored the paper's
    curriculum noise schedule, replacing the dead/active-code mismatch.
  - Algorithm 3 (guided exploration): fixed the best/worst (min/max)
    inversion bug in choose_commands_variation, which affected variation2 only.

MOPSO itself is untouched by either bug, so it is NOT re-run here -- existing
MOPSO solution CSVs are reused as buffer-init seeds via pareto_seed_path.

Results are saved under *_fixed folders, kept separate from the original
(pre-fix) results so both can still be compared.

Run in the `pcn` conda env (the tested environment for main_original.py /
pcn_continuous.py):

    conda activate pcn
    python rerun_fixed_all_settings.py
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
# existing, already-computed MOPSO solutions (unaffected by either bug) reused as seeds
MOPSO_SEED_PATH = {
    "stat_low": "./MOPSO/MOPSO_stationary_lognormal_3.29,0.23/MOPSO_run_stationary_{run_id}.csv",
    "stat_medium": "./MOPSO/MOPSO_stationary/MOPSO_run_stationary_{run_id}.csv",
    "stat_high": "./MOPSO/MOPSO_stationary_lognormal_3.0,0.8/MOPSO_run_stationary_{run_id}.csv",
    "seasonal": "./MOPSO/MOPSO_seasonal/MOPSO_run_seasonal_{run_id}.csv",
    "GBM": "./MOPSO/MOPSO_GBM/MOPSO_run_GBM_{run_id}.csv",
    "gs": "./MOPSO/MOPSO_run_gs_{run_id}.csv",  # train-period solution (not the test1 file)
}
VERSIONS = ["original", "variation1", "variation2"]
N_RUNS = 10

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

        for version in VERSIONS:
            out_dir = f"./PCN_fixed/PCN_{setting}_{version}_fixed"
            if os.path.isdir(out_dir) and len([f for f in os.listdir(out_dir) if f.endswith(".csv")]) >= N_RUNS:
                print(f"[{setting} | {version}] already complete, skipping")
                continue

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

                seed_path = MOPSO_SEED_PATH[setting].format(run_id=run_id) if version != "original" else None

                logdir = f"/PCN/PCN_{setting}_{version}_fixed/"
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
                    pareto_seed_path=seed_path,
                )
                elapsed = time.time() - start

                out_dir = f"./PCN_fixed/PCN_{setting}_{version}_fixed"
                os.makedirs(out_dir, exist_ok=True)
                out_path = f"{out_dir}/PCN_{setting}_{version}_fixed_run_{run_id}.csv"

                if setting == "gs":
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
