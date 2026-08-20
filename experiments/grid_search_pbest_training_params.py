import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _sub in ("baselines", "envs", "models", "utils", "experiments"):
    _p = _os.path.join(_ROOT, _sub)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
_os.chdir(_ROOT)

"""Reusable helpers for grid-searching Guided PCN (variation2) hyperparameters
against the PBEST-TIE-BREAK-FIXED MOPSO seed -- the previous 7-parameter
grid search (grid_search_training_params.py) used the original (non-fixed)
MOPSO seed to isolate each hyperparameter's own effect, but now that pbest
is the adopted final seed, verdicts should be re-checked against it since a
better seed can change which hyperparameters help.

Baseline matches rerun_variation2_pbest_seed_all_settings.py's exact
TRAIN_KWARGS (the already-existing, already-compared pbest baseline in
./PCN_pbest_seed/PCN_{setting}_variation2_pbestseed/ -- no need to rerun it).

Run in the `pcn` conda env; import run_one() from a per-setting/per-variant
driver script (see run_pbest_grid_single_setting.py).
"""

import os
import pickle
import random
import shutil
import types
import uuid
from datetime import datetime

import gym
import numpy as np
import pandas as pd
import torch

import pcn_continuous
from main_original import ContinuousHead
from pcn_continuous import train, _generate_demand_sequence

_RealInventory = pcn_continuous.Inventory

SETTINGS = ["GBM", "stat_low", "stat_medium", "stat_high", "seasonal", "gs"]
DEMAND_ENV_FOR_SETTING = {
    "GBM": "GBM",
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
    "GBM": [0.0001, 0.1, 0.1, 0.1],
}
MOPSO_PBEST_SEED_PATH = {
    "GBM": "./results/mopso/MOPSO_GBM_pbest_only/MOPSO_run_GBM_{run_id}.csv",
    "stat_low": "./results/mopso/MOPSO_stat_low_pbest_only/MOPSO_run_stat_low_{run_id}.csv",
    "stat_medium": "./results/mopso/MOPSO_stat_medium_pbest_only/MOPSO_run_stat_medium_{run_id}.csv",
    "stat_high": "./results/mopso/MOPSO_stat_high_pbest_only/MOPSO_run_stat_high_{run_id}.csv",
    "seasonal": "./results/mopso/MOPSO_seasonal_pbest_only/MOPSO_run_seasonal_{run_id}.csv",
    "gs": "./results/mopso/MOPSO_gs_pbest_only/MOPSO_run_gs_{run_id}.csv",
}
N_RUNS = 10

BASELINE_TRAIN_KWARGS = dict(
    learning_rate=1e-4, batch_size=256, total_steps=1e6,
    n_model_updates=100, n_step_episodes=10, max_size=500,
    clip_grad_norm=None, stochastic_objective_selection=True,
)

# Parameters that add ~zero extra compute (pure hyperparameter, same amount
# of arithmetic/gradient-updates as baseline): learning_rate, clip_grad_norm,
# noise (the original PCN paper's command-noise regularizer -- Gaussian noise
# added to desired_return during training, currently disabled at noise=0.0
# in this project's baseline). (total_steps / n_model_updates / batch_size
# all cost meaningfully more and are handled separately.)
#
# noise candidates chosen from the actual desired_return scale observed in
# training logs (grid_search_full10run_output.log): the TC component of
# desired_return typically ranges ~5-12 during training, so 0.1/0.3/0.5 span
# a small-to-moderate perturbation relative to that scale.
FREE_PARAM_GRID = [
    ("learning_rate", 2e-4),
    ("learning_rate", 3e-4),
    ("clip_grad_norm", 0.5),
    ("clip_grad_norm", 2.0),
    ("noise", 0.1),
    ("noise", 0.3),
    ("noise", 0.5),
    ("action_noise_scale", 0.01),
    ("action_noise_scale", 0.05),
    ("action_noise_scale", 0.1),
    ("max_size", 1000),
    ("max_size", 200),  # shrinking instead of growing -- matches MOPSO's own population size
]


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


def run_one(setting, run_id, train_kwargs, out_dir, version="variation2"):
    demand_env = DEMAND_ENV_FOR_SETTING[setting]
    scaling_factor_vals = SCALING_FACTOR_FOR_DEMAND_ENV[demand_env]

    if setting in STATIONARY_MU_SIGMA:
        mu, sigma = STATIONARY_MU_SIGMA[setting]
        pcn_continuous.Inventory = make_patched_inventory(mu, sigma)
    else:
        pcn_continuous.Inventory = _RealInventory

    random.seed(run_id)
    np.random.seed(run_id)
    torch.manual_seed(run_id)

    device = "cpu"
    scaling_factor = torch.tensor([scaling_factor_vals]).to(device)

    if setting == "gs":
        with open("data/daily_demand_2015_good1.pkl", "rb") as f:
            train_seq = pickle.load(f)
        env = pcn_continuous.Inventory(scaling_factor, demand_env, train_seq)
    elif setting == "GBM":
        fixed_seq = _generate_demand_sequence()
        env = pcn_continuous.Inventory(scaling_factor, demand_env, fixed_seq)
    else:
        env = pcn_continuous.Inventory(scaling_factor, demand_env)
    env.nA = np.prod(env.action_space.shape)
    model = build_model(setting, demand_env, scaling_factor, device)

    seed_path = MOPSO_PBEST_SEED_PATH[setting].format(run_id=run_id) if version != "original" else None

    logdir = f"/PCN/PCN_{setting}_{version}_pbestgrid/"
    logdir += datetime.now().strftime("%Y-%m-%d_%H-%M-%S_") + str(uuid.uuid4())[:4] + "/"

    train_kwargs = dict(train_kwargs)
    if version != "variation2":
        train_kwargs.pop("stochastic_objective_selection", None)

    non_dom_rewards, mopso_time, pcn_time = train(
        version, demand_env, env, model, run_id, scaling_factor,
        logdir=logdir, pareto_seed_path=seed_path,
        **train_kwargs,
    )

    out_path = f"{out_dir}/PCN_{setting}_{version}_run_{run_id}.csv"

    if setting == "gs":
        src = f"./PCN/original_PCN_run_test1_{version}_{run_id}.csv"
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

    return n_solutions
