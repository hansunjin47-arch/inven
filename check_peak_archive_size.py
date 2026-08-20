"""Diagnostic: for the 5 non-GBM settings (stat_low, stat_medium, stat_high,
seasonal, gs), tracks the PEAK archive size reached at any point DURING
optimization (not just the final size) for both MOPSO and MOPSO-AMS,
uncapped. This directly answers whether an archive cap of Q=200 (MOPSO) or
Q=num_particles (MOPSO-AMS) would ever have fired mid-run, closing the gap
in the earlier final-size-only inference (final size can be smaller than
the peak, since later non-dominated points can dominate and remove earlier
archive members).

Run in the `pcn` or `gpils` conda env:

    conda activate pcn
    python check_peak_archive_size.py
"""

import pickle
import random
import types

import gym
import numpy as np
import torch

from main_original import Inventory
from MOPSO import MOPSO, evaluate_solution_seasonal, get_non_dominated
from MOPSO_AMS import MOPSO_AMS

N_CHECK_RUNS = 3  # a handful of seeds per setting is enough to answer "does it ever get close to 200?"
SCALING_FACTOR_VALS = [0.0001, 0.1, 0.1, 0.1]
STATIONARY_MU_SIGMA = {
    "stat_low": (3.29, 0.23),
    "stat_medium": (3.22, 0.45),
    "stat_high": (3.0, 0.8),
}


def _make_stationary_demand_fn(mu, sigma):
    def _get_demand_stationary(self, _mu=mu, _sigma=sigma):
        return np.random.lognormal(mean=_mu, sigma=_sigma)
    return _get_demand_stationary


def build_env(setting, demand_env, scaling_factor):
    if setting == "gs":
        with open("./daily_demand_2015_good1.pkl", "rb") as f:
            fixed_seq = pickle.load(f)
        env = Inventory(scaling_factor, demand_env, fixed_seq)
    else:
        env = Inventory(scaling_factor, demand_env)

    if setting in STATIONARY_MU_SIGMA:
        mu, sigma = STATIONARY_MU_SIGMA[setting]
        env.D = np.exp(mu + sigma ** 2 / 2) * 365
        env.action_space = gym.spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([env.D, env.D], dtype=np.float32),
            dtype=np.float32,
        )
        env.get_demand_stationary = types.MethodType(_make_stationary_demand_fn(mu, sigma), env)

    return env


def check_mopso_ams(setting, demand_env, run_id):
    random.seed(run_id)
    np.random.seed(run_id)
    torch.manual_seed(run_id)

    scaling_factor = torch.tensor([SCALING_FACTOR_VALS])
    env = build_env(setting, demand_env, scaling_factor)
    bounds = np.array([[0, env.D], [0, env.D]])

    if setting == "seasonal":
        def eval_fn(position):
            vec = np.array([position[0], position[1], 0.0, 0.0, 0.0, 0.0])
            return evaluate_solution_seasonal(vec, demand_env, env=env, period_idx=0)
        mopso_ams = MOPSO_AMS(num_particles=200, dim=2, bounds=bounds, num_iterations=100,
                               num_subswarms=4, max_archive_size=float("inf"))
        mopso_ams.optimize_variation(env, demand_env, eval_fn=eval_fn)
    else:
        mopso_ams = MOPSO_AMS(num_particles=200, dim=2, bounds=bounds, num_iterations=100,
                               num_subswarms=4, max_archive_size=float("inf"))
        mopso_ams.optimize_variation(env, demand_env)

    return mopso_ams.peak_archive_size


def check_mopso(setting, demand_env, run_id):
    random.seed(run_id)
    np.random.seed(run_id)
    torch.manual_seed(run_id)

    scaling_factor = torch.tensor([SCALING_FACTOR_VALS])
    env = build_env(setting, demand_env, scaling_factor)
    bounds = np.array([[0, env.D], [0, env.D]])
    particle_positions = np.array([np.random.uniform(low, high, size=200) for (low, high) in bounds]).T

    if setting == "seasonal":
        bounds6 = np.array([[0, env.D], [0, env.D], [0, env.D], [0, env.D], [0, env.D], [0, env.D]])
        particle_positions6 = np.array([np.random.uniform(low, high, size=200) for (low, high) in bounds6]).T
        mopso = MOPSO(num_particles=200, dim=6, bounds=bounds6, num_iterations=100,
                       initial_positions=particle_positions6)
        fixed_mask = np.array([True] * 6)
        fixed_mask[0] = False
        fixed_mask[1] = False
        mopso.optimize_seasonal(fixed_mask, demand_env, period_idx=0, env=env)
    else:
        mopso = MOPSO(num_particles=200, dim=2, bounds=bounds, num_iterations=100,
                       initial_positions=particle_positions)
        mopso.optimize_variation(env, demand_env)

    return mopso.peak_archive_size


if __name__ == "__main__":
    demand_env_for_setting = {
        "stat_low": "stationary", "stat_medium": "stationary", "stat_high": "stationary",
        "seasonal": "seasonal", "gs": "gs",
    }

    for setting, demand_env in demand_env_for_setting.items():
        mopso_peaks = [check_mopso(setting, demand_env, run_id) for run_id in range(1, N_CHECK_RUNS + 1)]
        ams_peaks = [check_mopso_ams(setting, demand_env, run_id) for run_id in range(1, N_CHECK_RUNS + 1)]
        print(f"{setting:12s}: MOPSO peak={mopso_peaks}  MOPSO-AMS peak={ams_peaks}")
