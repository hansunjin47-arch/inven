"""Generates MOPSO-AMS non-dominated fronts for stat_low, stat_medium,
stat_high, seasonal, and gs (GBM already done separately via
run_mopso_ams_gbm.py), 10 runs each, matching the existing MOPSO baseline's
evaluation budget (200 particles x 100 iterations) and per-setting
configuration (mu/sigma, fixed demand sequence) from
run_stochastic_objsel_all_settings.py.

Seasonal note: mirrors the existing MOPSO seasonal generation logic exactly
(pcn_continuous.py's MOPSO_run block, ~line 466-594): MOPSO-AMS only searches
period-0's (Q, r) via evaluate_solution_seasonal(..., period_idx=0), then
each period-0 candidate is held constant across all 3 periods to get the
aggregate (TC, Frequency, Amount), which is then re-filtered for
non-domination.

Run in the `pcn` or `gpils` conda env:

    conda activate pcn
    python run_mopso_ams_all_settings.py
"""

import os
import pickle
import random
import time

import numpy as np
import pandas as pd
import torch

from main_original import Inventory
from pcn_continuous import _generate_demand_sequence
from MOPSO import evaluate_solution_seasonal, get_non_dominated
from MOPSO_AMS import MOPSO_AMS

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
SCALING_FACTOR_VALS = [0.0001, 0.1, 0.1, 0.1]
N_RUNS = 10
NUM_PARTICLES = 200
NUM_ITERATIONS = 100
NUM_SUBSWARMS = 4


def _make_stationary_demand_fn(mu, sigma):
    def _get_demand_stationary(self, _mu=mu, _sigma=sigma):
        return np.random.lognormal(mean=_mu, sigma=_sigma)
    return _get_demand_stationary


def build_env(setting, demand_env, scaling_factor):
    if setting == "gs":
        with open("./daily_demand_2015_good1.pkl", "rb") as f:
            fixed_seq = pickle.load(f)
        env = Inventory(scaling_factor, demand_env, fixed_seq)
    elif demand_env == "GBM":
        env = Inventory(scaling_factor, demand_env, _generate_demand_sequence())
    else:
        env = Inventory(scaling_factor, demand_env)

    if setting in STATIONARY_MU_SIGMA:
        mu, sigma = STATIONARY_MU_SIGMA[setting]
        import types
        env.D = np.exp(mu + sigma ** 2 / 2) * 365
        import gym
        env.action_space = gym.spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([env.D, env.D], dtype=np.float32),
            dtype=np.float32,
        )
        env.get_demand_stationary = types.MethodType(_make_stationary_demand_fn(mu, sigma), env)

    return env


def seasonal_period0_eval_fn(env, demand_env):
    def _eval(position):
        vec = np.array([position[0], position[1], 0.0, 0.0, 0.0, 0.0])
        return evaluate_solution_seasonal(vec, demand_env, env=env, period_idx=0)
    return _eval


def run_seasonal(run_id, out_dir):
    demand_env = "seasonal"
    scaling_factor = torch.tensor([SCALING_FACTOR_VALS])
    env = build_env("seasonal", demand_env, scaling_factor)
    bounds = np.array([[0, env.D], [0, env.D]])

    mopso_ams = MOPSO_AMS(
        num_particles=NUM_PARTICLES, dim=2, bounds=bounds,
        num_iterations=NUM_ITERATIONS, num_subswarms=NUM_SUBSWARMS,
    )
    archive, _, _ = mopso_ams.optimize_variation(
        env, demand_env, eval_fn=seasonal_period0_eval_fn(env, demand_env)
    )

    # hold each period-0 candidate constant across all 3 periods, aggregate, re-filter
    full_eval = []
    for pos, _ in archive:
        Q0, r0 = pos[0], pos[1]
        total_cost = total_freq = total_amt = 0.0
        for p_idx in range(3):
            vec = np.array([Q0, r0, Q0, r0, Q0, r0])
            c, f, a = evaluate_solution_seasonal(vec, demand_env, env=env, period_idx=p_idx)
            total_cost += c
            total_freq += f
            total_amt += a
        full_eval.append((Q0, r0, total_cost, total_freq, total_amt))

    fitness = np.array([[e[2], e[3], e[4]] for e in full_eval])
    nd_mask = get_non_dominated(fitness) if len(fitness) > 0 else np.array([], dtype=bool)
    final = [full_eval[i] for i in range(len(full_eval)) if nd_mask[i]]

    data = {
        "Q1": [e[0] for e in final], "r1": [e[1] for e in final],
        "Q2": [e[0] for e in final], "r2": [e[1] for e in final],
        "Q3": [e[0] for e in final], "r3": [e[1] for e in final],
        "TC": [e[2] for e in final],
        "Frequency": [e[3] for e in final],
        "Amount": [e[4] for e in final],
    }
    df = pd.DataFrame(data)
    out_path = f"{out_dir}/MOPSO_AMS_run_seasonal_{run_id}.csv"
    df.to_csv(out_path, index=False)
    return df, out_path


if __name__ == "__main__":
    total_start = time.time()

    for setting in SETTINGS:
        demand_env = DEMAND_ENV_FOR_SETTING[setting]
        out_dir = f"./MOPSO_AMS/MOPSO_AMS_{setting}"
        os.makedirs(out_dir, exist_ok=True)

        for run_id in range(1, N_RUNS + 1):
            random.seed(run_id)
            np.random.seed(run_id)
            torch.manual_seed(run_id)

            print(f"=== {setting} | MOPSO-AMS | run {run_id} ===")
            start = time.time()

            if setting == "seasonal":
                df, out_path = run_seasonal(run_id, out_dir)
            else:
                scaling_factor = torch.tensor([SCALING_FACTOR_VALS])
                env = build_env(setting, demand_env, scaling_factor)
                bounds = np.array([[0, env.D], [0, env.D]])

                mopso_ams = MOPSO_AMS(
                    num_particles=NUM_PARTICLES, dim=2, bounds=bounds,
                    num_iterations=NUM_ITERATIONS, num_subswarms=NUM_SUBSWARMS,
                )
                archive, _, _ = mopso_ams.optimize_variation(env, demand_env)

                data = {
                    "Q (Order Size)": [p[0][0] for p in archive],
                    "k (Safety Stock)": [p[0][1] for p in archive],
                    "TC": [p[1][0] for p in archive],
                    "Frequency": [p[1][1] for p in archive],
                    "Amount": [p[1][2] for p in archive],
                }
                df = pd.DataFrame(data)
                out_path = f"{out_dir}/MOPSO_AMS_run_{setting}_{run_id}.csv"
                df.to_csv(out_path, index=False)

            elapsed = time.time() - start
            print(f"[{setting} | run {run_id}] {elapsed:.1f}s, {len(df)} solutions -> {out_path}")

    print(f"\nAll done in {time.time() - total_start:.1f}s")
