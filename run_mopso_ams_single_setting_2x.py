"""Generates MOPSO-AMS non-dominated fronts at 2x the matched-budget
evaluation count (I=200, T=200 -> 40,000 evals, vs matched-budget's
200x100=20,000). Rationale: Guided PCN's total pipeline cost is MOPSO's
seed-generation cost (~1x) plus its own PCN training cost (~1x, comparable
wall-clock to MOPSO per this session's measurements), so a ~2x MOPSO budget
is the more apples-to-apples EA baseline for the full pipeline, distinct
from both the 1x matched-budget and the paper's own 7.5x (I=150,T=1000)
comparisons.

Takes one setting per process so all 6 can run in parallel (mirrors
run_combined_winners_single_setting.py's pattern).

Run in the `pcn` or `gpils` conda env:

    conda activate pcn
    python run_mopso_ams_single_setting_2x.py <setting>

<setting> in: GBM, stat_low, stat_medium, stat_high, seasonal, gs
"""

import os
import pickle
import random
import sys
import time

import numpy as np
import pandas as pd
import torch

from main_original import Inventory
from pcn_continuous import _generate_demand_sequence
from MOPSO import evaluate_solution_seasonal, get_non_dominated
from MOPSO_AMS import MOPSO_AMS

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
SCALING_FACTOR_VALS = [0.0001, 0.1, 0.1, 0.1]
N_RUNS = 10
NUM_PARTICLES = 200
NUM_ITERATIONS = 200
NUM_SUBSWARMS = 4
MAX_ARCHIVE_SIZE = 200


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
        max_archive_size=MAX_ARCHIVE_SIZE,
    )
    archive, _, _ = mopso_ams.optimize_variation(
        env, demand_env, eval_fn=seasonal_period0_eval_fn(env, demand_env)
    )

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
    setting = sys.argv[1]
    demand_env = DEMAND_ENV_FOR_SETTING[setting]
    out_dir = f"./MOPSO_AMS_2x/MOPSO_AMS_{setting}"
    os.makedirs(out_dir, exist_ok=True)

    total_start = time.time()
    for run_id in range(1, N_RUNS + 1):
        random.seed(run_id)
        np.random.seed(run_id)
        torch.manual_seed(run_id)

        print(f"=== {setting} | MOPSO-AMS (2x matched budget) | run {run_id} ===")
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
                max_archive_size=MAX_ARCHIVE_SIZE,
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

    print(f"\n{setting} all done in {time.time() - total_start:.1f}s")
