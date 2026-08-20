"""Regenerates MOPSO's baseline (with tsou_pbest_update=True, the pbest
tie-break fix that showed a clear improvement on GBM) for the 5 non-GBM
settings (GBM already done via run_mopso_gbm_tsou_ablation.py's pbest_only
condition), 10 runs each, at the existing matched budget (200 particles x
100 iterations).

Mirrors the exact per-setting generation logic already used for MOPSO's
existing baselines (stat_low/medium/high mu,sigma patch; seasonal
period-0-only search via optimize_seasonal + 3-period aggregate re-filter;
gs in-sample 2015 seed data).

Run in the `pcn` or `gpils` conda env:

    conda activate pcn
    python run_mopso_all_settings_pbest_tiebreak.py
"""

import os
import pickle
import random
import time
import types

import gym
import numpy as np
import pandas as pd
import torch

from main_original import Inventory
from MOPSO import MOPSO, evaluate_solution_original, evaluate_solution_seasonal, get_non_dominated

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


def run_seasonal(run_id, out_dir):
    demand_env = "seasonal"
    scaling_factor = torch.tensor([SCALING_FACTOR_VALS])
    env = build_env("seasonal", demand_env, scaling_factor)

    bounds6 = np.array([[0, env.D]] * 6)
    particle_positions6 = np.array([np.random.uniform(low, high, size=NUM_PARTICLES) for (low, high) in bounds6]).T
    mopso = MOPSO(num_particles=NUM_PARTICLES, dim=6, bounds=bounds6, num_iterations=NUM_ITERATIONS,
                   initial_positions=particle_positions6, tsou_pbest_update=True)
    fixed_mask = np.array([True] * 6)
    fixed_mask[0] = False
    fixed_mask[1] = False
    period0_archive = mopso.optimize_seasonal(fixed_mask, demand_env, period_idx=0, env=env)

    full_eval = []
    for pos, _ in period0_archive:
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
    out_path = f"{out_dir}/MOPSO_run_seasonal_{run_id}.csv"
    df.to_csv(out_path, index=False)
    return df, out_path


if __name__ == "__main__":
    total_start = time.time()

    for setting in SETTINGS:
        demand_env = DEMAND_ENV_FOR_SETTING[setting]
        out_dir = f"./MOPSO_tsou_ablation/MOPSO_{setting}_pbest_only"
        os.makedirs(out_dir, exist_ok=True)

        for run_id in range(1, N_RUNS + 1):
            random.seed(run_id)
            np.random.seed(run_id)
            torch.manual_seed(run_id)

            print(f"=== pbest_only | {setting} | run {run_id} ===")
            start = time.time()

            if setting == "seasonal":
                df, out_path = run_seasonal(run_id, out_dir)
            else:
                scaling_factor = torch.tensor([SCALING_FACTOR_VALS])
                env = build_env(setting, demand_env, scaling_factor)
                bounds = np.array([[0, env.D], [0, env.D]])
                particle_positions = np.array([np.random.uniform(low, high, size=NUM_PARTICLES) for (low, high) in bounds]).T

                mopso = MOPSO(num_particles=NUM_PARTICLES, dim=2, bounds=bounds, num_iterations=NUM_ITERATIONS,
                               initial_positions=particle_positions, tsou_pbest_update=True)
                solution, _, _ = mopso.optimize_variation(env, demand_env)

                data = {
                    "Q (Order Size)": [p[0][0] for p in solution],
                    "k (Safety Stock)": [p[0][1] for p in solution],
                    "TC": [p[1][0] for p in solution],
                    "Frequency": [p[1][1] for p in solution],
                    "Amount": [p[1][2] for p in solution],
                }
                df = pd.DataFrame(data)
                out_path = f"{out_dir}/MOPSO_run_{setting}_{run_id}.csv"
                df.to_csv(out_path, index=False)

                if setting == "gs":
                    # gs's comparison baseline is out-of-sample (2016 test1),
                    # matching MOPSO/MOPSO-AMS's existing test1 correction --
                    # see MOPSO_run_gs_test1_*.csv / reevaluate_mopso_ams_gs_test1.py
                    with open("./daily_demand_2016_good1.pkl", "rb") as f:
                        fixed_seq_2016 = pickle.load(f)
                    env_test = Inventory(scaling_factor, "gs", fixed_demand_seq=fixed_seq_2016)
                    test_results = []
                    for _, row in df.iterrows():
                        Q, k = row["Q (Order Size)"], row["k (Safety Stock)"]
                        TC, Frequency, Amount = evaluate_solution_original(Q, k, env_test, "gs")
                        test_results.append({"Q (Order Size)": Q, "k (Safety Stock)": k,
                                              "TC": TC, "Frequency": Frequency, "Amount": Amount})
                    test_df = pd.DataFrame(test_results)
                    objectives = test_df[["TC", "Frequency", "Amount"]].values * -1
                    nd_mask = get_non_dominated(objectives)
                    test1_df = test_df[nd_mask].copy()
                    test1_path = f"{out_dir}/MOPSO_run_gs_test1_{run_id}.csv"
                    test1_df.to_csv(test1_path, index=False)

            elapsed = time.time() - start
            print(f"[pbest_only | {setting} | run {run_id}] {elapsed:.1f}s, {len(df)} solutions -> {out_path}")

    print(f"\nAll done in {time.time() - total_start:.1f}s")
