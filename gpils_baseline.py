"""GPI-LS baseline for the (Q, r) continuous-review inventory problem.

Algorithm: GPILSContinuousAction from morl-baselines (Alegre, Bazzan, Roijers,
Nowe, da Silva, "Sample-Efficient Multi-Objective Learning via Generalized
Policy Improvement Prioritization", AAMAS 2023, https://arxiv.org/abs/2301.07784).
Per that paper's Appendix (continuous-action section), GPI-PD/GPI-LS with
continuous actions extends TD3 to multi-objective RL (see the
GPIPDContinuousAction docstring in morl_baselines/multi_policy/gpi_pd/
gpi_pd_continuous_action.py: "It extends the TD3 algorithm to multi-objective
RL."). GPILSContinuousAction is that same class with dyna=False (no learned
dynamics model, i.e. the model-free GPI-LS variant). This script only wraps
environment/evaluation/export logic around the library's own implementation;
the algorithm itself is not reimplemented.

Environment: the existing Inventory simulation class in main_original.py is
reused as-is (reset()/step_original()/step_gs()) -- only a thin gymnasium.Env
wrapper is added here to adapt its calling convention to what GPI-LS/
mo-gymnasium expects. "stationary" and "seasonal" work standalone; "gs" and
"GBM" additionally need a fixed_demand_seq (see get_demand_fixed in
main_original.py), which can be passed through via InventoryMOEnv's
fixed_demand_seq argument.

Run in the dedicated `gpils` conda env (Python 3.10 -- required because
pycddlib, a hard dependency of morl-baselines' linear-support code, only ships
Windows wheels up to cp310). The `pcn` env used to run main.py/
main_original.py has torch/gymnasium versions far too old for morl-baselines,
so a separate env is used rather than reusing it:

    conda create -n gpils python=3.10
    conda activate gpils
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install morl-baselines gym matplotlib pywin32
    python gpils_baseline.py

Note: main_original.py unconditionally does `from pygmo import hypervolume`
at module level, but pygmo has no Windows wheel for Python 3.10 (Windows
support was dropped in pygmo 2.14+). We stub that module below purely to
satisfy the import -- compute_hypervolume() itself is never called here.
"""

import os
import random
import sys
import time
import types

import numpy as np
import pandas as pd

if "pygmo" not in sys.modules:
    _pygmo_stub = types.ModuleType("pygmo")
    _pygmo_stub.hypervolume = None
    sys.modules["pygmo"] = _pygmo_stub

import gymnasium as gym
import torch
from gymnasium import spaces

from main_original import Inventory
from morl_baselines.common.evaluation import policy_evaluation_mo
from morl_baselines.common.weights import equally_spaced_weights
from morl_baselines.multi_policy.gpi_pd.gpi_pd_continuous_action import (
    GPILSContinuousAction,
)

# ---------------------------------------------------------------------------
# Experiment configuration (mirrors main.py's demand_env / n_runs conventions)
# ---------------------------------------------------------------------------
DEMAND_ENV = "stationary"
N_RUNS = 10
TOTAL_TIMESTEPS = 150_000       # matches the GPI-LS paper's own MO-Hopper budget (Fig. 4a: 10 x 1.5e4)
TIMESTEPS_PER_ITER = 15_000     # matches the paper's Fig. 4a plotting/CCS-search granularity for MO-Hopper
NUM_EVAL_WEIGHTS_FOR_FRONT = 50  # size of the exported Pareto front
NUM_EVAL_EPISODES_FOR_FRONT = 5  # episodes averaged per weight (demand is stochastic)
REF_POINT = np.array([-100000.0, -365.0, -1000000.0])  # same convention as PCN's ref_point
EPISODE_LEN = 365
# Measured on this machine (RTX 5090) with the default net_arch=[256,256]/batch_size=128:
# CPU 205.9s vs CUDA 285.9s for an identical smoke run -- GPU was slower (kernel-launch/
# transfer overhead dominates at this network/batch size). Set to "cuda" anyway to use
# otherwise-idle GPU capacity while other CPU-bound experiments are running.
DEVICE = "cpu"


class InventoryMOEnv(gym.Env):
    """gymnasium.Env wrapper around main_original.py's Inventory class.

    Adapts Inventory.reset()/step_original()/step_gs() (which return plain
    numpy arrays/tuples, not the gymnasium 2-tuple/5-tuple convention) to the
    interface morl-baselines' GPILSContinuousAction expects. The simulation
    itself (demand, lead time, cost/stockout accounting) is untouched.
    """

    def __init__(self, demand_env="stationary", episode_len=EPISODE_LEN, fixed_demand_seq=None):
        super().__init__()
        self.demand_env = demand_env
        self.episode_len = episode_len

        scaling_factor = torch.tensor([[0.1, 0.1, 0.1, 0.1]])
        self.sim = Inventory(scaling_factor, demand_env, fixed_demand_seq=fixed_demand_seq)

        low = np.asarray(self.sim.action_space.low, dtype=np.float32)
        high = np.asarray(self.sim.action_space.high, dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.sim.max_L + 1,), dtype=np.float32)
        self.reward_space = spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.sim.reset(self.demand_env, day=0)
        return np.asarray(obs, dtype=np.float32), {}

    def step(self, action):
        if self.demand_env == "gs":
            n_obs, reward, day, action_executed = self.sim.step_gs(action)
        else:
            n_obs, reward, day, action_executed = self.sim.step_original(action, self.demand_env)
        terminated = bool(day >= self.episode_len)
        return (
            np.asarray(n_obs, dtype=np.float32),
            np.asarray(reward, dtype=np.float32),
            terminated,
            False,
            {"action_executed": action_executed},
        )


def non_dominated_front(points):
    """Keep only the non-dominated (Pareto-optimal, maximization) rows of points."""
    points = np.asarray(points)
    keep = np.ones(len(points), dtype=bool)
    for i, p in enumerate(points):
        for j, q in enumerate(points):
            if i != j and np.all(q >= p) and np.any(q > p):
                keep[i] = False
                break
    return points[keep]


def run_one_seed(run_id, demand_env, fixed_demand_seq=None, setting_label=None):
    """setting_label: name used for the output folder/file only (defaults to
    demand_env). Needed because stat_low/stat_medium/stat_high all share
    demand_env="stationary" and would otherwise overwrite each other's CSVs."""
    if setting_label is None:
        setting_label = demand_env

    random.seed(run_id)
    np.random.seed(run_id)
    torch.manual_seed(run_id)

    env = InventoryMOEnv(demand_env=demand_env, fixed_demand_seq=fixed_demand_seq)
    eval_env = InventoryMOEnv(demand_env=demand_env, fixed_demand_seq=fixed_demand_seq)

    agent = GPILSContinuousAction(
        env=env,
        gamma=0.99,  # paper Appendix 2.1, MO-Hopper: "We considered gamma = 0.99"
        learning_rate=3e-4,  # paper Appendix 2.2.2: "Adam (with learning rate 3e-4 ...)"
        batch_size=256,  # paper Appendix 2.2.2: "... mini-batches of size 256" (library default is 128)
        net_arch=[256, 256],  # paper Appendix 2.2.2, MO-Hopper: "2 layers with 256 neurons"
        gradient_updates=20,  # paper Appendix 2.2.2: "we used G = 20" (matches library default)
        per=True,
        alpha=0.6,  # paper Appendix 2.2.3: "we used alpha = 0.6 in all domains" (matches library default)
        min_priority=0.1,  # paper Appendix 2.2.3: "kappa = 0.1 for the MO-Hopper domain" (matches library default)
        policy_noise=0.02,  # paper Appendix 2.2.4, MO-Hopper: "Gaussian noise with standard deviation of 0.02"
        # tau, buffer_size, num_q_nets, delay_policy_update, noise_clip, learning_starts:
        # left at morl-baselines' own GPIPDContinuousAction/GPILSContinuousAction defaults --
        # the paper doesn't state overrides for these for MO-Hopper.
        log=False,
        seed=run_id,
        device=DEVICE,
    )

    print(f"\n[GPI-LS | setting={setting_label} | run {run_id}] training for {TOTAL_TIMESTEPS} timesteps...")
    start = time.time()
    agent.train(
        total_timesteps=TOTAL_TIMESTEPS,
        eval_env=eval_env,
        ref_point=REF_POINT,
        weight_selection_algo="gpi-ls",
        timesteps_per_iter=TIMESTEPS_PER_ITER,
        checkpoints=False,
    )
    elapsed = time.time() - start
    print(f"[run {run_id}] training done in {elapsed:.1f}s")

    # Deploy with GPI enabled: pick the best learned sub-policy per query weight.
    agent.use_gpi = True

    eval_weights = equally_spaced_weights(3, n=NUM_EVAL_WEIGHTS_FOR_FRONT)
    vec_returns = [
        policy_evaluation_mo(agent, eval_env, w, rep=NUM_EVAL_EPISODES_FOR_FRONT)[2] for w in eval_weights
    ]
    front = non_dominated_front(np.array(vec_returns))

    df = pd.DataFrame(
        {
            "TC": -front[:, 0],
            "Frequency": -front[:, 1],
            "Amount": -front[:, 2],
        }
    )

    out_dir = f"./GPILS/GPILS_{setting_label}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/GPILS_run_{setting_label}_{run_id}.csv"
    df.to_csv(out_path, index=False)
    print(f"[run {run_id}] saved {len(df)} non-dominated points to {out_path}")

    return df


if __name__ == "__main__":
    total_start = time.time()
    for run_id in range(1, N_RUNS + 1):
        run_one_seed(run_id, DEMAND_ENV)
    print(f"\nAll {N_RUNS} runs done in {time.time() - total_start:.1f}s")
