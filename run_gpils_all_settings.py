"""Runs the GPI-LS MORL baseline (gpils_baseline.py) across all 6 settings
used elsewhere in this project (stat_low, stat_medium, stat_high, seasonal,
GBM, gs), one setting at a time (all 10 runs for a setting complete before
the next setting starts).

gpils_baseline.py itself is left untouched except for two additive, optional
parameters on run_one_seed() (fixed_demand_seq, setting_label) needed to
support GBM/gs (which require a fixed demand sequence, like the PCN-side
experiments) and to keep stat_low/stat_medium/stat_high from overwriting
each other's output folder (they all share demand_env="stationary").

For stat_low/stat_high, main_original.Inventory's hardcoded (mu=3.22,
sigma=0.45) stationary demand is overridden the same way
rerun_fixed_all_settings.py does it for the PCN-side experiments: by
monkeypatching gpils_baseline's own `Inventory` name (InventoryMOEnv looks it
up in gpils_baseline's module namespace at call time, so reassigning it here
after import is enough -- gpils_baseline.py's own algorithm code is not
touched).

Run in the `gpils` conda env (see gpils_baseline.py's docstring for setup):

    conda activate gpils
    python run_gpils_all_settings.py
"""

import pickle
import random
import types

import numpy as np
import torch

import gpils_baseline
from main_original import Inventory as _RealInventory

_RealInventory_ref = gpils_baseline.Inventory  # unmodified class, for restoring on stat_medium/seasonal/GBM/gs

STATIONARY_MU_SIGMA = {
    "stat_low": (3.29, 0.23),
    "stat_high": (3.0, 0.8),
}
SETTINGS = ["stat_low", "stat_medium", "stat_high", "seasonal", "GBM", "gs"]
DEMAND_ENV_FOR_SETTING = {
    "stat_low": "stationary",
    "stat_medium": "stationary",
    "stat_high": "stationary",
    "seasonal": "seasonal",
    "GBM": "GBM",
    "gs": "gs",
}
N_RUNS = 10


def _generate_gbm_demand_sequence(mu=0.1, sigma=1.0, S0=20, days=365):
    """Matches pcn_continuous.py's _generate_demand_sequence exactly, copied
    inline to avoid importing pcn_continuous (and its unconditional pygmo
    import) into the gpils env."""
    dt = 1 / days
    dB = np.random.randn(days) * np.sqrt(dt)
    B = np.cumsum(dB)
    t = np.linspace(0, 1, days)
    demand_seq = S0 * np.exp((mu - 0.5 * sigma ** 2) * t + sigma * B)
    demand_seq = np.clip(demand_seq, 0.0, 60.0)
    return demand_seq


def _make_stationary_demand_fn(mu, sigma):
    def _get_demand_stationary(self, _mu=mu, _sigma=sigma):
        return np.random.lognormal(mean=_mu, sigma=_sigma)
    return _get_demand_stationary


def make_patched_inventory(mu, sigma):
    """Wraps main_original.Inventory: same construction, then fixes D /
    action_space / demand sampling for the given (mu, sigma) stationary
    variance level."""

    def patched(scaling_factor, demand_env, fixed_demand_seq=None):
        env = _RealInventory(scaling_factor, demand_env, fixed_demand_seq)
        D = np.exp(mu + sigma ** 2 / 2) * 365
        env.D = D
        import gym
        env.action_space = gym.spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([D, D], dtype=np.float32),
            dtype=np.float32,
        )
        env.get_demand_stationary = types.MethodType(_make_stationary_demand_fn(mu, sigma), env)
        return env

    return patched


if __name__ == "__main__":
    import time

    total_start = time.time()
    for setting in SETTINGS:
        demand_env = DEMAND_ENV_FOR_SETTING[setting]

        if setting in STATIONARY_MU_SIGMA:
            mu, sigma = STATIONARY_MU_SIGMA[setting]
            gpils_baseline.Inventory = make_patched_inventory(mu, sigma)
        else:
            gpils_baseline.Inventory = _RealInventory_ref

        print(f"\n===== setting: {setting} (demand_env={demand_env}) =====")
        setting_start = time.time()
        for run_id in range(1, N_RUNS + 1):
            random.seed(run_id)
            np.random.seed(run_id)
            torch.manual_seed(run_id)

            fixed_demand_seq = None
            if setting == "GBM":
                fixed_demand_seq = _generate_gbm_demand_sequence()
            elif setting == "gs":
                with open("./daily_demand_2015_good1.pkl", "rb") as f:
                    fixed_demand_seq = pickle.load(f)

            gpils_baseline.run_one_seed(
                run_id, demand_env, fixed_demand_seq=fixed_demand_seq, setting_label=setting
            )

        print(f"[{setting}] done in {time.time() - setting_start:.1f}s")

    print(f"\nAll settings done in {time.time() - total_start:.1f}s")
