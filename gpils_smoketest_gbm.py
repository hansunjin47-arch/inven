"""GPI-LS smoke test matched to the CAPQL smoke test conditions: GBM setting,
run_id=1, total_timesteps=50000 (vs GPI-LS's own default 150k/stationary in
gpils_baseline.py). Kept as a separate script so gpils_baseline.py's own
tuned config is untouched.

Run in the `gpils` conda env:

    conda activate gpils
    python gpils_smoketest_gbm.py
"""

import time

import numpy as np
import pandas as pd
import torch

from gpils_baseline import (
    InventoryMOEnv, non_dominated_front, DEVICE,
)
from morl_baselines.common.evaluation import policy_evaluation_mo
from morl_baselines.common.weights import equally_spaced_weights
from morl_baselines.multi_policy.gpi_pd.gpi_pd_continuous_action import GPILSContinuousAction


def _generate_demand_sequence(mu=0.1, sigma=1.0, S0=20, days=365):
    """Copied from pcn_continuous.py to avoid importing that module here --
    it pulls in `logger.Logger`, which requires h5py (not installed in the
    `gpils` conda env). This function itself has no such dependency."""
    dt = 1 / days
    dB = np.random.randn(days) * np.sqrt(dt)
    B = np.cumsum(dB)
    t = np.linspace(0, 1, days)
    demand_seq = S0 * np.exp((mu - 0.5 * sigma ** 2) * t + sigma * B)
    return np.clip(demand_seq, 0.0, 60.0)


TOTAL_TIMESTEPS = 50_000
TIMESTEPS_PER_ITER = 5_000
REF_POINT = np.array([-100000.0, -365.0, -1000000.0])

if __name__ == "__main__":
    run_id = 1
    import random
    random.seed(run_id)
    np.random.seed(run_id)
    torch.manual_seed(run_id)

    fixed_seq = _generate_demand_sequence()
    env = InventoryMOEnv(demand_env="GBM", fixed_demand_seq=fixed_seq)
    eval_env = InventoryMOEnv(demand_env="GBM", fixed_demand_seq=fixed_seq)

    agent = GPILSContinuousAction(
        env=env, gamma=0.99, learning_rate=3e-4, batch_size=256,
        net_arch=[256, 256], gradient_updates=20, per=True, alpha=0.6,
        min_priority=0.1, policy_noise=0.02, log=False, seed=run_id, device=DEVICE,
    )

    print(f"=== GPI-LS smoke test: GBM, total_steps={TOTAL_TIMESTEPS} ===")
    start = time.time()
    agent.train(
        total_timesteps=TOTAL_TIMESTEPS, eval_env=eval_env, ref_point=REF_POINT,
        weight_selection_algo="gpi-ls", timesteps_per_iter=TIMESTEPS_PER_ITER,
        checkpoints=False,
    )
    elapsed = time.time() - start
    print(f"Training done in {elapsed:.1f}s")

    agent.use_gpi = True
    eval_weights = equally_spaced_weights(3, n=30)
    vec_returns = [policy_evaluation_mo(agent, eval_env, w, rep=1)[2] for w in eval_weights]
    front = non_dominated_front(np.array(vec_returns))

    print(f"Non-dominated front: {len(front)} points")
    df = pd.DataFrame({"TC": -front[:, 0], "Frequency": -front[:, 1], "Amount": -front[:, 2]})
    print(df)
    df.to_csv("gpils_smoketest_GBM.csv", index=False)
