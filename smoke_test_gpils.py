"""Quick smoke test for gpils_baseline.py's GPI-LS pipeline -- verifies the
whole thing (env wrapper, agent construction, training, GPI-based eval,
front export) actually works end-to-end, without waiting for a full
300,000-timestep run. Uses a much smaller timestep budget (5,000 instead of
300,000) so we get a pass/fail answer quickly.

Run in the `gpils` conda env, unbuffered so prints show up immediately:

    conda activate gpils
    python -u smoke_test_gpils.py
"""

import time

import numpy as np
import torch

import gpils_baseline as gb

SETTING = "stat_medium"  # plain "stationary", no monkeypatch needed
DEMAND_ENV = "stationary"
RUN_ID = 1

# same total budget as the previous smoke test (~40 min), but split into 5
# iterations instead of 1, to check whether a genuine multi-point front
# (not just a single collapsed point) emerges
SMALL_TOTAL_TIMESTEPS = 15_000
SMALL_TIMESTEPS_PER_ITER = 3_000

if __name__ == "__main__":
    import random
    random.seed(RUN_ID)
    np.random.seed(RUN_ID)
    torch.manual_seed(RUN_ID)

    env = gb.InventoryMOEnv(demand_env=DEMAND_ENV)
    eval_env = gb.InventoryMOEnv(demand_env=DEMAND_ENV)

    print("[smoke test] constructing GPILSContinuousAction agent...", flush=True)
    agent = gb.GPILSContinuousAction(
        env=env,
        gamma=0.99,
        learning_rate=3e-4,
        batch_size=256,
        net_arch=[256, 256],
        gradient_updates=20,
        per=True,
        alpha=0.6,
        min_priority=0.1,
        policy_noise=0.02,
        log=False,
        seed=RUN_ID,
        device=gb.DEVICE,
    )
    print("[smoke test] agent constructed OK. Starting training...", flush=True)

    start = time.time()
    agent.train(
        total_timesteps=SMALL_TOTAL_TIMESTEPS,
        eval_env=eval_env,
        ref_point=gb.REF_POINT,
        weight_selection_algo="gpi-ls",
        timesteps_per_iter=SMALL_TIMESTEPS_PER_ITER,
        checkpoints=False,
    )
    print(f"[smoke test] training done in {time.time() - start:.1f}s total. Running GPI eval...", flush=True)

    agent.use_gpi = True
    eval_weights = gb.equally_spaced_weights(3, n=10)
    vec_returns = [
        gb.policy_evaluation_mo(agent, eval_env, w, rep=2)[2] for w in eval_weights
    ]
    front = gb.non_dominated_front(np.array(vec_returns))
    print(f"[smoke test] eval done. Non-dominated front has {len(front)} points:")
    print(front)
    print("[smoke test] PIPELINE OK.")
