"""Re-tests action_noise_scale=0.1 on top of the softmax(|z|)-based
stochastic objective selection fix, for GBM/variation2.

Previously (deterministic argmax objective selection), action_noise_scale
was found to have "almost no effect" -- but that was because the
exploration loop contributed exactly 0% to the final non-dominated set at
the time, so nothing about action-level noise during rollouts could have
mattered. Now that softmax-based selection makes exploration contribute the
majority of the final front (78-100% across 10 GBM runs), action noise may
have a much more meaningful effect. This test checks that directly.

Run in the `pcn` conda env:

    conda activate pcn
    python test_actionnoise_on_softmax_gbm.py
"""

import os
import time
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
import torch

import pcn_continuous
from main_original import ContinuousHead
from pcn_continuous import train

_RealInventory = pcn_continuous.Inventory

DEMAND_ENV = "GBM"
SCALING_FACTOR_VALS = [0.0001, 0.1, 0.1, 0.1]
MOPSO_SEED_PATH = "./MOPSO/MOPSO_GBM/MOPSO_run_GBM_{run_id}.csv"
VERSION = "variation2"
RUN_IDS = list(range(1, 11))
ACTION_NOISE_SCALE = 0.1
OUT_DIR = f"./PCN_stochastic_objsel_test/PCN_{DEMAND_ENV}_{VERSION}_stochobjsel_actionnoise"

if __name__ == "__main__":
    import random

    os.makedirs(OUT_DIR, exist_ok=True)
    total_start = time.time()
    for run_id in RUN_IDS:
        random.seed(run_id)
        np.random.seed(run_id)
        torch.manual_seed(run_id)

        device = "cpu"
        scaling_factor = torch.tensor([SCALING_FACTOR_VALS]).to(device)
        env = _RealInventory(scaling_factor, DEMAND_ENV)
        env.nA = np.prod(env.action_space.shape)
        model = ContinuousHead(_RealInventory(scaling_factor, DEMAND_ENV).to(device), DEMAND_ENV)

        seed_path = MOPSO_SEED_PATH.format(run_id=run_id)
        logdir = f"/PCN/diag_stochobjsel_actionnoise_{DEMAND_ENV}_{VERSION}_{run_id}/"
        logdir += datetime.now().strftime("%Y-%m-%d_%H-%M-%S_") + str(uuid.uuid4())[:4] + "/"

        print(f"=== {DEMAND_ENV} | {VERSION} | stochastic_objsel + action_noise={ACTION_NOISE_SCALE} | run {run_id} ===")
        start = time.time()
        non_dom_rewards, mopso_time, pcn_time = train(
            VERSION, DEMAND_ENV, env, model, run_id, scaling_factor,
            learning_rate=1e-4, batch_size=256, total_steps=1e6,
            n_model_updates=100, n_step_episodes=10, max_size=500,
            noise=0.0, logdir=logdir, clip_grad_norm=None,
            pareto_seed_path=seed_path,
            stochastic_objective_selection=True,
            action_noise_scale=ACTION_NOISE_SCALE,
        )
        elapsed = time.time() - start

        out_path = f"{OUT_DIR}/PCN_{DEMAND_ENV}_{VERSION}_stochobjsel_actionnoise_run_{run_id}.csv"
        df = pd.DataFrame(
            {
                "TC": [-1 * p[0] for p in non_dom_rewards],
                "Frequency": [-1 * p[1] for p in non_dom_rewards],
                "Amount": [-1 * p[2] for p in non_dom_rewards],
            }
        )
        df.to_csv(out_path, index=False)

        print(f"[run {run_id}] {elapsed:.1f}s, {len(df)} solutions -> {out_path}")

    print(f"\nAll done in {time.time() - total_start:.1f}s")
