"""Tests train()'s `noise` parameter (Gaussian noise added to desired_return
during gradient descent, in update_model()) on top of softmax objective
selection, for GBM/variation2.

This is a different mechanism from action_noise_scale (which perturbs the
action during rollout execution, and was already found to hurt performance
when combined with softmax). This one perturbs the *target return* used
when computing the training loss on stored transitions -- directly matching
the extension described in Reymond et al. 2024 ("Exploring the Pareto front
of multi-objective COVID-19 mitigation policies using RL"), Section 3.4.2:
"when [the transition function] is stochastic... we add a limited amount of
random noise to R_t when performing gradient descent, which reduces the
risk of overfitting." Our Inventory environment has genuinely stochastic
transitions (random daily demand), matching this justification exactly.
`noise` has been 0.0 in every experiment run so far this session.

Run in the `pcn` conda env:

    conda activate pcn
    python test_return_noise_softmax_gbm.py
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
RETURN_NOISE = 0.1
OUT_DIR = f"./PCN_stochastic_objsel_test/PCN_{DEMAND_ENV}_{VERSION}_stochobjsel_returnnoise"

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
        logdir = f"/PCN/diag_stochobjsel_returnnoise_{DEMAND_ENV}_{VERSION}_{run_id}/"
        logdir += datetime.now().strftime("%Y-%m-%d_%H-%M-%S_") + str(uuid.uuid4())[:4] + "/"

        print(f"=== {DEMAND_ENV} | {VERSION} | stochastic_objsel + return_noise={RETURN_NOISE} | run {run_id} ===")
        start = time.time()
        non_dom_rewards, mopso_time, pcn_time = train(
            VERSION, DEMAND_ENV, env, model, run_id, scaling_factor,
            learning_rate=1e-4, batch_size=256, total_steps=1e6,
            n_model_updates=100, n_step_episodes=10, max_size=500,
            noise=RETURN_NOISE, logdir=logdir, clip_grad_norm=None,
            pareto_seed_path=seed_path,
            stochastic_objective_selection=True,
        )
        elapsed = time.time() - start

        out_path = f"{OUT_DIR}/PCN_{DEMAND_ENV}_{VERSION}_stochobjsel_returnnoise_run_{run_id}.csv"
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
