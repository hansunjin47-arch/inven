"""Tests a "balanced" scaling_factor for GBM/variation2 (softmax objective
selection) -- rescaling each objective by roughly 1/(typical magnitude) so
TC/Frequency/Amount all land near O(1) before being fed into the
conditioning network, instead of the current [0.0001, 0.1, 0.1, 0.1] which
(per MOPSO's own GBM solutions: TC~4.1, Freq~20, Amount~987) produces wildly
different scales (TC~0.0004, Freq~2.0, Amount~99).

Balanced scaling_factor used here: [0.25, 0.05, 0.001, 0.1]
  - TC: 4.1 * 0.25 ~= 1.0
  - Freq: 20 * 0.05 ~= 1.0
  - Amount: 987 * 0.001 ~= 1.0
  - horizon: left at 0.1 (unchanged)

The user recalls having tried balanced scaling before and finding it hurt
exploration -- this test re-checks that empirically on top of the current
(softmax) fixes, rather than relying on that recollection alone.

Run in the `pcn` conda env:

    conda activate pcn
    python test_balanced_scaling_gbm.py
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
BALANCED_SCALING_FACTOR = [0.25, 0.05, 0.001, 0.1]
MOPSO_SEED_PATH = "./MOPSO/MOPSO_GBM/MOPSO_run_GBM_{run_id}.csv"
VERSION = "variation2"
RUN_IDS = list(range(1, 11))
OUT_DIR = f"./PCN_scaling_ablation_test/PCN_{DEMAND_ENV}_{VERSION}_balancedscaling"

if __name__ == "__main__":
    import random

    os.makedirs(OUT_DIR, exist_ok=True)
    total_start = time.time()
    for run_id in RUN_IDS:
        random.seed(run_id)
        np.random.seed(run_id)
        torch.manual_seed(run_id)

        device = "cpu"
        scaling_factor = torch.tensor([BALANCED_SCALING_FACTOR]).to(device)
        env = _RealInventory(scaling_factor, DEMAND_ENV)
        env.nA = np.prod(env.action_space.shape)
        model = ContinuousHead(_RealInventory(scaling_factor, DEMAND_ENV).to(device), DEMAND_ENV)

        seed_path = MOPSO_SEED_PATH.format(run_id=run_id)
        logdir = f"/PCN/diag_balancedscaling_{DEMAND_ENV}_{VERSION}_{run_id}/"
        logdir += datetime.now().strftime("%Y-%m-%d_%H-%M-%S_") + str(uuid.uuid4())[:4] + "/"

        print(f"=== {DEMAND_ENV} | {VERSION} | balanced scaling_factor={BALANCED_SCALING_FACTOR} | run {run_id} ===")
        start = time.time()
        non_dom_rewards, mopso_time, pcn_time = train(
            VERSION, DEMAND_ENV, env, model, run_id, scaling_factor,
            learning_rate=1e-4, batch_size=256, total_steps=1e6,
            n_model_updates=100, n_step_episodes=10, max_size=500,
            noise=0.0, logdir=logdir, clip_grad_norm=None,
            pareto_seed_path=seed_path,
            stochastic_objective_selection=True,
        )
        elapsed = time.time() - start

        out_path = f"{OUT_DIR}/PCN_{DEMAND_ENV}_{VERSION}_balancedscaling_run_{run_id}.csv"
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
