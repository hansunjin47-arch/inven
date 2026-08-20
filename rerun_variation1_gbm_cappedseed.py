"""Reruns variation1 (PCN with buffer init.) on GBM, seeded from the
archive-capped MOPSO front (MOPSO_capped/MOPSO_GBM/, Q=200) instead of the
original uncapped MOPSO front -- needed because Algorithm 2's evolutionary
buffer initialization is seeded directly from MOPSO's non-dominated set, so
changing MOPSO's archive mechanism changes variation1's seed data too.

Run in the `pcn` conda env (after run_mopso_gbm_capped.py has completed):

    conda activate pcn
    python rerun_variation1_gbm_cappedseed.py
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
MOPSO_SEED_PATH = "./MOPSO_capped/MOPSO_GBM/MOPSO_run_GBM_{run_id}.csv"
VERSION = "variation1"
RUN_IDS = list(range(1, 11))
OUT_DIR = f"./PCN_capped_seed/PCN_{DEMAND_ENV}_{VERSION}_cappedseed"

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

        logdir = f"/PCN/{VERSION}_{DEMAND_ENV}_cappedseed_{run_id}/"
        logdir += datetime.now().strftime("%Y-%m-%d_%H-%M-%S_") + str(uuid.uuid4())[:4] + "/"

        print(f"=== {DEMAND_ENV} | {VERSION} | capped-MOPSO seed | run {run_id} ===")
        start = time.time()
        non_dom_rewards, mopso_time, pcn_time = train(
            VERSION, DEMAND_ENV, env, model, run_id, scaling_factor,
            learning_rate=1e-4, batch_size=256, total_steps=1e6,
            n_model_updates=100, n_step_episodes=10, max_size=500,
            noise=0.0, logdir=logdir, clip_grad_norm=None,
            pareto_seed_path=seed_path,
        )
        elapsed = time.time() - start

        out_path = f"{OUT_DIR}/PCN_{DEMAND_ENV}_{VERSION}_cappedseed_run_{run_id}.csv"
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
