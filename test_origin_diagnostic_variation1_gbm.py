"""Same origin diagnostic as test_ndfilter_variation2_gbm.py, but for
variation1 (PCN with buffer init., Algorithm 1's random exploration -- no
guided/Algorithm 3 mechanism) on GBM, for direct comparison against
variation2's buffer-init vs exploration-loop split.

Run in the `pcn` conda env:

    conda activate pcn
    python test_origin_diagnostic_variation1_gbm.py
"""

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
VERSION = "variation1"
RUN_IDS = list(range(1, 11))

if __name__ == "__main__":
    import random

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

        logdir = f"/PCN/diag_origin_{DEMAND_ENV}_{VERSION}_{run_id}/"
        logdir += datetime.now().strftime("%Y-%m-%d_%H-%M-%S_") + str(uuid.uuid4())[:4] + "/"

        print(f"=== {DEMAND_ENV} | {VERSION} | run {run_id} ===")
        start = time.time()
        non_dom_rewards, mopso_time, pcn_time = train(
            VERSION,
            DEMAND_ENV,
            env,
            model,
            run_id,
            scaling_factor,
            learning_rate=1e-4,
            batch_size=256,
            total_steps=1e6,
            n_model_updates=100,
            n_step_episodes=10,
            max_size=500,
            noise=0.0,
            logdir=logdir,
            clip_grad_norm=None,
            pareto_seed_path=seed_path,
        )
        elapsed = time.time() - start
        print(f"[run {run_id}] {elapsed:.1f}s, {len(non_dom_rewards)} solutions")

    print(f"\nAll done in {time.time() - total_start:.1f}s")
