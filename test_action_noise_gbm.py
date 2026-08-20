"""Tests whether enabling action-level exploration noise during training
rollouts (choose_action's noise_std, currently hardcoded to 0 in the "_fixed"
results) helps Guided PCN (variation2) escape the early plateau observed on
GBM. Uses action_noise_scale=0.1, matching the continuous-action PCN reference
implementation this project's paper cites (pcn-covid-main/pcn.py:169-173,
which uses std=0.1 in its [0,1]-normalized action space).

Everything else (total_steps, MOPSO seed, all other hyperparameters) is kept
identical to rerun_fixed_all_settings.py's GBM/variation2 run, so results are
directly comparable. Saved to a separate folder -- does not touch the
existing PCN_fixed/PCN_GBM_variation2_fixed results.

Run in the `pcn` conda env:

    conda activate pcn
    python test_action_noise_gbm.py
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
VERSION = "variation2"
N_RUNS = 10
ACTION_NOISE_SCALE = 0.1

OUT_DIR = f"./PCN_noise_test/PCN_{DEMAND_ENV}_{VERSION}_actionnoise"

if __name__ == "__main__":
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    total_start = time.time()
    for run_id in range(1, N_RUNS + 1):
        import random
        random.seed(run_id)
        np.random.seed(run_id)
        torch.manual_seed(run_id)

        device = "cpu"
        scaling_factor = torch.tensor([SCALING_FACTOR_VALS]).to(device)
        env = _RealInventory(scaling_factor, DEMAND_ENV)
        env.nA = np.prod(env.action_space.shape)
        model = ContinuousHead(_RealInventory(scaling_factor, DEMAND_ENV).to(device), DEMAND_ENV)

        seed_path = MOPSO_SEED_PATH.format(run_id=run_id)

        logdir = f"/PCN/PCN_{DEMAND_ENV}_{VERSION}_actionnoise/"
        logdir += datetime.now().strftime("%Y-%m-%d_%H-%M-%S_") + str(uuid.uuid4())[:4] + "/"

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
            action_noise_scale=ACTION_NOISE_SCALE,
        )
        elapsed = time.time() - start

        out_path = f"{OUT_DIR}/PCN_{DEMAND_ENV}_{VERSION}_actionnoise_run_{run_id}.csv"
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
