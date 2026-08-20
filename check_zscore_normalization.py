"""Checks whether choose_commands_variation's z-score-based objective
selection (which always picks TC=obj0 on GBM, per compare_improve_magnitude.py)
is due to improper normalization, or a genuine property of the buffer's
statistics -- by logging the actual |z| per objective (not just the winner)
and the underlying (best-mean gap, std) pair for each.

Run in the `pcn` conda env:

    conda activate pcn
    python check_zscore_normalization.py
"""

import random
import time

import numpy as np
import torch

import pcn_continuous
from main_original import ContinuousHead
from pcn_continuous import train

_RealInventory = pcn_continuous.Inventory

DEMAND_ENV = "GBM"
SCALING_FACTOR_VALS = [0.0001, 0.1, 0.1, 0.1]
MOPSO_SEED_PATH = "./MOPSO/MOPSO_GBM/MOPSO_run_GBM_{run_id}.csv"
VERSION = "variation2"
RUN_IDS = [1, 2]
OBJ_NAMES = {0: "TC", 1: "Frequency", 2: "Amount"}

if __name__ == "__main__":
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
        logdir = f"/PCN/diag_zscore_{DEMAND_ENV}_{VERSION}_{run_id}/"

        print(f"=== {DEMAND_ENV} | {VERSION} | run {run_id} ===")
        train(
            VERSION, DEMAND_ENV, env, model, run_id, scaling_factor,
            learning_rate=1e-4, batch_size=256, total_steps=1e6,
            n_model_updates=100, n_step_episodes=10, max_size=500,
            noise=0.0, logdir=logdir, clip_grad_norm=None,
            pareto_seed_path=seed_path,
        )

    print(f"\n===== |z-score| per objective (across {len(RUN_IDS)} runs) =====")
    for obj_i in [0, 1, 2]:
        vals = np.abs(np.array(pcn_continuous.diag_all_zscores[obj_i]))
        gaps = np.array([g for g, s in pcn_continuous.diag_gap_and_std[obj_i]])
        stds = np.array([s for g, s in pcn_continuous.diag_gap_and_std[obj_i]])
        print(
            f"  obj {obj_i} ({OBJ_NAMES[obj_i]}): |z| mean={vals.mean():.4f}, median={np.median(vals):.4f}, "
            f"max={vals.max():.4f} | gap(best-mean) mean={gaps.mean():.4f} | std mean={stds.mean():.4f}"
        )

    win_counts = {0: 0, 1: 0, 2: 0}
    n_iters = len(pcn_continuous.diag_all_zscores[0])
    for i in range(n_iters):
        zs = [abs(pcn_continuous.diag_all_zscores[o][i]) for o in [0, 1, 2]]
        win_counts[int(np.argmax(zs))] += 1
    print(f"\nwinner counts out of {n_iters} iterations: {win_counts}")

    print(f"\nAll done in {time.time() - total_start:.1f}s")
