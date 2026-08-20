"""Compares the actual per-objective improvement/perturbation magnitude
sampled each iteration by:
  - choose_commands_original (Algorithm 1's random exploration, used by
    'original' and 'variation1'): perturbation = U(0, sigma_j)
  - choose_commands_variation (Algorithm 3's guided exploration, 'variation2'):
    improve_amount = Phi(|z_j|) * (cost_mean_j - cost_best_j)

Runs both on GBM (3 seeds each) and prints summary stats per objective
(TC=0, Frequency=1, Amount=2), reading pcn_continuous.diag_original_perturbation
/ diag_variation_improve_amount (diagnostic-only globals, not used by training).

Run in the `pcn` conda env:

    conda activate pcn
    python compare_improve_magnitude.py
"""

import random
import time
import uuid
from datetime import datetime

import numpy as np
import torch

import pcn_continuous
from main_original import ContinuousHead
from pcn_continuous import train

_RealInventory = pcn_continuous.Inventory

DEMAND_ENV = "GBM"
SCALING_FACTOR_VALS = [0.0001, 0.1, 0.1, 0.1]
MOPSO_SEED_PATH = "./MOPSO/MOPSO_GBM/MOPSO_run_GBM_{run_id}.csv"
RUN_IDS = list(range(1, 11))
OBJ_NAMES = {0: "TC", 1: "Frequency", 2: "Amount"}


def run_version(version):
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
        logdir = f"/PCN/diag_magnitude_{DEMAND_ENV}_{version}_{run_id}/"
        logdir += datetime.now().strftime("%Y-%m-%d_%H-%M-%S_") + str(uuid.uuid4())[:4] + "/"

        print(f"=== {DEMAND_ENV} | {version} | run {run_id} ===")
        start = time.time()
        train(
            version,
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
        print(f"[run {run_id}] {time.time() - start:.1f}s")


def print_summary(label, diag_dict):
    print(f"\n===== {label} =====")
    for obj_i in [0, 1, 2]:
        vals = np.array(diag_dict[obj_i])
        if len(vals) == 0:
            print(f"  obj {obj_i} ({OBJ_NAMES[obj_i]}): no samples")
            continue
        print(
            f"  obj {obj_i} ({OBJ_NAMES[obj_i]}): n={len(vals)}, "
            f"mean={vals.mean():.4f}, median={np.median(vals):.4f}, "
            f"std={vals.std():.4f}, min={vals.min():.4f}, max={vals.max():.4f}"
        )


if __name__ == "__main__":
    total_start = time.time()

    run_version("variation1")
    print_summary("variation1 (choose_commands_original): U(0, sigma) perturbation", pcn_continuous.diag_original_perturbation)

    run_version("variation2")
    print_summary("variation2 (choose_commands_variation): Phi(|z|)*(mean-best) improve_amount", pcn_continuous.diag_variation_improve_amount)

    print(f"\nAll done in {time.time() - total_start:.1f}s")
