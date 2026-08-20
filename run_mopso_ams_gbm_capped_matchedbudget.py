"""Generates MOPSO-AMS non-dominated fronts for GBM, 10 runs, matching the
existing MOPSO baseline's evaluation budget (200 particles x 100 iterations)
for an apples-to-apples comparison.

Run in the `pcn` or `gpils` conda env:

    conda activate pcn
    python run_mopso_ams_gbm.py
"""

import os
import random
import time

import numpy as np
import pandas as pd
import torch

from main_original import Inventory
from pcn_continuous import _generate_demand_sequence
from MOPSO_AMS import MOPSO_AMS

DEMAND_ENV = "GBM"
SCALING_FACTOR_VALS = [0.0001, 0.1, 0.1, 0.1]
RUN_IDS = list(range(1, 11))
OUT_DIR = f"./MOPSO_AMS_capped_matchedbudget/MOPSO_AMS_{DEMAND_ENV}"

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    total_start = time.time()

    for run_id in RUN_IDS:
        random.seed(run_id)
        np.random.seed(run_id)
        torch.manual_seed(run_id)

        device = "cpu"
        scaling_factor = torch.tensor([SCALING_FACTOR_VALS]).to(device)
        fixed_seq = _generate_demand_sequence()
        env = Inventory(scaling_factor, DEMAND_ENV, fixed_seq)
        bounds = np.array([[0, env.D], [0, env.D]])

        print(f"=== {DEMAND_ENV} | MOPSO-AMS | run {run_id} ===")
        start = time.time()
        mopso_ams = MOPSO_AMS(num_particles=200, dim=2, bounds=bounds, num_iterations=100, num_subswarms=4)
        archive, top_n_solutions, all_particles = mopso_ams.optimize_variation(env, DEMAND_ENV)
        elapsed = time.time() - start

        data = {
            "Q (Order Size)": [p[0][0] for p in archive],
            "k (Safety Stock)": [p[0][1] for p in archive],
            "TC": [p[1][0] for p in archive],
            "Frequency": [p[1][1] for p in archive],
            "Amount": [p[1][2] for p in archive],
        }
        df = pd.DataFrame(data)
        out_path = f"{OUT_DIR}/MOPSO_AMS_run_{DEMAND_ENV}_{run_id}.csv"
        df.to_csv(out_path, index=False)
        print(f"[run {run_id}] {elapsed:.1f}s, {len(df)} solutions -> {out_path}")

    print(f"\nAll done in {time.time() - total_start:.1f}s")
