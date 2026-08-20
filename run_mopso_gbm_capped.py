"""Regenerates MOPSO's GBM baseline with the archive cap enabled
(max_archive_size=200, matching num_particles), for a fair comparison
against the archive-capped MOPSO-AMS run (run_mopso_ams_gbm_paperparams.py).

The original (uncapped) MOPSO_run_GBM_*.csv files are left untouched --
this writes to a separate directory so both remain available.

Mirrors the exact env/particle-initialization pattern used for the existing
uncapped baseline (pcn_continuous.py:660-666).

Run in the `pcn` or `gpils` conda env:

    conda activate pcn
    python run_mopso_gbm_capped.py
"""

import os
import random
import time

import numpy as np
import pandas as pd
import torch

from main_original import Inventory
from pcn_continuous import _generate_demand_sequence
from MOPSO import MOPSO

DEMAND_ENV = "GBM"
SCALING_FACTOR_VALS = [0.0001, 0.1, 0.1, 0.1]
RUN_IDS = list(range(1, 11))
OUT_DIR = f"./MOPSO_capped/MOPSO_{DEMAND_ENV}"

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
        particle_positions = np.array([np.random.uniform(low, high, size=200) for (low, high) in bounds]).T

        print(f"=== {DEMAND_ENV} | MOPSO (archive cap Q=200) | run {run_id} ===")
        start = time.time()
        mopso = MOPSO(num_particles=200, dim=2, bounds=bounds, num_iterations=100,
                       initial_positions=particle_positions, max_archive_size=200)
        solution, solution2, solution3 = mopso.optimize_variation(env, DEMAND_ENV)
        elapsed = time.time() - start

        data = {
            "Q (Order Size)": [p[0][0] for p in solution],
            "k (Safety Stock)": [p[0][1] for p in solution],
            "TC": [p[1][0] for p in solution],
            "Frequency": [p[1][1] for p in solution],
            "Amount": [p[1][2] for p in solution],
        }
        df = pd.DataFrame(data)
        out_path = f"{OUT_DIR}/MOPSO_run_{DEMAND_ENV}_{run_id}.csv"
        df.to_csv(out_path, index=False)
        print(f"[run {run_id}] {elapsed:.1f}s, {len(df)} solutions -> {out_path}")

    print(f"\nAll done in {time.time() - total_start:.1f}s")
