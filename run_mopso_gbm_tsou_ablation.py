"""Runs MOPSO on GBM (10 runs each) under 5 conditions to isolate which
Tsou (2008)-faithfulness fix actually affects performance:

  1. baseline          -- all three False (current/original behavior)
  2. pbest_only        -- tsou_pbest_update=True only
  3. archive_removal_only -- tsou_archive_removal=True only
  4. interleaved_only  -- tsou_interleaved_loop=True only
  5. all_combined      -- all three True (fully Tsou-faithful)

Mirrors the exact env/particle-initialization pattern used for the existing
uncapped baseline (pcn_continuous.py:660-666), so condition 1 should
reproduce the existing ./MOPSO/MOPSO_GBM/ baseline exactly.

Run in the `pcn` or `gpils` conda env:

    conda activate pcn
    python run_mopso_gbm_tsou_ablation.py
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

CONDITIONS = {
    # baseline intentionally omitted -- with all three flags False the code
    # path is identical to the original, so ./MOPSO/MOPSO_GBM/ already IS
    # this condition's data (matched seeds 1-10).
    "pbest_only": dict(tsou_pbest_update=True, tsou_archive_removal=False, tsou_interleaved_loop=False),
    "archive_removal_only": dict(tsou_pbest_update=False, tsou_archive_removal=True, tsou_interleaved_loop=False),
    "interleaved_only": dict(tsou_pbest_update=False, tsou_archive_removal=False, tsou_interleaved_loop=True),
    "all_combined": dict(tsou_pbest_update=True, tsou_archive_removal=True, tsou_interleaved_loop=True),
}

if __name__ == "__main__":
    total_start = time.time()

    for cond_name, flags in CONDITIONS.items():
        out_dir = f"./MOPSO_tsou_ablation/MOPSO_{DEMAND_ENV}_{cond_name}"
        os.makedirs(out_dir, exist_ok=True)

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

            print(f"=== {cond_name} | {DEMAND_ENV} | run {run_id} ===")
            start = time.time()
            mopso = MOPSO(num_particles=200, dim=2, bounds=bounds, num_iterations=100,
                           initial_positions=particle_positions, **flags)
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
            out_path = f"{out_dir}/MOPSO_run_{DEMAND_ENV}_{run_id}.csv"
            df.to_csv(out_path, index=False)
            print(f"[{cond_name} | run {run_id}] {elapsed:.1f}s, {len(df)} solutions -> {out_path}")

    print(f"\nAll done in {time.time() - total_start:.1f}s")
