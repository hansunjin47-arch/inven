"""Runs GPI-LS on GBM at the paper's own MO-Hopper training budget
(TOTAL_TIMESTEPS=150_000, already the default in gpils_baseline.py --
matches the GPI-LS paper's Fig. 4a and is close to morl-baselines' own
--num-timesteps 200000 continuous-MuJoCo benchmark convention), instead of
the much smaller 15k/50k smoke tests tried earlier. Saves results to CSV via
gpils_baseline.py's own run_one_seed (./GPILS/GPILS_GBM/GPILS_run_GBM_{run_id}.csv).

Run in the `gpils` conda env:

    conda activate gpils
    python run_gpils_gbm_paperbudget.py
"""

import time

import numpy as np

import gpils_baseline as gb


def _generate_demand_sequence(mu=0.1, sigma=1.0, S0=20, days=365):
    """Copied from pcn_continuous.py to avoid importing that module here --
    it pulls in `logger.Logger`, which requires h5py (not installed in the
    `gpils` conda env). This function itself has no such dependency."""
    dt = 1 / days
    dB = np.random.randn(days) * np.sqrt(dt)
    B = np.cumsum(dB)
    t = np.linspace(0, 1, days)
    demand_seq = S0 * np.exp((mu - 0.5 * sigma ** 2) * t + sigma * B)
    return np.clip(demand_seq, 0.0, 60.0)


N_RUNS = 10

if __name__ == "__main__":
    total_start = time.time()
    fixed_seq = _generate_demand_sequence()
    for run_id in range(1, N_RUNS + 1):
        start = time.time()
        gb.run_one_seed(run_id, "GBM", fixed_demand_seq=fixed_seq, setting_label="GBM")
        print(f"[GBM | run {run_id}] {time.time() - start:.1f}s")
    print(f"\nAll {N_RUNS} runs done in {time.time() - total_start:.1f}s")
