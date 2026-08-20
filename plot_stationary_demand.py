"""Plots the daily demand sequence for each of the 3 stationary variance
levels (low/medium/high), one figure per run (matching the fixed_seq_plot_*
convention already used for GBM), using the same per-run seeding convention
(np.random.seed(run_id)) as the actual experiments. Demand is i.i.d.
Lognormal(mu, sigma) each day -- no genuine trend is expected, this is for
visually comparing how much day-to-day noise each variance level produces.

    conda activate pcn
    python plot_stationary_demand.py
"""

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STATIONARY_MU_SIGMA = {
    "stat_low": (3.29, 0.23),
    "stat_medium": (3.22, 0.45),
    "stat_high": (3.0, 0.8),
}
N_RUNS = 10
DAYS = 365


def plot_setting_run(setting, mu, sigma, run_id):
    np.random.seed(run_id)
    demand_seq = np.random.lognormal(mean=mu, sigma=sigma, size=DAYS)

    plt.figure(figsize=(12, 4))
    plt.plot(demand_seq, label="Generated Demand Sequence")
    plt.xlabel("Day")
    plt.ylabel("Demand")
    plt.title(f"Demand Sequence ({setting}, Lognormal({mu}, {sigma}))")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    os.makedirs(f"demand_plot/{setting}", exist_ok=True)
    out_path = f"demand_plot/{setting}/run_{run_id}.png"
    plt.savefig(out_path)
    plt.close()
    return out_path


if __name__ == "__main__":
    for setting, (mu, sigma) in STATIONARY_MU_SIGMA.items():
        for run_id in range(1, N_RUNS + 1):
            path = plot_setting_run(setting, mu, sigma, run_id)
            print(f"[{setting} | run {run_id}] saved {path}")
