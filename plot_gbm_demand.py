"""Plots a representative daily demand trajectory for the GBM setting
(Figure 5 in the paper), using the exact same generator as training
(pcn_continuous._generate_demand_sequence: mu=0.1 drift, sigma=1.0 volatility,
S0=20 initial mean demand, clipped to [0, 60]).

    conda activate pcn
    python plot_gbm_demand.py
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

SEED = 1  # run_id=1 -- matches grid_search_pbest_training_params.run_one()'s
          # random.seed(run_id)/np.random.seed(run_id) immediately followed
          # by fixed_seq = _generate_demand_sequence(), nothing else consumes
          # np.random state in between for the GBM branch


def generate_demand_sequence(mu=0.1, sigma=1.0, S0=20, days=365):
    dt = 1 / days
    dB = np.random.randn(days) * np.sqrt(dt)
    B = np.cumsum(dB)
    t = np.linspace(0, 1, days)
    demand_seq = S0 * np.exp((mu - 0.5 * sigma ** 2) * t + sigma * B)
    demand_seq = np.clip(demand_seq, 0.0, 60.0)
    return demand_seq


def main():
    np.random.seed(SEED)
    demand_seq = generate_demand_sequence()

    plt.figure(figsize=(12, 4))
    plt.plot(demand_seq, label="Daily Demand")
    plt.xlabel("Day")
    plt.ylabel("Demand")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("demand_plot_GBM.pdf")
    plt.close()
    print("saved demand_plot_GBM.pdf")


if __name__ == "__main__":
    main()
