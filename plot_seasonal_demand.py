"""Plots the representative daily demand sequence for the seasonal setting
(Figure 4 in the paper): TruncNormal(10, 2^2) for days 1-120, TruncGamma(10, 1)
for days 121-240, TruncPoisson(10) for days 241-365, each truncated to
[0, 3*mean) -- replicating InventorySim.get_demand_seasonal() in
main_original.py (mu1=10.0, std1=2.0). Annotated with period-divider lines
and per-period distribution labels, matching the original published figure.

    conda activate pcn
    python plot_seasonal_demand.py
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

DAYS = 365
SEED = 0
PERIOD1_END = 120
PERIOD2_END = 240


def get_demand_seasonal(day, rng):
    if day <= PERIOD1_END:  # period 1: truncated normal
        mu, std = 10.0, 2.0
        truncated_point = mu * 3
        while True:
            d = rng.normal(mu, std)
            if 0.0 <= d < truncated_point:
                return d
    elif day <= PERIOD2_END:  # period 2: truncated gamma
        alpha, beta = 10.0, 1.0
        truncated_point = alpha / beta * 3
        while True:
            d = rng.gamma(alpha, 1.0 / beta)
            if 0.0 <= d < truncated_point:
                return d
    else:  # period 3: truncated poisson
        lamb = 10
        truncated_point = lamb * 3
        while True:
            d = rng.poisson(lamb)
            if 0.0 <= d < truncated_point:
                return d


def main():
    rng = np.random.RandomState(SEED)
    demand_seq = [get_demand_seasonal(day, rng) for day in range(1, DAYS + 1)]

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(demand_seq, label="Daily Demand", color="tab:blue", linewidth=1)

    ax.axvline(PERIOD1_END, color="red", linestyle="--", label="End of Period 1")
    ax.axvline(PERIOD2_END, color="green", linestyle="--", label="End of Period 2")

    y_top = 26
    ax.text(PERIOD1_END / 2, y_top, "Period 1: Trunc. Normal\n$\\mu=10.0, \\sigma=2.0$",
            color="red", ha="center", va="top", fontsize=11)
    ax.text((PERIOD1_END + PERIOD2_END) / 2, y_top, "Period 2: Trunc. Gamma\n$\\alpha=10.0, \\beta=1.0$",
            color="green", ha="center", va="top", fontsize=11)
    ax.text((PERIOD2_END + DAYS) / 2, y_top, "Period 3: Trunc. Poisson\n$\\lambda=10.0$",
            color="purple", ha="center", va="top", fontsize=11)

    ax.set_xlabel("Day")
    ax.set_ylabel("Demand")
    ax.set_ylim(0, 27)
    ax.set_xlim(0, DAYS)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig("demand_plot_seasonal.pdf")
    plt.close(fig)
    print("saved demand_plot_seasonal.pdf")


if __name__ == "__main__":
    main()
