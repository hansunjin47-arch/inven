"""2D Pareto-front projections (total cost vs stockout frequency, total cost
vs stockout amount) for the stationary high-variance setting, matching the
paper's Figure 3. Regenerated with all six methods (adding MOPSO-AMS and
CAPQL to the original four) using the same finalized color palette as
plot_5methods_pbest_total_steps_2e6.py, and the current max_size=700 PCN
data (compare_five_methods_pbest.paths_for_setting).

Run in the `pcn` conda env:
    conda run -n pcn python plot_2d_cost_projections_6methods.py
"""

import os

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import pandas as pd

from compare_five_methods_pbest import METHOD_LABELS, N_RUNS, paths_for_setting

SETTING = "stat_high"
OUT_DIR_AMOUNT = "./result/pareto_2d_cost_amount"
OUT_DIR_FREQ = "./result/pareto_2d_cost_freq"
OUT_DIR_AMOUNT_BROKEN = "./result/pareto_2d_cost_amount_broken"
OUT_DIR_FREQ_BROKEN = "./result/pareto_2d_cost_freq_broken"

# Original PCN's degenerate single solution sits far above every other
# method's cost (see main.py's plot_representative_action_trajectories for
# the same pattern) -- a broken y-axis keeps the low-cost cluster readable
# while still showing that outlier.
LOW_YLIM = (0, 8)
HIGH_YLIM = (49, 52)

SERIES_STYLE = {
    "MOPSO": dict(color="#000000"),
    "MOPSO_AMS": dict(color="#1d4ed8"),
    "CAPQL": dict(color="#9333ea"),
    "PCN_original": dict(color="#16a34a"),
    "PCN_variation1": dict(color="#eab308"),
    "PCN_variation2": dict(color="#dc2626"),
}
MARKER_SIZE = 40
MARKER_ALPHA = 0.85


def load_front(path):
    df = pd.read_csv(path)
    return df[["TC", "Frequency", "Amount"]].to_numpy(dtype=float)


def plot_broken(fronts, x_col, xlabel, out_path):
    fig, (ax_high, ax_low) = plt.subplots(
        2, 1, sharex=True, figsize=(8, 6),
        gridspec_kw={"height_ratios": [1, 3]},
    )

    for method, style in SERIES_STYLE.items():
        front = fronts[method]
        ax_low.scatter(front[:, x_col], front[:, 0], label=METHOD_LABELS[method],
                       color=style["color"], s=MARKER_SIZE, alpha=MARKER_ALPHA)
        ax_high.scatter(front[:, x_col], front[:, 0],
                        color=style["color"], s=MARKER_SIZE, alpha=MARKER_ALPHA)

    ax_low.set_ylim(LOW_YLIM)
    ax_low.set_xlabel(xlabel)
    ax_low.set_ylabel("Total Cost (1e3)")
    ax_low.grid(True, alpha=0.3)
    ax_low.legend(loc="upper right")

    ax_high.set_ylim(HIGH_YLIM)
    ax_high.grid(True, alpha=0.3)
    ax_high.tick_params(labelbottom=False)

    d = 0.015
    kwargs = dict(transform=ax_high.transAxes, color="k", clip_on=False)
    ax_high.plot((-d, +d), (-d, +d), **kwargs)
    ax_high.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs.update(transform=ax_low.transAxes)
    ax_low.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_low.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.05)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_run(run_id):
    paths = paths_for_setting(SETTING)
    fronts = {m: load_front(paths[m].format(run=run_id)) for m in SERIES_STYLE}

    # ----- Cost vs Frequency -----
    fig, ax = plt.subplots(figsize=(8, 6))
    for method, style in SERIES_STYLE.items():
        front = fronts[method]
        ax.scatter(front[:, 1], front[:, 0], label=METHOD_LABELS[method],
                   color=style["color"], s=MARKER_SIZE, alpha=MARKER_ALPHA)
    ax.set_xlabel("Stockout Frequency")
    ax.set_ylabel("Total Cost (1e3)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(OUT_DIR_FREQ, exist_ok=True)
    fig.savefig(f"{OUT_DIR_FREQ}/pareto_cost_freq_stationary_{run_id}.pdf", dpi=150)
    plt.close(fig)

    # ----- Cost vs Amount -----
    fig, ax = plt.subplots(figsize=(8, 6))
    for method, style in SERIES_STYLE.items():
        front = fronts[method]
        ax.scatter(front[:, 2], front[:, 0], label=METHOD_LABELS[method],
                   color=style["color"], s=MARKER_SIZE, alpha=MARKER_ALPHA)
    ax.set_xlabel("Stockout Amount")
    ax.set_ylabel("Total Cost (1e3)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(OUT_DIR_AMOUNT, exist_ok=True)
    fig.savefig(f"{OUT_DIR_AMOUNT}/pareto_cost_amount_stationary_{run_id}.pdf", dpi=150)
    plt.close(fig)

    # ----- Broken-axis versions -----
    os.makedirs(OUT_DIR_FREQ_BROKEN, exist_ok=True)
    plot_broken(
        fronts, x_col=1, xlabel="Stockout Frequency",
        out_path=f"{OUT_DIR_FREQ_BROKEN}/pareto_cost_freq_broken_stationary_{run_id}.pdf",
    )
    os.makedirs(OUT_DIR_AMOUNT_BROKEN, exist_ok=True)
    plot_broken(
        fronts, x_col=2, xlabel="Stockout Amount",
        out_path=f"{OUT_DIR_AMOUNT_BROKEN}/pareto_cost_amount_broken_stationary_{run_id}.pdf",
    )

    print(f"[run {run_id}] saved cost-freq and cost-amount plots (flat + broken)", flush=True)


if __name__ == "__main__":
    for run_id in range(1, N_RUNS + 1):
        plot_run(run_id)
    print("done")
