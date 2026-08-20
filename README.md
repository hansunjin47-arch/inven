# Guided PCN: Guided Multi-objective Reinforcement Learning for Inventory Management

Code accompanying the manuscript *"Guided Multi-objective Reinforcement Learning for
Inventory Management"* (submitted to *Omega*).

The paper formulates a multi-objective continuous-review $(r, Q)$ inventory problem
(total cost, stockout frequency, stockout amount) and proposes **Guided PCN**, an
extension of the Pareto Conditioned Network (PCN) with two additions:

- **Evolutionary buffer initialization** — seeds PCN's experience buffer with
  perturbed, curriculum-scaled samples anchored on MOPSO's Pareto-optimal solutions.
- **Guided exploration** — replaces PCN's random target-return perturbation with a
  statistically informed update based on buffer-wide improvement scores.

Guided PCN is evaluated against MOPSO, MOPSO-AMS, CAPQL, and the original PCN across
stationary, seasonal, and geometric-Brownian-motion demand, plus a real-world case
study.

## Repository structure

```
MOPSO.py, MOPSO_AMS.py          Evolutionary baselines
pcn_continuous.py                 Original PCN / PCN+buffer-init / Guided PCN (core training loop)
capql_baseline.py, capql_uniform_recipe.py   CAPQL baseline
main.py, main_original.py         Simulation environment, evaluation utilities, action-trajectory plotting
logger.py, compute_metrics.py     Shared utilities (HV/IGD/SC/spacing/spread)

compare_five_methods_pbest.py     Central results loader used by all downstream analysis/plotting
grid_search_pbest_training_params.py   Training entry point (run_one) used to (re)generate PCN results

_gen_paper_tables_700.py          Generates the manuscript's result tables
_measure_comp_time_stat_low.py    Generates the computation-time table
_regen_gbm_source_max_size_700.py           Regenerates the GBM action-trajectory source data
_regen_gbm_trajectories_6methods.py         Generates the GBM Q/r trajectory figure
plot_2d_cost_projections_6methods.py        2D cost-vs-service-level projection figures
plot_5methods_pbest_total_steps_2e6.py      3D Pareto-front figures
plot_seasonal_demand.py, plot_gbm_demand.py Demand-sequence figures
read_data.py                      Real-world case-study data extraction and demand figures

figures/                          Final manuscript figures (PDF, vector)
result/gbm_trajectory_source/     Source data behind the GBM action-trajectory figure
MOPSO_tsou_ablation/, MOPSO_AMS_2x/, capql_results/, capql_results_export/, PCN_pbest_grid_free/
                                   Per-run result CSVs (10 runs x 6 demand settings) for each algorithm
grid_search_sweep_summary.csv     Summary of the hyperparameter sweep behind the adopted PCN configuration
```

## Notes on scope

This repository contains the code and result data needed to reproduce the paper's
tables and figures from `compare_five_methods_pbest.py` onward. It does not include:

- Raw real-world sales data used in the case study (proprietary retailer data).
- The many one-off scripts used during hyperparameter search and ablation exploration
  prior to settling on the reported configuration; `grid_search_sweep_summary.csv`
  summarizes the outcome of that search.

## Requirements

Python 3.x with `numpy`, `pandas`, `torch`, `matplotlib`, and `pygmo` (for hypervolume
computation). No `requirements.txt` is provided yet; dependencies were managed via a
conda environment during development.
