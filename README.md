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
baselines/          Evolutionary and MORL baselines
    MOPSO.py             Multi-objective particle swarm optimization
    MOPSO_AMS.py          MOPSO with an adaptive multiple selection strategy
    capql_baseline.py     Concave-augmented Pareto Q-learning (CAPQL)
    capql_uniform_recipe.py   CAPQL training entry point (uniform preference recipe)

models/              The proposed method
    pcn_continuous.py    Original PCN / PCN+buffer-init / Guided PCN (continuous action space)

envs/                Shared simulation environment
    main_original.py    Inventory simulator, demand generators, MDP definitions
    main.py              Evaluation utilities and action-trajectory plotting

utils/
    logger.py            Training logger
    compute_metrics.py    HV / IGD / set coverage / spacing / maximum spread

experiments/         Scripts that reproduce the paper's tables and figures
    compare_five_methods_pbest.py         Central results loader (paths_for_setting)
    grid_search_pbest_training_params.py   Training entry point (run_one)
    _gen_paper_tables_700.py               Result tables (Tables 1-13)
    _measure_comp_time_stat_low.py          Computation-time table (Table 13)
    _regen_gbm_source_max_size_700.py       GBM action-trajectory source data
    _regen_gbm_trajectories_6methods.py     GBM Q/r trajectory figure (Figure 6)
    plot_2d_cost_projections_6methods.py    2D cost-vs-service-level figures (Figure 3)
    plot_5methods_pbest_total_steps_2e6.py  3D Pareto-front figures (Figures 2, 8-10)
    plot_seasonal_demand.py, plot_gbm_demand.py   Demand-sequence figures (Figures 4-5)
    read_data.py                             Real-world case-study data extraction (Figure 7)

data/                Derived experiment inputs (small, non-proprietary)
    daily_demand_2015_good1.pkl, daily_demand_2016_good1.pkl
        Daily demand extracted from the real-world case study (train/test years)

results/             Per-run result CSVs (10 runs x 6 demand settings) for every algorithm
    mopso/, mopso_ams/, capql/, capql_export/, pcn/
    gbm_trajectory_source/    Source data behind the GBM action-trajectory figure
    grid_search_sweep_summary.csv   Summary of the hyperparameter sweep behind
                                     the adopted PCN configuration

figures/              Final manuscript figures (PDF, vector)
```

All scripts add `baselines/`, `envs/`, `models/`, `utils/`, and `experiments/` to
`sys.path` and `chdir` to the repository root on import, so they can be run directly
regardless of the working directory, e.g.:

```
python experiments/_gen_paper_tables_700.py
```

## Notes on scope

This repository contains the code and result data needed to reproduce the paper's
tables and figures from `experiments/compare_five_methods_pbest.py` onward. It does
not include:

- Raw real-world sales data used in the case study (proprietary retailer data).
- The many one-off scripts used during hyperparameter search and ablation exploration
  prior to settling on the reported configuration; `results/grid_search_sweep_summary.csv`
  summarizes the outcome of that search.

## Requirements

Python 3.x with `numpy`, `pandas`, `torch`, `matplotlib`, and `pygmo` (for hypervolume
computation). No `requirements.txt` is provided yet; dependencies were managed via a
conda environment during development.
