CAPQL results (uniform recipe: obs-norm ON, alpha=0.05, start_steps=10000,
total_steps=300000, 10 seeds/scenario), organized by scenario.

Format: every CSV has columns TC, Frequency, Amount -- the same convention
used by the existing MOPSO baseline CSVs in ./MOPSO/ (MOPSO's own CSVs
additionally have "Q (Order Size)" and "k (Safety Stock)" decision-variable
columns, which don't apply to CAPQL's output, so those two are simply
absent here; the three shared objective columns match exactly).

Per scenario folder (GBM, stat_medium, stat_high, seasonal, stat_low, gs):
  pooled_front.csv   - all 10 seeds' fronts pooled and re-filtered via
                        non_dominated_front() (final reported result)
  seed_runs/*.csv    - each seed's own individual non-dominated front
                        (raw, not pooled) -- 10 files per scenario

Top level:
  metrics_summary.csv          - spacing/max_spread/hypervolume (main_original.py's
                                   own functions) for the pooled fronts, vs MOPSO
                                   where a MOPSO baseline exists for that scenario
  per_seed_summary_metrics.csv - same metrics but per-seed mean +- std (10 seeds)
  per_seed_per_run_metrics.csv - same metrics for each individual seed run

Note: gs collapses to 1 point in all 10/10 seeds (a known, confirmed
structural issue with obs-norm on this scenario specifically, not a data
error -- see pooled_front.csv/gs, which has exactly 1 row).
