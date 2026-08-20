import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _sub in ("baselines", "envs", "models", "utils", "experiments"):
    _p = _os.path.join(_ROOT, _sub)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
_os.chdir(_ROOT)

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import main
main.plt = plt

main.plot_representative_action_trajectories(
    mopso_csv_path="./results/mopso/MOPSO_GBM_pbest_only/MOPSO_run_GBM_1.csv",
    pcn_original_csv_path="results/gbm_trajectory_source/original_non_dominated_action_trajectories_1.csv",
    pcn_var1_csv_path="results/gbm_trajectory_source/variation1_non_dominated_action_trajectories_1.csv",
    pcn_var2_csv_path="results/gbm_trajectory_source/variation2_non_dominated_action_trajectories_1.csv",
    save_dir="results",
    demand_env="GBM",
    run_id=1,
    mopso_ams_csv_path="./results/mopso_ams/MOPSO_AMS_GBM/MOPSO_AMS_run_GBM_1.csv",
    capql_csv_path="./results/capql/uniform_300k/GBM_run1_action_trajectory.csv",
)
print("done")
