"""Same as reevaluate_mopso_ams_gs_test1.py but for the paper-params
(I=150, T=1000, archive cap Q=150) MOPSO-AMS gs run, which is missing the
out-of-sample test1 post-processing that run_mopso_ams_all_settings_paperparams.py
never added for the gs branch.

Run in the `pcn` or `gpils` conda env, only after
./MOPSO_AMS_paperparams/MOPSO_AMS_gs/ has all 10 in-sample run files:

    conda activate pcn
    python reevaluate_mopso_ams_paperparams_gs_test1.py
"""

import pickle

import pandas as pd
import torch

from main_original import Inventory
from MOPSO import evaluate_solution_original, get_non_dominated

N_RUNS = 10
SCALING_FACTOR_VALS = [0.0001, 0.1, 0.1, 0.1]
IN_SAMPLE_PATH = "./MOPSO_AMS_paperparams/MOPSO_AMS_gs/MOPSO_AMS_run_gs_{run_id}.csv"
OUT_PATH = "./MOPSO_AMS_paperparams/MOPSO_AMS_gs/MOPSO_AMS_run_gs_test1_{run_id}.csv"

if __name__ == "__main__":
    with open("./daily_demand_2016_good1.pkl", "rb") as f:
        fixed_seq_2016 = pickle.load(f)

    scaling_factor = torch.tensor([SCALING_FACTOR_VALS])

    for run_id in range(1, N_RUNS + 1):
        env = Inventory(scaling_factor, "gs", fixed_demand_seq=fixed_seq_2016)

        in_sample_df = pd.read_csv(IN_SAMPLE_PATH.format(run_id=run_id))

        test_results = []
        for _, row in in_sample_df.iterrows():
            Q, k = row["Q (Order Size)"], row["k (Safety Stock)"]
            TC, Frequency, Amount = evaluate_solution_original(Q, k, env, "gs")
            test_results.append({
                "Q (Order Size)": Q,
                "k (Safety Stock)": k,
                "TC": TC,
                "Frequency": Frequency,
                "Amount": Amount,
            })

        test_df = pd.DataFrame(test_results)
        objectives = test_df[["TC", "Frequency", "Amount"]].values * -1
        is_pareto = get_non_dominated(objectives)
        pareto_df = test_df[is_pareto].copy()

        out_path = OUT_PATH.format(run_id=run_id)
        pareto_df.to_csv(out_path, index=False)
        print(f"[run {run_id}] {len(in_sample_df)} in-sample solutions -> {len(pareto_df)} out-of-sample non-dominated -> {out_path}")
