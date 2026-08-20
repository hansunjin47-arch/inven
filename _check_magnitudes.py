import pandas as pd

for setting, path in [
    ("stat_low MOPSO", "./MOPSO/MOPSO_stationary_lognormal_3.29,0.23/MOPSO_run_stationary_1.csv"),
    ("stat_medium MOPSO", "./MOPSO/MOPSO_stationary/MOPSO_run_stationary_1.csv"),
    ("stat_high MOPSO", "./MOPSO/MOPSO_stationary_lognormal_3.0,0.8/MOPSO_run_stationary_1.csv"),
]:
    df = pd.read_csv(path)
    print(setting)
    print(df[["TC", "Frequency", "Amount"]].describe().loc[["mean", "min", "max"]])
    print()
