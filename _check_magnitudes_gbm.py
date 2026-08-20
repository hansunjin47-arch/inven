import pandas as pd

df = pd.read_csv("./MOPSO/MOPSO_GBM/MOPSO_run_GBM_1.csv")
print("GBM MOPSO")
print(df[["TC", "Frequency", "Amount"]].describe().loc[["mean", "min", "max"]])
