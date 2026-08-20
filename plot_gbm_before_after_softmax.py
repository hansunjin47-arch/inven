"""3D Pareto front plots for GBM, 4 methods (MOPSO, Original PCN, PCN with
buffer init., Guided PCN -- CMODE excluded), before vs after the softmax
objective-selection fix to Guided PCN. MOPSO/Original PCN/PCN with buffer
init. are identical in both plots (unaffected by the fix); only Guided
PCN's data source differs:

  - before: ./PCN_fixed/PCN_GBM_variation2_fixed (pre-session, deterministic
    argmax objective selection)
  - after:  ./PCN_stochastic_objsel_test/PCN_GBM_variation2_stochobjsel
    (softmax(|z|)-based stochastic objective selection)

Run in the `gpils` conda env:

    conda activate gpils
    python plot_gbm_before_after_softmax.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RUN_ID = 1
OUT_DIR = "./result/pareto_gbm_before_after_softmax"

PATHS_BEFORE = {
    "MOPSO": f"./MOPSO/MOPSO_GBM/MOPSO_run_GBM_{RUN_ID}.csv",
    "Original PCN": f"./PCN_fixed/PCN_GBM_original_fixed/PCN_GBM_original_fixed_run_{RUN_ID}.csv",
    "PCN with buffer init.": f"./PCN_fixed/PCN_GBM_variation1_fixed/PCN_GBM_variation1_fixed_run_{RUN_ID}.csv",
    "Guided PCN": f"./PCN_fixed/PCN_GBM_variation2_fixed/PCN_GBM_variation2_fixed_run_{RUN_ID}.csv",
}
PATHS_AFTER = {
    **{k: v for k, v in PATHS_BEFORE.items() if k != "Guided PCN"},
    "Guided PCN": f"./PCN_stochastic_objsel_test/PCN_GBM_variation2_stochobjsel/PCN_GBM_variation2_stochobjsel_run_{RUN_ID}.csv",
}

SERIES_STYLE = {
    "MOPSO": dict(marker="x", color="black"),
    "Original PCN": dict(marker="s", facecolors="none", edgecolor="blue"),
    "PCN with buffer init.": dict(marker="o", facecolors="none", edgecolor="red"),
    "Guided PCN": dict(marker="^", facecolors="none", edgecolor="green"),
}


def load_front(path):
    df = pd.read_csv(path)
    return df[["TC", "Frequency", "Amount"]].to_numpy(dtype=float)


def plot_one(paths, title, out_path):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    for method, style in SERIES_STYLE.items():
        front = load_front(paths[method])
        kwargs = dict(marker=style["marker"], s=60, label=method)
        if "color" in style:
            kwargs["color"] = style["color"]
        if "facecolors" in style:
            kwargs["facecolors"] = style["facecolors"]
        if "edgecolor" in style:
            kwargs["edgecolor"] = style["edgecolor"]
        ax.scatter(front[:, 1], front[:, 2], front[:, 0], **kwargs)  # Freq, Amount, TC

    ax.set_xlabel("Stockout Frequency")
    ax.set_ylabel("Stockout Amount")
    ax.set_zlabel("Total Cost (1e3)")
    ax.set_title(title)
    ax.legend()

    os.makedirs(OUT_DIR, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    p1 = plot_one(PATHS_BEFORE, f"GBM Pareto Front -- BEFORE (deterministic argmax), Run {RUN_ID}", f"{OUT_DIR}/gbm_before_run{RUN_ID}.png")
    print(f"saved {p1}")
    p2 = plot_one(PATHS_AFTER, f"GBM Pareto Front -- AFTER (softmax obj. selection), Run {RUN_ID}", f"{OUT_DIR}/gbm_after_run{RUN_ID}.png")
    print(f"saved {p2}")
