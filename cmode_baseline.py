"""CMODE baseline for the (Q, r) continuous-review inventory problem.

Algorithm: Cooperative Differential Evolution with Multiple Populations for
Multiobjective Optimization (CMODE) -- J. Wang, W. Zhang, J. Zhang,
IEEE Transactions on Cybernetics, Vol. 46, No. 12, pp. 2848-2861, 2016.
DOI: 10.1109/TCYB.2015.2490669

No public code release was found for this paper (checked: GitHub code search,
the paper text itself, general web search) -- this is a direct implementation
of the paper's Algorithm 1 (main loop), Algorithm 2 (archive DE), and Eqs.
(8)-(20) (JADE-style adaptive DE for subpopulations, jDE-style adaptive DE
for the archive, SPEA2-style kNN archive truncation).

For an M-objective problem, CMODE runs M single-objective subpopulations
(one per objective) plus one archive population; all M+1 populations
cooperate via the archive to approximate the whole Pareto front. Here
M=3 (TC, Frequency, Amount), matching the project's existing MOPSO/PCN
objectives.

Environment/fitness reuse: this script reuses Inventory (main_original.py)
and evaluate_solution_original (MOPSO.py) as-is -- the decision vector is a
fixed (Q, r) pair evaluated over a full 365-day simulation, exactly like the
project's existing MOPSO baseline. This keeps CMODE directly comparable to
MOPSO (same environment, same fitness function, same decision variables).

Run in the `gpils` conda env (already has torch/gym/matplotlib/pywin32
needed to import Inventory -- see gpils_baseline.py's docstring for why a
pygmo stub is needed and why that particular env was created):

    conda activate gpils
    python cmode_baseline.py

Notes on paper ambiguities resolved here (the paper doesn't specify these
explicitly):
  - Boundary handling for mutant/trial vectors: clipped to [low, high],
    matching MOPSO.py's own clipping convention in this project.
  - F/CR of an individual newly entering the archive from a subpopulation
    (rather than surviving from the old archive or being a jDE offspring):
    initialized to F=0.5, CR=0.9, matching the paper's stated initial archive
    values (Section III-C / Algorithm 1 line 11).
"""

import os
import pickle
import sys
import time
import types

import numpy as np
import pandas as pd

# main_original.py unconditionally does `from pygmo import hypervolume` at
# module level; pygmo has no Windows wheel for this env's Python version and
# is never actually used here (compute_hypervolume() is not called by this
# script). See gpils_baseline.py's docstring for the same issue.
if "pygmo" not in sys.modules:
    _pygmo_stub = types.ModuleType("pygmo")
    _pygmo_stub.hypervolume = None
    sys.modules["pygmo"] = _pygmo_stub

import gym  # noqa: E402  (only for rebuilding action_space bounds when patching stationary variance)
import torch  # noqa: E402  (needed only because Inventory is an nn.Module)
from main_original import Inventory  # noqa: E402
from MOPSO import evaluate_solution_original  # noqa: E402

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
# Settings match the paper's Section 5 experimental design (Guided PCN / Omega submission):
# stationary demand at 3 variance levels (Table 1-6), seasonal (Table 7-8), GBM (Table 9-10).
# The real-world case study (Section 5.4, 'gs' demand_env) needs a train(2015)/test(2016) split
# and real sales data and is handled separately (not included in SETTINGS below).
SETTINGS = ["stat_low", "stat_medium", "stat_high", "seasonal", "GBM"]
# demand_env string Inventory/evaluate_solution_original expect, per setting
DEMAND_ENV_FOR_SETTING = {
    "stat_low": "stationary",
    "stat_medium": "stationary",
    "stat_high": "stationary",
    "seasonal": "seasonal",
    "GBM": "GBM",
}
# (mu, sigma) of the underlying normal for Lognormal(mu, sigma) daily demand -- paper Sec. 5.2:
# low variance Lognormal(3.29, 0.23), medium Lognormal(3.22, 0.45), high Lognormal(3.0, 0.8).
# Only used for the "stat_*" settings; main_original.py's Inventory hardcodes the medium values,
# so stat_low/stat_high patch the instance after construction (see build_env()).
STATIONARY_MU_SIGMA = {
    "stat_low": (3.29, 0.23),
    "stat_medium": (3.22, 0.45),
    "stat_high": (3.0, 0.8),
}
N_RUNS = 10
M = 3  # objectives: TC, Frequency, Amount
NP = 20  # subpopulation size (paper Sec. IV-B: "fixed to be 20")
ARCHIVE_SIZE = 100  # paper: "all algorithms with an archive have the same archive size of 100"
# Total fitness evaluations (each = one full 365-day rollout via evaluate_solution_original).
# Matched to this project's own MOPSO budget (num_particles=200 x num_iterations=100 = 20,000),
# used consistently across main.py/pcn_continuous.py, rather than the original CMODE paper's own
# benchmark budget (3E+5 evaluations on cheap synthetic test functions) -- for a direct,
# apples-to-apples comparison against the existing MOPSO baseline in this project.
MAX_NFE = 20_000
c = 0.1  # JADE adaptation rate (Eq. 10, 13)


def _generate_gbm_demand_sequence(mu=0.1, sigma=1.0, S0=20, days=365):
    """Copied as-is from pcn_continuous.py's _generate_demand_sequence (same
    parameters as used there and matching the paper's GBM setting, Sec. 5.3)."""
    dt = 1 / days
    dB = np.random.randn(days) * np.sqrt(dt)
    B = np.cumsum(dB)
    t = np.linspace(0, 1, days)
    demand_seq = S0 * np.exp((mu - 0.5 * sigma ** 2) * t + sigma * B)
    demand_seq = np.clip(demand_seq, 0.0, 60.0)
    return demand_seq


def _make_stationary_demand_fn(mu, sigma):
    def _get_demand_stationary(self):
        return np.random.lognormal(mean=mu, sigma=sigma)
    return _get_demand_stationary


def build_env(setting):
    """Construct an Inventory instance configured for the given setting.

    'seasonal' and 'GBM' are already fully parameterized correctly in
    Inventory (main_original.py) -- no patching needed. The 3 stationary
    variance levels are patched post-construction since main_original.py
    hardcodes the medium-variance (mu=3.22, sigma=0.45) values directly
    inside get_demand_stationary()/__init__ rather than storing them as
    instance attributes (main_original.py is left untouched; this only
    overrides the bound method / attributes on this one instance).
    """
    scaling_factor = torch.tensor([[0.1, 0.1, 0.1, 0.1]])
    demand_env = DEMAND_ENV_FOR_SETTING[setting]

    if setting == "GBM":
        seq = _generate_gbm_demand_sequence(mu=0.1, sigma=1.0, S0=20, days=365)
        return Inventory(scaling_factor, demand_env, fixed_demand_seq=seq)

    env = Inventory(scaling_factor, demand_env)

    if setting in STATIONARY_MU_SIGMA:
        mu, sigma = STATIONARY_MU_SIGMA[setting]
        theoretical_mean = np.exp(mu + sigma ** 2 / 2)
        env.D = theoretical_mean * 365
        env.get_demand_stationary = types.MethodType(_make_stationary_demand_fn(mu, sigma), env)
        env.action_space = gym.spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([env.D, env.D], dtype=np.float32),
            dtype=np.float32,
        )

    return env


def clip_bounds(x, low, high):
    return np.clip(x, low, high)


def evaluate(position, env, demand_env):
    """F(X) = (TC, Frequency, Amount), all minimization objectives."""
    Q, r = position
    tc, freq, amt = evaluate_solution_original(Q, r, env, demand_env)
    return np.array([tc, freq, amt], dtype=np.float64)


def lehmer_mean(values):
    values = np.asarray(values)
    return np.sum(values ** 2) / np.sum(values)


def sample_jade_F(mu_F, rng):
    """Eq. (9): Cauchy(mu_F, 0.1), truncated to 1 if >=1, regenerated if <=0."""
    while True:
        f = mu_F + 0.1 * np.tan(np.pi * (rng.random() - 0.5))  # Cauchy sample
        if f > 0:
            return min(f, 1.0)


def sample_jade_CR(mu_CR, rng):
    """Eq. (12): Normal(mu_CR, 0.1), truncated to [0, 1]."""
    cr = rng.normal(mu_CR, 0.1)
    return min(max(cr, 0.0), 1.0)


def non_dominated_front(points, tol=0.0):
    """Indices of the non-dominated (minimization) rows of points."""
    points = np.asarray(points)
    keep = np.ones(len(points), dtype=bool)
    for i, p in enumerate(points):
        if not keep[i]:
            continue
        for j, q in enumerate(points):
            if i != j and keep[j] and np.all(q <= p + tol) and np.any(q < p - tol):
                keep[i] = False
                break
    return np.where(keep)[0]


def truncate_archive(positions, fitnesses, F_list, CR_list, size_limit):
    """SPEA2-style kNN density truncation (paper Sec. III-D): repeatedly
    remove the individual with the smallest kth-nearest-neighbor distance
    (i.e. most crowded), recomputing k = sqrt(|A|) and distances after each
    removal. Objectives are min-max normalized first ("normalized nearest
    neighbor distances", as stated in the paper)."""
    positions = list(positions)
    fitnesses = list(fitnesses)
    F_list = list(F_list)
    CR_list = list(CR_list)

    while len(positions) > size_limit:
        fit_arr = np.array(fitnesses)
        lo, hi = fit_arr.min(axis=0), fit_arr.max(axis=0)
        denom = np.where(hi - lo == 0, 1.0, hi - lo)
        norm_fit = (fit_arr - lo) / denom

        n = len(positions)
        k = max(1, int(round(np.sqrt(n))))
        dists = np.linalg.norm(norm_fit[:, None, :] - norm_fit[None, :, :], axis=-1)
        np.fill_diagonal(dists, np.inf)
        sorted_dists = np.sort(dists, axis=1)
        sigma_k = sorted_dists[:, k - 1] if k - 1 < sorted_dists.shape[1] else sorted_dists[:, -1]
        density = 1.0 / (sigma_k + 2.0)

        worst = int(np.argmax(density))  # most crowded individual
        del positions[worst]
        del fitnesses[worst]
        del F_list[worst]
        del CR_list[worst]

    return positions, fitnesses, F_list, CR_list


def optimize(env, demand_env, rng):
    """Core CMODE loop (Algorithm 1 + 2). Every fitness evaluation during
    optimization runs against `env`. Returns the final archive (positions,
    fitnesses evaluated against the same `env`)."""
    low = np.asarray(env.action_space.low, dtype=np.float64)
    high = np.asarray(env.action_space.high, dtype=np.float64)
    D = len(low)

    nfe = 0

    # --- initialization (Algorithm 1, lines 1-11) ---
    sub_pos = [rng.uniform(low, high, size=(NP, D)) for _ in range(M)]
    sub_fit = []  # sub_fit[m][i] = full F(X) vector (M,) for individual i in subpop m
    for m in range(M):
        fits = np.array([evaluate(sub_pos[m][i], env, demand_env) for i in range(NP)])
        sub_fit.append(fits)
        nfe += NP
    mu_F = [0.5] * M
    mu_CR = [0.5] * M

    # archive: extract non-dominated solutions across all subpopulations
    all_pos = np.vstack(sub_pos)
    all_fit = np.vstack(sub_fit)
    nd_idx = non_dominated_front(all_fit)
    archive_pos = [all_pos[i].copy() for i in nd_idx]
    archive_fit = [all_fit[i].copy() for i in nd_idx]
    archive_F = [0.5] * len(archive_pos)
    archive_CR = [0.9] * len(archive_pos)

    while nfe < MAX_NFE:
        # --- DE for subpopulations (Algorithm 1, lines 13-32; Eqs. 8-13) ---
        for m in range(M):
            S_F, S_CR = [], []
            trial_pos = np.zeros((NP, D))
            trial_fit_m = np.zeros(NP)  # only need objective m for selection
            trial_fit_full = np.zeros((NP, M))
            F_i_list = np.zeros(NP)
            CR_i_list = np.zeros(NP)

            best_idx = int(np.argmin(sub_fit[m][:, m]))
            X_best = sub_pos[m][best_idx]

            for i in range(NP):
                F_i = sample_jade_F(mu_F[m], rng)
                CR_i = sample_jade_CR(mu_CR[m], rng)
                F_i_list[i] = F_i
                CR_i_list[i] = CR_i

                if len(archive_pos) > 0:
                    A_r = archive_pos[rng.integers(len(archive_pos))]
                else:
                    # archive empty: pick randomly from the best individuals of the OTHER subpopulations
                    other_m = [k for k in range(M) if k != m]
                    om = other_m[rng.integers(len(other_m))]
                    other_best_idx = int(np.argmin(sub_fit[om][:, om]))
                    A_r = sub_pos[om][other_best_idx]

                others = [k for k in range(NP) if k != i]
                i1, i2 = rng.choice(others, size=2, replace=False)
                X_i = sub_pos[m][i]

                V_i = X_i + F_i * (X_best - X_i) + F_i * (sub_pos[m][i1] - sub_pos[m][i2]) + F_i * (A_r - X_i)
                V_i = clip_bounds(V_i, low, high)

                U_i = X_i.copy()
                j_rand = rng.integers(D)
                for j in range(D):
                    if rng.random() <= CR_i or j == j_rand:
                        U_i[j] = V_i[j]
                U_i = clip_bounds(U_i, low, high)

                fit_full = evaluate(U_i, env, demand_env)
                nfe += 1
                trial_pos[i] = U_i
                trial_fit_full[i] = fit_full
                trial_fit_m[i] = fit_full[m]

            for i in range(NP):
                if trial_fit_m[i] < sub_fit[m][i, m]:
                    sub_pos[m][i] = trial_pos[i]
                    sub_fit[m][i] = trial_fit_full[i]
                    S_F.append(F_i_list[i])
                    S_CR.append(CR_i_list[i])

            if len(S_F) > 0:
                mu_F[m] = (1 - c) * mu_F[m] + c * lehmer_mean(S_F)
                mu_CR[m] = (1 - c) * mu_CR[m] + c * np.mean(S_CR)

        # --- DE for archive population (Algorithm 2, Eqs. 14-16) ---
        offspring_pos, offspring_fit, offspring_F, offspring_CR = [], [], [], []
        AS = len(archive_pos)
        if AS >= 4:
            for i in range(AS):
                if rng.random() < 0.1:
                    F_i = rng.uniform(0.1, 1.0)
                else:
                    F_i = archive_F[i]
                if rng.random() < 0.1:
                    CR_i = rng.uniform(0.0, 1.0)
                else:
                    CR_i = archive_CR[i]

                others = [k for k in range(AS) if k != i]
                i1, i2, i3 = rng.choice(others, size=3, replace=False)
                V_i = archive_pos[i1] + F_i * (archive_pos[i2] - archive_pos[i3])
                V_i = clip_bounds(V_i, low, high)

                U_i = archive_pos[i].copy()
                j_rand = rng.integers(D)
                for j in range(D):
                    if rng.random() <= CR_i or j == j_rand:
                        U_i[j] = V_i[j]
                U_i = clip_bounds(U_i, low, high)

                fit_full = evaluate(U_i, env, demand_env)
                nfe += 1
                offspring_pos.append(U_i)
                offspring_fit.append(fit_full)
                offspring_F.append(F_i)
                offspring_CR.append(CR_i)

        # --- archive update scheme (Sec. III-D) ---
        cand_pos = list(np.vstack(sub_pos)) + list(archive_pos) + offspring_pos
        cand_fit = list(np.vstack(sub_fit)) + list(archive_fit) + offspring_fit
        # F/CR bookkeeping: archive survivors keep their own F/CR; offspring keep theirs;
        # anything newly admitted from a subpopulation gets the paper's stated init values.
        cand_F = [0.5] * (M * NP) + list(archive_F) + list(offspring_F)
        cand_CR = [0.9] * (M * NP) + list(archive_CR) + list(offspring_CR)

        nd_idx = non_dominated_front(np.array(cand_fit))
        archive_pos = [cand_pos[i] for i in nd_idx]
        archive_fit = [cand_fit[i] for i in nd_idx]
        archive_F = [cand_F[i] for i in nd_idx]
        archive_CR = [cand_CR[i] for i in nd_idx]

        if len(archive_pos) > ARCHIVE_SIZE:
            archive_pos, archive_fit, archive_F, archive_CR = truncate_archive(
                archive_pos, archive_fit, archive_F, archive_CR, ARCHIVE_SIZE
            )

    return np.array(archive_pos), np.array(archive_fit)


def run_one_seed(run_id, setting, rng):
    demand_env = DEMAND_ENV_FOR_SETTING[setting]
    env = build_env(setting)
    return optimize(env, demand_env, rng)


def run_one_seed_case_study(run_id, rng):
    """Real-world case study (Section 5.4): optimize CMODE against 2015
    ('good1'/sunblock) demand, then re-evaluate the resulting archive's
    (Q, r) positions -- unchanged, no further optimization -- against 2016
    demand. Matches the paper's own train/test protocol for MOPSO: "the
    solutions optimized on the training data were directly applied to the
    test period." The Inventory 'gs' branch already implements the paper's
    case-study cost structure (A=38,000 setup / 14,250 ordering / hc=9.5),
    stochastic lead time (3/4/5 days, p=0.3/0.4/0.3), and the simplified
    2D state -- no patching needed, unlike the stationary variance levels.
    """
    scaling_factor = torch.tensor([[0.1, 0.1, 0.1, 0.1]])

    with open("./daily_demand_2015_good1.pkl", "rb") as f:
        train_seq = pickle.load(f)
    with open("./daily_demand_2016_good1.pkl", "rb") as f:
        test_seq = pickle.load(f)

    train_env = Inventory(scaling_factor, "gs", fixed_demand_seq=train_seq)
    archive_pos, _ = optimize(train_env, "gs", rng)

    test_env = Inventory(scaling_factor, "gs", fixed_demand_seq=test_seq)
    test_fit = np.array([evaluate(pos, test_env, "gs") for pos in archive_pos])

    # Matches main_original.py's own MOPSO test1 pipeline (lines ~549-556): after
    # re-evaluating on 2016 test data, re-filter to solutions that are STILL
    # non-dominated under the test-period objectives (a 2015-optimal solution can
    # become dominated once scored against a different demand pattern).
    n = len(test_fit)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j and np.all(test_fit[j] <= test_fit[i]) and np.any(test_fit[j] < test_fit[i]):
                keep[i] = False
                break

    return archive_pos[keep], test_fit[keep]


if __name__ == "__main__":
    total_start = time.time()
    for setting in SETTINGS:
        for run_id in range(1, N_RUNS + 1):
            # Fresh seed per (setting, run_id): reproducible regardless of what
            # other settings/runs consumed from the RNG beforehand.
            rng = np.random.default_rng(run_id)
            np.random.seed(run_id)  # evaluate_solution_original/Inventory use global np.random

            start = time.time()
            positions, fitnesses = run_one_seed(run_id, setting, rng)
            elapsed = time.time() - start

            df = pd.DataFrame(
                {
                    "Q (Order Size)": positions[:, 0],
                    "k (Safety Stock)": positions[:, 1],
                    "TC": fitnesses[:, 0],
                    "Frequency": fitnesses[:, 1],
                    "Amount": fitnesses[:, 2],
                }
            )
            out_dir = f"./CMODE/CMODE_{setting}"
            os.makedirs(out_dir, exist_ok=True)
            out_path = f"{out_dir}/CMODE_run_{setting}_{run_id}.csv"
            df.to_csv(out_path, index=False)
            print(f"[{setting} | run {run_id}] {elapsed:.1f}s, archive size {len(df)}, saved to {out_path}")

    # Real-world case study: train on 2015 demand, evaluate (test-period) on 2016 --
    # matches MOPSO/PCN's "test1" naming convention (e.g. MOPSO_run_gs_test1_{run}.csv).
    for run_id in range(1, N_RUNS + 1):
        rng = np.random.default_rng(run_id)
        np.random.seed(run_id)

        start = time.time()
        positions, fitnesses = run_one_seed_case_study(run_id, rng)
        elapsed = time.time() - start

        df = pd.DataFrame(
            {
                "Q (Order Size)": positions[:, 0],
                "k (Safety Stock)": positions[:, 1],
                "TC": fitnesses[:, 0],
                "Frequency": fitnesses[:, 1],
                "Amount": fitnesses[:, 2],
            }
        )
        out_dir = "./CMODE/CMODE_gs"
        os.makedirs(out_dir, exist_ok=True)
        out_path = f"{out_dir}/CMODE_run_gs_test1_{run_id}.csv"
        df.to_csv(out_path, index=False)
        print(f"[case-study | run {run_id}] {elapsed:.1f}s, archive size {len(df)}, saved to {out_path}")

    print(f"\nAll settings x {N_RUNS} runs (+ case study) done in {time.time() - total_start:.1f}s")
