import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _sub in ("baselines", "envs", "models", "utils", "experiments"):
    _p = _os.path.join(_ROOT, _sub)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
_os.chdir(_ROOT)

import os
import gym
import numpy as np
from scipy.stats import truncnorm
import torch.nn as nn
import random
from scipy.stats import gamma
from pygmo import hypervolume
import pandas as pd

class ContinuousHead(nn.Module):
    def __init__(self, base):
        super(ContinuousHead, self).__init__()
        self.base = base
        self.alpha = 8.0
        self.theta = 1.0
        self.L = 3
        self.D = self.alpha * self.theta * 365  # average annual demand
        self.sigma_L = np.sqrt(self.alpha) * self.theta * np.sqrt(self.L)  # (gamma) std_daily =  sqrt(alpha) * theta

    def forward(self, state, desired_return, desired_horizon):
        '''function for embedding'''

        x = self.base(state, desired_return, desired_horizon)
        x = torch.tanh(x)  # x ∈ [-1, 1]
        low = torch.tensor([0.0, 0.0], dtype=x.dtype, device=x.device)
        high = torch.tensor([self.D, self.D / self.sigma_L], dtype=x.dtype, device=x.device)
        scaled = 0.5 * (x + 1) * (high - low) + low  # scaling : Q ∈ [0, D], k ∈ [0, D/sigma_L]

        return scaled


class Inventory(nn.Module):
    def __init__(self, scaling_factor):
        super(Inventory, self).__init__()
        self.D = self.alpha * self.theta * 365  # average annual demand
        self.sigma_L = np.sqrt(self.alpha) * self.theta * np.sqrt(self.L)  # standard deviation of demand during lead time
        self.A = 80  # ordering cost
        self.c = 27.5  # unit item cost
        self.h = 0.26  # inventory carrying rate
        self.L = 3  # lead time

        self.alpha = 8.0
        self.theta = 1.0

        '''self.D = 104 # average annual demand
        self.A = 20 # ordering cost
        self.c = 350  # unit item cost
        self.h = 0.24 # inventory carrying rate
        self.sigma_L = 1 # standard deviation of demand during lead time
        self.L = 14 # lead time'''

        self.action_space = gym.spaces.Box(low=np.array([0, 0]), high=np.array([self.D, self.D / self.sigma_L]), dtype=np.float32)  # action space(Q,k)
        self.scaling_factor = scaling_factor
        self.s_emb = nn.Sequential(nn.Linear(self.L + 1, 64), nn.Sigmoid())
        self.c_emb = nn.Sequential(nn.Linear(4, 64), nn.Sigmoid())  # [desired_return, desired_horizon]
        self.fc = nn.Sequential(nn.Linear(64, 2))  # fully connected layer (latent representation)

    def forward(self, state, desired_return, desired_horizon):
        '''function for embedding'''

        c = torch.cat((desired_return, desired_horizon), dim=-1)
        c = c * self.scaling_factor
        c = self.c_emb(c)
        s = self.s_emb(state.float())  # (batch_size, 64)
        hidden = self.fc(s * c)

        return hidden

    def get_demand(self):
        '''function for generating daily demand from zero-upper-truncated normal distribution'''

        mean = self.D / 365  # average annual mean of demand
        stddev = self.sigma_L / (np.sqrt(self.L))  # standard deviation of demand
        lower_bound = 0
        upper_bound = self.D / 365 * 2  # bound of daily demand

        return gamma.rvs(a=self.alpha, scale=self.theta, size=1)[0]

        # return truncnorm.rvs((lower_bound - mean) / stddev, (upper_bound - mean) / stddev, loc=mean, scale=stddev, size=1)[0]

    def reset(self):
        '''function for setting the initial environment'''

        self.inventory_level_list = np.array([0] * 400)  # inventory level during 1 year
        self.on_order_list = np.array([0] * 400)  # on_order list during lead time
        self.day = 0  # day index
        self.last_order_day = -100

        obs = np.concatenate([
            [self.inventory_level_list[self.day]],  # → shape (1,)
            self.on_order_list[self.day: self.day + self.L]  # → shape (L,)
        ], axis=0)  # 최종 shape: (L+1,)

        return obs

    def step(self, action):
        '''function for proceeding one step'''

        Q, k = action
        cost = 0
        stockout_frequency = 0
        stockout_amount = 0
        daily_demand = self.get_demand()

        self.day += 1  # update the date
        self.inventory_level_list[self.day] = self.inventory_level_list[self.day - 1]
        self.inventory_level_list[self.day] += self.on_order_list[self.day]  # replenishment at the start of the day
        cost += self.h / 365 * self.c * max(0, self.inventory_level_list[self.day])  # compute holding cost right after replenishment
        self.inventory_level_list[self.day] -= daily_demand  # reflect daily demand to inventory level
        cost -= self.p * min(daily_demand, max(0, self.inventory_level_list[self.day]))

        if self.day > self.L + 1 and self.inventory_level_list[
            self.day] < 0:  # compute frequency of stockout and amount of stockout
            stockout_frequency += 1
            stockout_amount += self.inventory_level_list[self.day] * -1

        # and all(abs(x) < 1e-6 for x in self.on_order_list[self.day + 1 : self.day + self.L])
        if self.day + self.L < 365 and self.day - self.last_order_day >= self.L and self.inventory_level_list[self.day] < (self.D / 365) * self.L + k * self.sigma_L:
            self.on_order_list[self.day + self.L - 1] = Q  # order if the current inventory level is lower than safety stock
            cost += self.A  # ordering cost
            self.last_order_day = self.day

        n_obs = np.concatenate([[self.inventory_level_list[self.day]],  # → shape (1,)
                                self.on_order_list[self.day: self.day + self.L]  # → shape (L,)
                                ], axis=0)  # 최종 shape: (L + 1,)
        reward = np.array([cost * -1 / 1000, stockout_frequency * -1 / 10, stockout_amount * -1 / 10])  # reward\

        return n_obs, reward, self.day, self.last_order_day

def dominates(p1, p2):
    return np.all(p1 <= p2) and np.any(p1 < p2)

"""Coverage metric: C(A, B) = 비율 of B dominated by A"""
def set_coverage(A, B):

    count = 0
    for b in B:
        if any(dominates(a, b) for a in A):
            count += 1
    return count / len(B) if len(B) > 0 else 0.0

def max_spread(solutions):
    """
    Maximum spread = 거리 공간에서 가장 멀리 떨어진 두 점 사이의 거리
    """
    if len(solutions) < 2:
        return 0.0
    dists = np.linalg.norm(solutions[:, None, :] - solutions[None, :, :], axis=-1)
    return np.max(dists)

def spacing(solutions):
    """
    Spacing = 해들 간 가장 가까운 거리의 표준편차
    (해들 사이 pairwise 거리의 편차 측정)
    """
    if len(solutions) < 2:
        return 0.0
    # 각 해에 대해 가장 가까운 이웃까지의 거리
    distances = []
    for i, s in enumerate(solutions):
        other = np.delete(solutions, i, axis=0)
        d = np.min(np.linalg.norm(other - s, axis=1))
        distances.append(d)
    distances = np.array(distances)
    return np.std(distances)

def compute_hypervolume(pcn_front, ref_point=[10000, 100, 1000]):
    """
    pcn_front: shape (N, 3), where each row is an objective vector
    ref_point: list or array of size 3, representing the reference point for HV calculation
    """
    # Hypervolume expects minimization; no need to invert if already using -cost etc.
    hv = hypervolume(pcn_front)
    hv_value = hv.compute(ref_point)

    return hv_value

def get_non_dominated_front(points):
    """
    points: shape (N, 3), 여러 방법의 결과를 합친 점들
    반환: points 중 non-dominated(파레토 최적)인 점들만 추출
    """
    points = np.asarray(points)
    keep = np.ones(len(points), dtype=bool)
    for i, p in enumerate(points):
        for j, q in enumerate(points):
            if i != j and dominates(q, p):
                keep[i] = False
                break
    return points[keep]

def normalize_front(front, ideal_point, nadir_point):
    """
    objective를 [0, 1] 범위로 min-max 정규화 (ideal_point=최소값, nadir_point=최대값 기준).
    HV/IGD 계산 전 표준적으로 수행하는 전처리 (pymoo 등 다목적 최적화 라이브러리의 관례).
    """
    front = np.asarray(front)
    denom = np.where(nadir_point - ideal_point == 0, 1.0, nadir_point - ideal_point)
    return (front - ideal_point) / denom

def compute_igd(front, reference_front):
    """
    IGD (Inverted Generational Distance): reference_front의 각 점에서
    front까지의 최소 유클리드 거리의 평균. 값이 작을수록 reference_front를 잘 커버함을 의미.
    reference_front: true Pareto front가 없으므로 비교 대상 방법들의 결과를 합쳐
    non-dominated 처리한 근사 참조 집합을 사용 (get_non_dominated_front 참고).
    """
    front = np.asarray(front)
    reference_front = np.asarray(reference_front)
    if len(front) == 0 or len(reference_front) == 0:
        return np.nan
    dists = np.linalg.norm(reference_front[:, None, :] - front[None, :, :], axis=-1)
    return np.mean(np.min(dists, axis=1))

def plot_representative_action_trajectories(
    mopso_csv_path,
    pcn_original_csv_path,
    pcn_var1_csv_path,
    pcn_var2_csv_path,
    save_dir,
    demand_env,
    run_id,
    mopso_ams_csv_path=None,
    capql_csv_path=None,
):
    """
    6개 알고리즘의 대표 Q, k trajectory를 그리되,
    Q용 broken-axis figure 1개, k용 broken-axis figure 1개를 각각 저장.

    - MOPSO, MOPSO-AMS: 정적 Q, k -> 수평선
    - PCN 계열: Episode==0 trajectory 사용
    - CAPQL: 상태 조건부 정책의 단일 rollout (Episode 컬럼 없음) 사용
    """
    # Matches the finalized palette in plot_5methods_pbest_total_steps_2e6.py
    COLOR_MOPSO = "#000000"
    COLOR_MOPSO_AMS = "#1d4ed8"
    COLOR_CAPQL = "#9333ea"
    COLOR_PCN_ORIGINAL = "#16a34a"
    COLOR_PCN_VAR1 = "#eab308"
    COLOR_PCN_VAR2 = "#dc2626"

    # -----------------------------------
    # 1) 유틸 함수들
    # -----------------------------------
    def find_column(df, candidates):
        """df에서 candidates 리스트 중 존재하는 첫 번째 컬럼명을 반환."""
        for c in candidates:
            if c in df.columns:
                return c
        raise KeyError(f"Columns not found. Tried: {candidates}")

    def load_first_episode(csv_path):
        """PCN 계열 CSV에서 Episode==0 trajectory 로드."""
        df = pd.read_csv(csv_path)
        if "Episode" not in df.columns:
            raise ValueError(f"{csv_path} has no 'Episode' column (MOPSO CSV?).")
        first_ep = df["Episode"].min()
        df_ep = df[df["Episode"] == first_ep].copy()
        return df_ep.sort_values("Step")

    def load_mopso_policy(csv_path):
        """MOPSO CSV: 단일 정적 정책 -> 첫 row의 Q, k 사용."""
        df = pd.read_csv(csv_path)
        # 컬럼 이름에서 쉼표/공백 제거
        df.columns = [c.replace(",", "").strip() for c in df.columns]

        Q = df["Q (Order Size)"].iloc[0]
        k_col = find_column(df, ["k (Safety Stock)", "k"])
        k = df[k_col].iloc[0]
        return Q, k

    def load_capql_trajectory(csv_path):
        """CAPQL CSV: Episode 컬럼 없이 하나의 연속 rollout -> Step 순 정렬."""
        df = pd.read_csv(csv_path)
        return df.sort_values("Step")

    # -----------------------------------
    # 2) 데이터 로드
    # -----------------------------------
    Q_mopso, k_mopso = load_mopso_policy(mopso_csv_path)
    Q_mopso_ams, k_mopso_ams = load_mopso_policy(mopso_ams_csv_path) if mopso_ams_csv_path else (None, None)
    capql_df = load_capql_trajectory(capql_csv_path) if capql_csv_path else None

    pcn_orig_df = load_first_episode(pcn_original_csv_path)
    pcn_var1_df = load_first_episode(pcn_var1_csv_path)
    pcn_var2_df = load_first_episode(pcn_var2_csv_path)

    # k 컬럼 이름 자동 탐색
    k_candidates = [
        "k (Safety Stock)", "k (Safety Stock),",
        "r (Reorder Point)", "r (Reorder Point),",
        "k", "r"
    ]
    k_col_orig = find_column(pcn_orig_df, k_candidates)
    k_col_var1 = find_column(pcn_var1_df, k_candidates)
    k_col_var2 = find_column(pcn_var2_df, k_candidates)

    # step 범위
    max_step = max(
        pcn_orig_df["Step"].max(),
        pcn_var1_df["Step"].max(),
        pcn_var2_df["Step"].max(),
        capql_df["Step"].max() if capql_df is not None else 0,
    )
    days = np.arange(max_step + 1)

    os.makedirs(save_dir, exist_ok=True)

    # -----------------------------------
    # 3) Broken-axis: Q trajectories
    # -----------------------------------
    # GBM 그림 기준으로 대략 맞춰둔 범위 (필요하면 숫자 조정)
    low_ylim_Q = (200, 1350)      # 아래쪽 확대 구간 (CAPQL의 넓은 변동폭 포함)
    high_ylim_Q = (3400, 3700)  # 위쪽 PCN_original 구간

    fig_Q, (ax_Q_high, ax_Q_low) = plt.subplots(
        2, 1,
        sharex=True,
        figsize=(10, 7),
        gridspec_kw={"height_ratios": [1, 3]},
    )

    # 아래 축: 주요 구간
    lw = 1.5  # 모든 선 굵기 통일

    ax_Q_low.plot(
        days,
        np.full_like(days, Q_mopso, dtype=float),
        linestyle='-',
        color=COLOR_MOPSO,
        linewidth=lw,
        label='MOPSO',
    )
    if Q_mopso_ams is not None:
        ax_Q_low.plot(
            days,
            np.full_like(days, Q_mopso_ams, dtype=float),
            linestyle='-',
            color=COLOR_MOPSO_AMS,
            linewidth=lw,
            label='MOPSO-AMS',
        )
    if capql_df is not None:
        ax_Q_low.plot(
            capql_df["Step"],
            capql_df["Q (Order Size)"],
            linestyle='-',
            marker='d',
            markerfacecolor='none',
            markeredgecolor=COLOR_CAPQL,
            color=COLOR_CAPQL,
            linewidth=lw,
            label='CAPQL',
        )
    ax_Q_low.plot(
        pcn_orig_df["Step"],
        pcn_orig_df["Q (Order Size)"],
        linestyle='-',
        marker='s',
        markerfacecolor='none',
        markeredgecolor=COLOR_PCN_ORIGINAL,
        color=COLOR_PCN_ORIGINAL,
        linewidth=lw,
        label='Original PCN',
    )
    ax_Q_low.plot(
        pcn_var1_df["Step"],
        pcn_var1_df["Q (Order Size)"],
        linestyle='-',
        marker='o',
        markerfacecolor='none',
        markeredgecolor=COLOR_PCN_VAR1,
        color=COLOR_PCN_VAR1,
        linewidth=lw,
        label='PCN + buffer init',
    )
    ax_Q_low.plot(
        pcn_var2_df["Step"],
        pcn_var2_df["Q (Order Size)"],
        linestyle='-',
        marker='^',
        markerfacecolor='none',
        markeredgecolor=COLOR_PCN_VAR2,
        color=COLOR_PCN_VAR2,
        linewidth=lw,
        label='Guided PCN',
    )

    ax_Q_low.set_ylim(low_ylim_Q)
    ax_Q_low.set_xlabel("Day")
    ax_Q_low.set_ylabel("Order quantity (Q)")
    ax_Q_low.grid(True)
    ax_Q_low.legend(loc="upper right")

    # 위 축: outlier (Original PCN)만
    ax_Q_high.plot(
        pcn_orig_df["Step"],
        pcn_orig_df["Q (Order Size)"],
        linestyle='-',
        marker='s',
        markerfacecolor='none',
        markeredgecolor=COLOR_PCN_ORIGINAL,
        color=COLOR_PCN_ORIGINAL,
        linewidth=lw,
    )
    ax_Q_high.set_ylim(high_ylim_Q)
    ax_Q_high.grid(True)
    ax_Q_high.tick_params(labelbottom=False)  # x 라벨은 아래 축만

    # 축 break 표시
    d = 0.015
    kwargs = dict(transform=ax_Q_high.transAxes, color='k', clip_on=False)
    ax_Q_high.plot((-d, +d), (-d, +d), **kwargs)
    ax_Q_high.plot((1 - d, 1 + d), (-d, +d), **kwargs)

    kwargs.update(transform=ax_Q_low.transAxes)
    ax_Q_low.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_Q_low.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    fig_Q.tight_layout()
    out_path_Q = os.path.join(save_dir, f"action_trajectories_Q_{demand_env}_{run_id}.pdf")
    fig_Q.savefig(out_path_Q, dpi=300)
    plt.close(fig_Q)

    # -----------------------------------
    # 4) Broken-axis: k trajectories
    # -----------------------------------
    low_ylim_k = (0, 500)       # 아래쪽 확대 구간
    high_ylim_k = (3400, 4400)  # 위쪽 PCN_original 구간

    fig_k, (ax_k_high, ax_k_low) = plt.subplots(
        2, 1,
        sharex=True,
        figsize=(10, 7),
        gridspec_kw={"height_ratios": [1, 3]},
    )

    # 아래 축
    ax_k_low.plot(
        days,
        np.full_like(days, k_mopso, dtype=float),
        linestyle='-',
        color=COLOR_MOPSO,
        linewidth=lw,
        label='MOPSO',
    )
    if k_mopso_ams is not None:
        ax_k_low.plot(
            days,
            np.full_like(days, k_mopso_ams, dtype=float),
            linestyle='-',
            color=COLOR_MOPSO_AMS,
            linewidth=lw,
            label='MOPSO-AMS',
        )
    if capql_df is not None:
        ax_k_low.plot(
            capql_df["Step"],
            capql_df["k (Safety Stock)"],
            linestyle='-',
            marker='d',
            markerfacecolor='none',
            markeredgecolor=COLOR_CAPQL,
            color=COLOR_CAPQL,
            linewidth=lw,
            label='CAPQL',
        )
    ax_k_low.plot(
        pcn_orig_df["Step"],
        pcn_orig_df[k_col_orig],
        linestyle='-',
        marker='s',
        markerfacecolor='none',
        markeredgecolor=COLOR_PCN_ORIGINAL,
        color=COLOR_PCN_ORIGINAL,
        linewidth=lw,
        label='Original PCN',
    )
    ax_k_low.plot(
        pcn_var1_df["Step"],
        pcn_var1_df[k_col_var1],
        linestyle='-',
        marker='o',
        markerfacecolor='none',
        markeredgecolor=COLOR_PCN_VAR1,
        color=COLOR_PCN_VAR1,
        linewidth=lw,
        label='PCN + buffer init',
    )
    ax_k_low.plot(
        pcn_var2_df["Step"],
        pcn_var2_df[k_col_var2],
        linestyle='-',
        marker='^',
        markerfacecolor='none',
        markeredgecolor=COLOR_PCN_VAR2,
        color=COLOR_PCN_VAR2,
        linewidth=lw,
        label='Guided PCN',
    )

    ax_k_low.set_ylim(low_ylim_k)
    ax_k_low.set_xlabel("Day")
    ax_k_low.set_ylabel("Reorder point (r)")
    ax_k_low.grid(True)
    ax_k_low.legend(loc="upper right")

    # 위 축: outlier (Original PCN)
    ax_k_high.plot(
        pcn_orig_df["Step"],
        pcn_orig_df[k_col_orig],
        linestyle='-',
        marker='s',
        markerfacecolor='none',
        markeredgecolor=COLOR_PCN_ORIGINAL,
        color=COLOR_PCN_ORIGINAL,
        linewidth=lw,
    )
    ax_k_high.set_ylim(high_ylim_k)
    ax_k_high.grid(True)
    ax_k_high.tick_params(labelbottom=False)

    # 축 break 표시
    d = 0.015
    kwargs = dict(transform=ax_k_high.transAxes, color='k', clip_on=False)
    ax_k_high.plot((-d, +d), (-d, +d), **kwargs)
    ax_k_high.plot((1 - d, 1 + d), (-d, +d), **kwargs)

    kwargs.update(transform=ax_k_low.transAxes)
    ax_k_low.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_k_low.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    fig_k.tight_layout()
    out_path_k = os.path.join(save_dir, f"action_trajectories_k_{demand_env}_{run_id}.pdf")
    fig_k.savefig(out_path_k, dpi=300)
    plt.close(fig_k)

    #print("[Saved]", out_path_Q)
    #print("[Saved]", out_path_k)

if __name__ == "__main__":
    import torch
    from datetime import datetime
    import uuid
    from pcn_continuous import train # PCN original
    from MOPSO import*

    n_runs = 10
    start = time.time()

    '''pcn_results = []
    mopso_results = []
    set_coverage_A_B = []  # PCN ⊃ MOPSO
    set_coverage_B_A = []  # MOPSO ⊃ PCN
    spacings_pcn = []
    spacings_mopso = []
    spreads_pcn = []
    spreads_mopso = []
    hv_pcn = []
    hv_mopso = []

    for run_id in range(1, n_runs + 1):
        print(f"\n[Run {run_id}/{n_runs}]")

        # 고정 시드 설정: run_id 기준
        random.seed(run_id)
        np.random.seed(run_id)
        torch.manual_seed(run_id)

        ## --------------------- PCN 실행 ---------------------
        device = 'cpu'
        scaling_factor = torch.tensor([[0.1, 0.1, 0.1, 0.1]]).to(device)
        env = Inventory(scaling_factor)

        nA = np.prod(env.action_space.shape)
        model = Inventory(scaling_factor).to(device)
        model = ContinuousHead(model)
        env.nA = nA

        logdir = '/'
        logdir += datetime.now().strftime('%Y-%m-%d_%H-%M-%S_') + str(uuid.uuid4())[:4] + '/'
        print(f"[PCN logdir: {logdir}]")

        non_dom_rewards = train(
            env,
            model,
            run_id,
            learning_rate=1e-4,
            batch_size=256,
            total_steps=1e6,
            n_model_updates=100,
            n_er_episodes=50,
            n_step_episodes=10,
            max_size=300,
            max_return=np.array([0, 0, 0]),
            ref_point=np.array([-100000, -365., -1000000.]),
            noise=0.0,
            logdir=logdir,
            clip_grad_norm= None
        )

        # CSV 저장
        df = pd.DataFrame({
            "TC": [-1 * p[0] for p in non_dom_rewards],
            "Frequency": [-1 * p[1] for p in non_dom_rewards],
            "Amount": [-1 * p[2] for p in non_dom_rewards]
        })
        df.to_csv(f"./PCN/original_PCN_run{run_id}.csv", index=False)

        pcn_front = np.array([[-1 * r[0], -1 * r[1], -1 * r[2]] for r in non_dom_rewards])
        pcn_results.append(pcn_front)
        spacings_pcn.append(spacing(pcn_front))
        spreads_pcn.append(max_spread(pcn_front))
        #hv_pcn.append(compute_hypervolume(pcn_front))

        ## --------------------- MOPSO 실행 ---------------------
        # 동일 시드 재사용

        print(f"--- MOPSO Run {run_id} ---")


        bounds = np.array([[0, env.D], [0, env.D]])  # search space for Q and k
        mopso = MOPSO(num_particles=200, dim=2, bounds=bounds, num_iterations=100)
        solution = mopso.optimize()

        data = {
            "Q (Order Size)": [p[0][0] for p in solution],
            "k (Safety Stock)": [p[0][1] for p in solution],
            "TC": [p[1][0] for p in solution],
            "Frequency": [p[1][1] for p in solution],
            "Amount": [p[1][2] for p in solution]
        }
        df = pd.DataFrame(data)
        df.to_csv(f"./MOPSO/MOPSO_run{run_id}.csv", index=False)
        print(df)
        mopso_front = np.array([[p[1][0], p[1][1], p[1][2]] for p in solution])
        mopso_results.append(mopso_front)

        spacings_mopso.append(spacing(mopso_front))
        spreads_mopso.append(max_spread(mopso_front))
        #hv_mopso.append(compute_hypervolume(mopso_front))
        
        sc_A_B = set_coverage(pcn_front, mopso_front)
        sc_B_A = set_coverage(mopso_front, pcn_front)
        set_coverage_A_B.append(sc_A_B)
        set_coverage_B_A.append(sc_B_A)

    print("\n================= Summary =================")
    print(f"Set Coverage (PCN ⊃ MOPSO): {np.mean(set_coverage_A_B):.4f} ± {np.std(set_coverage_A_B):.4f}")
    print(f"Set Coverage (MOPSO ⊃ PCN): {np.mean(set_coverage_B_A):.4f} ± {np.std(set_coverage_B_A):.4f}")
    print(f"Spacing (PCN): {np.mean(spacings_pcn):.4f} ± {np.std(spacings_pcn):.4f}")
    print(f"Spacing (MOPSO): {np.mean(spacings_mopso):.4f} ± {np.std(spacings_mopso):.4f}")
    print(f"Max Spread (PCN): {np.mean(spreads_pcn):.4f} ± {np.std(spreads_pcn):.4f}")
    print(f"Max Spread (MOPSO): {np.mean(spreads_mopso):.4f} ± {np.std(spreads_mopso):.4f}")
    #print(f"HV (PCN): {np.mean(hv_pcn):.4f} ± {np.std(hv_pcn):.4f}")
    #print(f"HV (MOPSO): {np.mean(hv_mopso):.4f} ± {np.std(hv_mopso):.4f}")

    end = time.time()
    print("elapsed time", end - start)'''

    #csv 파일 이용
    demand_env = 'stationary'
    selected_runs = [1,2,3,4,5,6,7,8,9,10]
    #selected_runs = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
    #selected_runs = [1]

    # Initialize lists to store metrics from each run
    spacings_original = []
    spacings_variation1 = []
    spacings_variation2 = []
    spacings_mopso = []

    spreads_original = []
    spreads_variation1 = []
    spreads_variation2 = []
    spreads_mopso = []

    SC_original_mopso = []
    SC_mopso_original = []
    SC_original_variation1 = []
    SC_variation1_original = []
    SC_original_variation2 = []
    SC_variation2_original = []
    SC_variation1_variation2 = []
    SC_variation2_variation1 = []
    SC_variation1_mopso = []
    SC_mopso_variation1 = []
    SC_variation2_mopso = []
    SC_mopso_variation2 = []

    hv_original = []
    hv_variation1 = []
    hv_variation2 = []
    hv_mopso = []

    igd_original = []
    igd_variation1 = []
    igd_variation2 = []
    igd_mopso = []

    # Initialize lists to store the AVERAGE of objectives from each run
    mopso_means = []
    pcn_original_means = []
    pcn_variation1_means = []
    pcn_variation2_means = []

    for run_id in selected_runs:
        if demand_env == 'gs':
            mopso_df = pd.read_csv(f"./MOPSO/MOPSO_{demand_env}/MOPSO_run_{demand_env}_test1_{run_id}.csv")
        else:  # stationary, seasonal, GBM
            mopso_df = pd.read_csv(f"./MOPSO/MOPSO_{demand_env}/MOPSO_run_{demand_env}_{run_id}.csv")
        mopso_front = mopso_df[["TC", "Frequency", "Amount"]].to_numpy()
        mopso_means.append(np.mean(mopso_front, axis=0))

        if demand_env == 'gs':
            pcn_original_df = pd.read_csv(
                f"./PCN/PCN_{demand_env}_original/original_PCN_run_test1_original_{run_id}.csv")
        else:  # stationary, seasonal, GBM
            pcn_original_df = pd.read_csv(f"./PCN/PCN_{demand_env}_original/PCN_{demand_env}_original_run_{run_id}.csv")
        pcn_original_front = pcn_original_df[["TC", "Frequency", "Amount"]].to_numpy()
        pcn_original_means.append(np.mean(pcn_original_front, axis=0))

        # ----- Load PCN variation1 front -----
        if demand_env == 'gs':
            pcn_variation1_df = pd.read_csv(
                f"./PCN/PCN_{demand_env}_variation1/original_PCN_run_test1_variation1_{run_id}.csv")
        else:  # stationary, seasonal, GBM
            pcn_variation1_df = pd.read_csv(
                f"./PCN/PCN_{demand_env}_variation1/PCN_{demand_env}_variation1_run_{run_id}.csv")
        pcn_variation1_front = pcn_variation1_df[["TC", "Frequency", "Amount"]].to_numpy()
        pcn_variation1_means.append(np.mean(pcn_variation1_front, axis=0))

        # ----- Load PCN variation2 front -----
        if demand_env == 'gs':
            pcn_variation2_df = pd.read_csv(
                f"./PCN/PCN_{demand_env}_variation2/original_PCN_run_test1_variation2_{run_id}.csv")
        else:  # stationary, seasonal, GBM
            pcn_variation2_df = pd.read_csv(
                f"./PCN/PCN_{demand_env}_variation2/PCN_{demand_env}_variation2_run_{run_id}.csv")
        pcn_variation2_front = pcn_variation2_df[["TC", "Frequency", "Amount"]].to_numpy()
        pcn_variation2_means.append(np.mean(pcn_variation2_front, axis=0))

        # ----- HV / IGD: 표준 관례대로 objective를 [0,1] 정규화 후 계산 -----
        # (ideal/nadir point는 4개 방법 결과를 합친 combined_front 기준으로 산출)
        combined_front = np.vstack([mopso_front, pcn_original_front, pcn_variation1_front, pcn_variation2_front])
        ideal_point = combined_front.min(axis=0)
        nadir_point = combined_front.max(axis=0)

        mopso_front_norm = normalize_front(mopso_front, ideal_point, nadir_point)
        pcn_original_front_norm = normalize_front(pcn_original_front, ideal_point, nadir_point)
        pcn_variation1_front_norm = normalize_front(pcn_variation1_front, ideal_point, nadir_point)
        pcn_variation2_front_norm = normalize_front(pcn_variation2_front, ideal_point, nadir_point)

        hv_ref_point = np.array([1.1, 1.1, 1.1])  # 정규화 공간에서 nadir(=1.0) + 10% 마진 (pymoo 등의 표준 관례)

        igd_reference_front = get_non_dominated_front(combined_front)  # true PF 대용 근사 참조 집합 (원 스케일에서 추출)
        igd_reference_front_norm = normalize_front(igd_reference_front, ideal_point, nadir_point)

        # ----- Plotting and Metrics (same as before) -----
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(
            mopso_df["Frequency"],
            mopso_df["Amount"],
            mopso_df["TC"],
            marker='x',  # 십자가
            color='black',  # 검정
            s=60,
            label='MOPSO'
        )
        # ... (생략: 나머지 plotting 코드) ...
        ax.scatter(
            pcn_original_df["Frequency"],
            pcn_original_df["Amount"],
            pcn_original_df["TC"],
            marker='s',  # 사각형
            facecolors='none',
            edgecolor='blue',  # 빨강
            s=60,
            label='PCN_original'
        )
        ax.scatter(
            pcn_variation1_df["Frequency"],
            pcn_variation1_df["Amount"],
            pcn_variation1_df["TC"],
            marker='o',  # 채워진 원
            facecolors='none',
            edgecolor='red',
            s=60,
            label='PCN_variation1'
        )
        ax.scatter(
            pcn_variation2_df["Frequency"],
            pcn_variation2_df["Amount"],
            pcn_variation2_df["TC"],
            marker='^',  # 세모 (위로 향함)
            facecolors='none',
            edgecolor='green',  # 초록
            s=60,
            label='PCN_variation2'
        )
        ax.set_xlabel("Stockout Frequency")
        ax.set_ylabel("Stockout Amount")
        ax.set_zlabel("Total Cost(1e6)")
        ax.set_title(f"{demand_env} Pareto Front Comparison (Run {run_id})")
        ax.legend()
        os.makedirs(f"results/pareto_original_4methods/{demand_env}", exist_ok=True)
        plt.savefig(f"results/pareto_original_4methods/{demand_env}/run{run_id}.png")
        plt.close()

        '''''# ----- Plot: Cost vs Amount -----
        plt.figure(figsize=(8, 6))

        # MOPSO
        plt.scatter(
            mopso_df["Amount"],
            mopso_df["TC"],
            marker='x',
            color='black',
            s=60,
            label='MOPSO'
        )

        # PCN original
        plt.scatter(
            pcn_original_df["Amount"],
            pcn_original_df["TC"],
            marker='s',
            facecolors='none',
            edgecolors='blue',
            s=60,
            label='PCN_original'
        )

        # PCN variation1
        plt.scatter(
            pcn_variation1_df["Amount"],
            pcn_variation1_df["TC"],
            marker='o',
            facecolors='none',
            edgecolor='red',
            s=60,
            label='PCN_variation1'
        )

        # PCN variation2
        plt.scatter(
            pcn_variation2_df["Amount"],
            pcn_variation2_df["TC"],
            marker='^',
            facecolors='none',
            edgecolors='green',
            s=60,
            label='PCN_variation2'
        )

        plt.xlabel("Stockout Amount")
        plt.ylabel("Total Cost (1e6)")
        plt.title(f"{demand_env} Cost vs Amount (Run {run_id})")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        os.makedirs(f"results/pareto_2d_cost_amount/{demand_env}", exist_ok=True)
        plt.savefig(f"results/pareto_2d_cost_amount/{demand_env}/run{run_id}.png")
        plt.close()'''

        # ===============================
        #   Broken-axis: Cost vs Frequency
        # ===============================

        low_ylim = (0, 5)  # 아래 확대 구간 (0 포함)
        high_ylim = (50, 55)  # 위쪽 outlier 구간

        fig, (ax_high, ax_low) = plt.subplots(
            2, 1,
            sharex=True,
            figsize=(8, 6),
            gridspec_kw={'height_ratios': [1, 3]}
        )

        # ----- 아래 축 (확대 영역) -----
        ax_low.scatter(mopso_df["Frequency"], mopso_df["TC"],
                       marker='x', color='black', s=60, label='MOPSO')

        ax_low.scatter(pcn_original_df["Frequency"], pcn_original_df["TC"],
                       marker='s', facecolors='none', edgecolors='blue', s=60,
                       label='PCN_original')

        ax_low.scatter(pcn_variation1_df["Frequency"], pcn_variation1_df["TC"],
                       marker='o', facecolors='none', edgecolors='red', s=60,
                       label='PCN_variation1')

        ax_low.scatter(pcn_variation2_df["Frequency"], pcn_variation2_df["TC"],
                       marker='^', facecolors='none', edgecolors='green', s=60,
                       label='PCN_variation2')

        ax_low.set_ylim(low_ylim)
        ax_low.set_xlabel("Stockout Frequency")
        ax_low.set_ylabel("Total Cost (1e6)")
        ax_low.grid(True)

        # ----- 위 축 (outlier용 작은 영역) -----
        ax_high.scatter(pcn_original_df["Frequency"], pcn_original_df["TC"],
                        marker='s', facecolors='none', edgecolors='blue', s=60)

        ax_high.set_ylim(high_ylim)
        ax_high.grid(True)
        ax_high.tick_params(labelbottom=False)  # x축 라벨은 아래 축만

        # ----- 범례: 아래 축에서 핸들/라벨을 가져와 위 축에 표시 -----
        handles, labels = ax_low.get_legend_handles_labels()
        ax_high.legend(handles, labels, loc="upper right")

        # ----- 축이 끊겼다는 표시(가로 '//' 느낌) -----
        d = 0.015  # 표시 크기
        kwargs = dict(color='k', clip_on=False)

        # 위 축 아래쪽
        ax_high.plot((-d, +d), (-d, +d), transform=ax_high.transAxes, **kwargs)
        ax_high.plot((1 - d, 1 + d), (-d, +d), transform=ax_high.transAxes, **kwargs)

        # 아래 축 위쪽
        ax_low.plot((-d, +d), (1 - d, 1 + d), transform=ax_low.transAxes, **kwargs)
        ax_low.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax_low.transAxes, **kwargs)

        fig.suptitle(f"{demand_env} Cost vs Frequency (Run {run_id})")

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.05)
        os.makedirs(f"results/pareto_2d_cost_freq_broken/{demand_env}", exist_ok=True)
        plt.savefig(f"results/pareto_2d_cost_freq_broken/{demand_env}/run{run_id}.png")
        plt.close()

        # ===============================
        #   Broken-axis: Cost vs Amount
        # ===============================

        fig, (ax_high, ax_low) = plt.subplots(
            2, 1,
            sharex=True,
            figsize=(8, 6),
            gridspec_kw={'height_ratios': [1, 3]}
        )

        # ----- 아래 축 (확대 영역) -----
        ax_low.scatter(mopso_df["Amount"], mopso_df["TC"],
                       marker='x', color='black', s=60, label='MOPSO')

        ax_low.scatter(pcn_original_df["Amount"], pcn_original_df["TC"],
                       marker='s', facecolors='none', edgecolors='blue', s=60,
                       label='PCN_original')

        ax_low.scatter(pcn_variation1_df["Amount"], pcn_variation1_df["TC"],
                       marker='o', facecolors='none', edgecolors='red', s=60,
                       label='PCN_variation1')

        ax_low.scatter(pcn_variation2_df["Amount"], pcn_variation2_df["TC"],
                       marker='^', facecolors='none', edgecolors='green', s=60,
                       label='PCN_variation2')

        ax_low.set_ylim(low_ylim)
        ax_low.set_xlabel("Stockout Amount")
        ax_low.set_ylabel("Total Cost (1e6)")
        ax_low.grid(True)

        # ----- 위 축 (outlier용 작은 영역) -----
        ax_high.scatter(pcn_original_df["Amount"], pcn_original_df["TC"],
                        marker='s', facecolors='none', edgecolors='blue', s=60)

        ax_high.set_ylim(high_ylim)
        ax_high.grid(True)
        ax_high.tick_params(labelbottom=False)

        # ----- 범례: 아래 축에서 가져와 위 축에 표시 -----
        handles, labels = ax_low.get_legend_handles_labels()
        ax_high.legend(handles, labels, loc="upper right")

        # ----- 축이 끊겼다는 표시 -----
        d = 0.015
        kwargs = dict(color='k', clip_on=False)

        # 위 축 아래쪽
        ax_high.plot((-d, +d), (-d, +d), transform=ax_high.transAxes, **kwargs)
        ax_high.plot((1 - d, 1 + d), (-d, +d), transform=ax_high.transAxes, **kwargs)

        # 아래 축 위쪽
        ax_low.plot((-d, +d), (1 - d, 1 + d), transform=ax_low.transAxes, **kwargs)
        ax_low.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax_low.transAxes, **kwargs)

        fig.suptitle(f"{demand_env} Cost vs Amount (Run {run_id})")

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.05)
        os.makedirs(f"results/pareto_2d_cost_amount_broken/{demand_env}", exist_ok=True)
        plt.savefig(f"results/pareto_2d_cost_amount_broken/{demand_env}/run{run_id}.png")
        plt.close()

        plot_representative_action_trajectories(
            mopso_csv_path="./MOPSO/MOPSO_GBM/MOPSO_run_GBM_1.csv",
            pcn_original_csv_path="./PCN/original_non_dominated_action_trajectories_1.csv",
            pcn_var1_csv_path="./PCN/variation1_non_dominated_action_trajectories_1.csv",
            pcn_var2_csv_path="./PCN/variation2_non_dominated_action_trajectories_1.csv",
            save_dir="results",
            demand_env='GBM',  # e.g., "stationary"
            run_id=run_id
        )

        # ----- Metrics (per run) -----
        spacings_original.append(spacing(pcn_original_front))
        spacings_variation1.append(spacing(pcn_variation1_front))
        spacings_variation2.append(spacing(pcn_variation2_front))
        spacings_mopso.append(spacing(mopso_front))

        spreads_original.append(max_spread(pcn_original_front))
        spreads_variation1.append(max_spread(pcn_variation1_front))
        spreads_variation2.append(max_spread(pcn_variation2_front))
        spreads_mopso.append(max_spread(mopso_front))

        SC_original_mopso.append(set_coverage(pcn_original_front, mopso_front))
        SC_mopso_original.append(set_coverage(mopso_front, pcn_original_front))
        SC_original_variation1.append(set_coverage(pcn_original_front, pcn_variation1_front))
        SC_variation1_original.append(set_coverage(pcn_variation1_front, pcn_original_front))
        SC_original_variation2.append(set_coverage(pcn_original_front, pcn_variation2_front))
        SC_variation2_original.append(set_coverage(pcn_variation2_front, pcn_original_front))
        SC_variation1_variation2.append(set_coverage(pcn_variation1_front, pcn_variation2_front))
        SC_variation2_variation1.append(set_coverage(pcn_variation2_front, pcn_variation1_front))
        SC_variation1_mopso.append(set_coverage(pcn_variation1_front, mopso_front))
        SC_mopso_variation1.append(set_coverage(mopso_front, pcn_variation1_front))
        SC_variation2_mopso.append(set_coverage(pcn_variation2_front, mopso_front))
        SC_mopso_variation2.append(set_coverage(mopso_front, pcn_variation2_front))

        hv_mopso.append(compute_hypervolume(mopso_front_norm, ref_point=hv_ref_point))
        hv_original.append(compute_hypervolume(pcn_original_front_norm, ref_point=hv_ref_point))
        hv_variation1.append(compute_hypervolume(pcn_variation1_front_norm, ref_point=hv_ref_point))
        hv_variation2.append(compute_hypervolume(pcn_variation2_front_norm, ref_point=hv_ref_point))

        igd_mopso.append(compute_igd(mopso_front_norm, igd_reference_front_norm))
        igd_original.append(compute_igd(pcn_original_front_norm, igd_reference_front_norm))
        igd_variation1.append(compute_igd(pcn_variation1_front_norm, igd_reference_front_norm))
        igd_variation2.append(compute_igd(pcn_variation2_front_norm, igd_reference_front_norm))

    # Summary (calculating averages from the lists populated in the loop)
    print("\n========== Selected Runs Summary ==========")
    print(f"Spacing (MOPSO): {np.mean(spacings_mopso):.4f} ± {np.std(spacings_mopso):.4f}")
    print(f"Spacing (PCN original): {np.mean(spacings_original):.4f} ± {np.std(spacings_original):.4f}")
    print(f"Spacing (PCN variation1): {np.mean(spacings_variation1):.4f} ± {np.std(spacings_variation1):.4f}")
    print(f"Spacing (PCN variation2): {np.mean(spacings_variation2):.4f} ± {np.std(spacings_variation2):.4f}")
    print(f"Max Spread (MOPSO): {np.mean(spreads_mopso):.4f} ± {np.std(spreads_mopso):.4f}")
    print(f"Max Spread (PCN original): {np.mean(spreads_original):.4f} ± {np.std(spreads_original):.4f}")
    print(f"Max Spread (PCN variation1): {np.mean(spreads_variation1):.4f} ± {np.std(spreads_variation1):.4f}")
    print(f"Max Spread (PCN variation2): {np.mean(spreads_variation2):.4f} ± {np.std(spreads_variation2):.4f}")
    print(f"HV (MOPSO): {np.mean(hv_mopso):.4f} ± {np.std(hv_mopso):.4f}")
    print(f"HV (PCN original): {np.mean(hv_original):.4f} ± {np.std(hv_original):.4f}")
    print(f"HV (PCN variation1): {np.mean(hv_variation1):.4f} ± {np.std(hv_variation1):.4f}")
    print(f"HV (PCN variation2): {np.mean(hv_variation2):.4f} ± {np.std(hv_variation2):.4f}")
    print(f"IGD (MOPSO): {np.mean(igd_mopso):.4f} ± {np.std(igd_mopso):.4f}")
    print(f"IGD (PCN original): {np.mean(igd_original):.4f} ± {np.std(igd_original):.4f}")
    print(f"IGD (PCN variation1): {np.mean(igd_variation1):.4f} ± {np.std(igd_variation1):.4f}")
    print(f"IGD (PCN variation2): {np.mean(igd_variation2):.4f} ± {np.std(igd_variation2):.4f}")

    '''# 순서 정의
    labels = ['MOPSO', 'Original', 'Variation1', 'Variation2']
    coverage_matrix = np.zeros((4, 4))

    # 평균값 삽입 (행: 지배자 A, 열: 피지배자 B)
    coverage_matrix[0, 1] = np.mean(SC_mopso_original)
    coverage_matrix[0, 2] = np.mean(SC_mopso_variation1)
    coverage_matrix[0, 3] = np.mean(SC_mopso_variation2)

    coverage_matrix[1, 0] = np.mean(SC_original_mopso)
    coverage_matrix[1, 2] = np.mean(SC_original_variation1)
    coverage_matrix[1, 3] = np.mean(SC_original_variation2)

    coverage_matrix[2, 0] = np.mean(SC_variation1_mopso)
    coverage_matrix[2, 1] = np.mean(SC_variation1_original)
    coverage_matrix[2, 3] = np.mean(SC_variation1_variation2)

    coverage_matrix[3, 0] = np.mean(SC_variation2_mopso)
    coverage_matrix[3, 1] = np.mean(SC_variation2_original)
    coverage_matrix[3, 2] = np.mean(SC_variation2_variation1)'''

    # 순서 정의
    labels = ['MOPSO', 'Original', 'Variation1', 'Variation2']

    # 표기 위해 object dtype 사용
    coverage_matrix = np.empty((4, 4), dtype=object)


    # Helper 함수
    def format_mean_std(arr):
        return f"{np.mean(arr):.2f} ± {np.std(arr):.2f}"


    # 평균 ± 표준편차 삽입
    coverage_matrix[0, 1] = format_mean_std(SC_mopso_original)
    coverage_matrix[0, 2] = format_mean_std(SC_mopso_variation1)
    coverage_matrix[0, 3] = format_mean_std(SC_mopso_variation2)

    coverage_matrix[1, 0] = format_mean_std(SC_original_mopso)
    coverage_matrix[1, 2] = format_mean_std(SC_original_variation1)
    coverage_matrix[1, 3] = format_mean_std(SC_original_variation2)

    coverage_matrix[2, 0] = format_mean_std(SC_variation1_mopso)
    coverage_matrix[2, 1] = format_mean_std(SC_variation1_original)
    coverage_matrix[2, 3] = format_mean_std(SC_variation1_variation2)

    coverage_matrix[3, 0] = format_mean_std(SC_variation2_mopso)
    coverage_matrix[3, 1] = format_mean_std(SC_variation2_original)
    coverage_matrix[3, 2] = format_mean_std(SC_variation2_variation1)

    # 대각선은 "-"로 표기
    for i in range(4):
        coverage_matrix[i, i] = "-"

    # DataFrame으로 변환 및 출력
    coverage_df = pd.DataFrame(coverage_matrix, index=labels, columns=labels)
    print("\n📊 Set Coverage Matrix (A dominates B):")
    print(coverage_df.round(2))

    # ----- Final Overall Averages (Average of Averages) -----
    # Convert lists of means to numpy arrays
    mean_mopso_all_runs = np.array(mopso_means)
    mean_original_all_runs = np.array(pcn_original_means)
    mean_variation1_all_runs = np.array(pcn_variation1_means)
    mean_variation2_all_runs = np.array(pcn_variation2_means)

    # Calculate the mean and std of the means
    mean_mopso = np.mean(mean_mopso_all_runs, axis=0)
    std_mopso = np.std(mean_mopso_all_runs, axis=0)
    mean_original = np.mean(mean_original_all_runs, axis=0)
    std_original = np.std(mean_original_all_runs, axis=0)
    mean_variation1 = np.mean(mean_variation1_all_runs, axis=0)
    std_variation1 = np.std(mean_variation1_all_runs, axis=0)
    mean_variation2 = np.mean(mean_variation2_all_runs, axis=0)
    std_variation2 = np.std(mean_variation2_all_runs, axis=0)

    print(f"\nAverage Objectives (MOPSO): "
          f"Total Cost={mean_mopso[0]:.4f} ± {std_mopso[0]:.4f}, "
          f"Stockout Frequency={mean_mopso[1]:.4f} ± {std_mopso[1]:.4f}, "
          f"Stockout Amount={mean_mopso[2]:.4f} ± {std_mopso[2]:.4f}")

    print(f"Average Objectives (PCN original): "
          f"Total Cost={mean_original[0]:.4f} ± {std_original[0]:.4f}, "
          f"Stockout Frequency={mean_original[1]:.4f} ± {std_original[1]:.4f}, "
          f"Stockout Amount={mean_original[2]:.4f} ± {std_original[2]:.4f}")

    print(f"Average Objectives (PCN variation1): "
          f"Total Cost={mean_variation1[0]:.4f} ± {std_variation1[0]:.4f},"
          f"Stockout Frequency={mean_variation1[1]:.4f} ± {std_variation1[1]:.4f}, "
          f"Stockout Amount={mean_variation1[2]:.4f} ± {std_variation1[2]:.4f}")

    print(f"Average Objectives (PCN variation2): "
          f"Total Cost={mean_variation2[0]:.4f} ± {std_variation2[0]:.4f}, "
          f"Stockout Frequency={mean_variation2[1]:.4f} ± {std_variation2[1]:.4f}, "
          f"Stockout Amount={mean_variation2[2]:.4f} ± {std_variation2[2]:.4f}")




