import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _sub in ("baselines", "envs", "models", "utils", "experiments"):
    _p = _os.path.join(_ROOT, _sub)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
_os.chdir(_ROOT)

import heapq
import time

import numpy as np
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from logger import Logger
import os
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
from MOPSO import*
from typing import Optional
from scipy.stats import qmc
import scipy.special
from datetime import datetime
import pickle
from collections import deque
import itertools
from main_original import Inventory

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# diagnostic-only: magnitude of the per-objective improvement/perturbation
# actually sampled each iteration, for comparing choose_commands_original's
# U(0, sigma) draw against choose_commands_variation's Phi(|z|)*(mean-best) --
# not used by training logic, purely for post-hoc inspection.
diag_original_perturbation = {0: [], 1: [], 2: []}
diag_variation_improve_amount = {0: [], 1: [], 2: []}
diag_all_zscores = {0: [], 1: [], 2: []}
diag_gap_and_std = {0: [], 1: [], 2: []}  # (best-mean gap, std) per objective, for checking normalization

@dataclass
class Transition(object):
    observation: np.ndarray
    action: Optional[np.ndarray]
    reward: np.ndarray
    next_observation: np.ndarray
    day: int
    action_executed : bool
    weight : float

device = 'cpu'

def crowding_distance(points):
    # first normalize accross dimensions
    points = (points-points.min(axis=0))/(points.ptp(axis=0)+1e-8)
    # sort points per dimension
    dim_sorted = np.argsort(points, axis=0)
    point_sorted = np.take_along_axis(points, dim_sorted, axis=0)
    # compute distances between lower and higher point
    distances = np.abs(point_sorted[:-2] - point_sorted[2:])
    # pad extrema's with 1, for each dimension
    distances = np.pad(distances, ((1,), (0,)), constant_values=1)
    # sum distances of each dimension of the same point
    crowding = np.zeros(points.shape)
    crowding[dim_sorted, np.arange(points.shape[-1])] = distances
    crowding = np.sum(crowding, axis=-1)
    return crowding

def get_non_dominated(solutions):
    '''function for getting non_dominated solutions'''

    is_efficient = np.ones(solutions.shape[0], dtype=bool)
    for i, c in enumerate(solutions):
        if is_efficient[i]:
            #is_efficient[is_efficient] = np.any(solutions[is_efficient] > c, axis=1) # not strict domination (original code)
            is_efficient[is_efficient] = ~(np.all(solutions[is_efficient] <= c, axis=1) & np.any(solutions[is_efficient] < c, axis=1)) # strict domination
            is_efficient[i] = 1

    return is_efficient

def nlargest_variation(experience_replay, n=200):
    '''function for select n episodes in the buffer'''

    returns = np.array([e[2][0].reward for e in experience_replay])

    nd_i = get_non_dominated(returns)
    nd = returns[nd_i]

    returns_exp = np.tile(np.expand_dims(returns, 1), (1, len(nd), 1))
    l2 = np.min(np.linalg.norm(returns_exp - nd, axis=-1), axis=-1) * -1

    l2_min, l2_max = l2.min(), l2.max()
    if l2_max - l2_min != 0:
        l2 = (l2 - l2_min) / (l2_max - l2_min)

    sorted_i = np.argsort(l2)
    largest = [experience_replay[i] for i in sorted_i[-n:]]

    for i in range(len(l2)):
        experience_replay[i] = (l2[i], experience_replay[i][1], experience_replay[i][2])
    heapq.heapify(experience_replay)

    return largest

def add_episode(transitions, experience_replay, gamma=1., max_size=500, step=0):
    '''function for updating episode'''

    for i in reversed(range(len(transitions)-1)): # compute return
        transitions[i].reward += gamma * transitions[i+1].reward

    if len(experience_replay) == max_size:  # pop smallest episode of heap if full, add new episode
        heapq.heappushpop(experience_replay, (1, step, transitions))
    else:
        heapq.heappush(experience_replay, (1, step, transitions))

def choose_action(env, model, obs, desired_return, desired_horizon, eval=False, action_noise_scale=0.0):
    '''function for choosing action'''

    if isinstance(obs, tuple):
        obs = np.concatenate(obs, axis=0)  # (L+3,) 만들어질 수도 있음
    obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)  # (1, L+3)

    # desired_* 처리 (3-objective 대응)
    if not isinstance(desired_return, torch.Tensor):
        desired_return = torch.tensor(desired_return, dtype=torch.float32).view(1, -1).to(device)  # (1, 3)
    if not isinstance(desired_horizon, torch.Tensor):
        desired_horizon = torch.tensor([desired_horizon], dtype=torch.float32).view(1, 1).to(device)  # (1, 1)

    # forward
    hidden = model(obs, desired_return, desired_horizon)
    hidden = hidden.detach().cpu().numpy()[0]
    action = hidden
    if not eval: # exploration using noise
        action_range = env.action_space.high - env.action_space.low
        noise_std = action_noise_scale * action_range # hyperparameter (noise)
        noise = np.random.normal(0, noise_std, size=action.shape).astype(np.float32)
        action = np.clip(action + noise, env.action_space.low, env.action_space.high)

    return action

def run_episode(demand_env, env, model, desired_return, desired_horizon, episode_len=365, eval=False, action_noise_scale=0.0):
    """
    Run an episode of length `episode_len` days.

    Args:
        env: 환경
        model: 학습/평가 모델
        desired_return: 초기 목표 return
        desired_horizon: 초기 목표 horizon
        episode_len: 실행할 기간 (default 365)
        eval: 평가용 flag
    Returns:
        transitions: list of Transition objects
    """
    transitions = []
    obs = env.reset(demand_env)
    day = 0

    while day < episode_len:
        # 액션 선택
        action = choose_action(env, model, obs, desired_return, desired_horizon, eval=eval, action_noise_scale=action_noise_scale)

        # 환경 step
        if demand_env == 'gs':
            n_obs, reward, day_in_env, action_executed = env.step_gs(action)
        else:
            n_obs, reward, day_in_env, action_executed = env.step_original(action, demand_env)

        # transition 기록
        transitions.append(Transition(
            observation=obs,
            action=action,
            reward=np.float32(reward).copy(),
            next_observation=n_obs,
            day=day_in_env,
            action_executed=action_executed,
            weight=1.0
        ))

        # desired_return/horizon 업데이트
        desired_return -= reward
        desired_horizon = max(desired_horizon - 1, 1.)
        obs = n_obs
        day = day_in_env

    return transitions

def choose_commands_variation(experience_replay, stochastic_objective_selection=False):
    '''
    PCN 모델의 다음 학습 목표(desired_return, desired_horizon)를
    전략적으로 선택하는 함수.
    choose_commands_original과 동일하게, nlargest로 뽑은 pool을
    non-dominated solution만으로 한 번 더 필터링한 뒤 통계량을 계산한다.

    stochastic_objective_selection: False(기본값)면 |z|가 가장 큰 objective를
    deterministic하게 고른다(argmax, 논문 서술과 동일). True면 softmax(|z|)
    확률로 objective를 샘플링한다(예전에 시도했던 stochastic 버전 재현/테스트용).
    '''
    # 1. buffer 전체에서 return과 horizon 추출
    # nlargest 활용
    episodes = nlargest_variation(experience_replay) # nlargest_variation
    returns = np.array([e[2][0].reward for e in episodes])
    horizons = np.array([len(e[2]) for e in episodes])

    # non-dominated filtering (choose_commands_original과 동일한 2단계 구조)
    nd_i = get_non_dominated(returns)
    returns = returns[nd_i]
    horizons = horizons[nd_i]

    # buffer stores reward = (-cost, -freq, -amt) (Eq. 13); Algorithm 3 is defined
    # directly on the minimization objectives, so work in cost-space here and
    # convert the result back to reward-space at the end.
    cost = -returns
    cost_mean = np.mean(cost, axis=0)
    cost_std = np.std(cost, axis=0)
    cost_best = np.min(cost, axis=0)
    z_scores = (cost_best - cost_mean) / (cost_std + 1e-8)  # Algorithm 3, Step 1

    for _obj in range(3):
        diag_all_zscores[_obj].append(float(z_scores[_obj]))
        diag_gap_and_std[_obj].append((float(cost_mean[_obj] - cost_best[_obj]), float(cost_std[_obj])))

    # Step 2: objective with the largest absolute score, worst episode for it
    if stochastic_objective_selection:
        probs = scipy.special.softmax(np.abs(z_scores))
        improve_idx = np.random.choice(3, p=probs)
    else:
        improve_idx = np.argmax(np.abs(z_scores))
    worst_idx_for_return = np.argmax(cost[:, improve_idx])
    desired_cost = cost[worst_idx_for_return].copy()

    # Step 3: improvement magnitude from the best-mean gap
    scaling_factor_return = norm.cdf(np.abs(z_scores[improve_idx]))
    improve_amount = -scaling_factor_return * (cost_best[improve_idx] - cost_mean[improve_idx])
    desired_cost[improve_idx] -= improve_amount

    diag_variation_improve_amount[int(improve_idx)].append(float(improve_amount))

    desired_return = -desired_cost

    # horizon: choose_commands_original과 동일하게, return의 근거가 된
    # episode(worst_idx_for_return)에서 그대로 가져온다.
    desired_horizon = np.float32(horizons[worst_idx_for_return] - 1)

    desired_return = np.float32(desired_return)
    return desired_return, desired_horizon

def choose_commands_original(experience_replay):
    '''function for choosing episode'''
    episodes = nlargest_variation(experience_replay)
    returns, horizons = list(zip(*[(e[2][0].reward, len(e[2])) for e in episodes]))
    nd_i = get_non_dominated(np.array(returns))
    returns = np.array(returns)[nd_i]
    horizons = np.array(horizons)[nd_i]
    m, s = np.mean(returns, axis=0), np.std(returns, axis=0)
    r_i = np.random.randint(0, len(returns))
    desired_horizon = np.float32(horizons[r_i] - 1)  # -2
    desired_return = returns[r_i].copy()
    #r_i = np.random.choice(objectives)
    obj_i = np.random.randint(0, 3)
    perturbation = np.random.uniform(high=s[obj_i])
    desired_return[obj_i] += perturbation
    desired_return = np.float32(desired_return)

    diag_original_perturbation[int(obj_i)].append(float(perturbation))

    return desired_return, desired_horizon

def update_model(model, opt, experience_replay, batch_size=256, noise=0., clip_grad_norm=None, use_weight=False):
    #variation
    '''function for updating model (train/test 모두 사용 가능)'''

    batch = []

    if use_weight: # test
        # Transition 단위로 평탄화해서 weight 기반 샘플링
        flat_eps = []
        for ep_idx, (_, step, ep) in enumerate(experience_replay):
            for t_idx, t in enumerate(ep):
                flat_eps.append((ep_idx, t_idx, t))

        weights = np.array([t.weight for _, _, t in flat_eps], dtype=np.float32)
        weights /= weights.sum()  # normalize
        s_i = np.random.choice(len(flat_eps), size=batch_size, replace=True, p=weights)
        samples = [flat_eps[i] for i in s_i]

    else: # train
        # 기존 방식: episode 단위로 uniform sampling
        s_i = np.random.choice(np.arange(len(experience_replay)), size=batch_size, replace=True)
        samples = []
        for i in s_i:
            ep = experience_replay[i][2]
            t = np.random.randint(0, len(ep))  # 에피소드 내 랜덤 timestep
            samples.append((i, t, ep[t]))

    # Transition → 학습 batch 구성
    for ep_idx, t_idx, transition in samples:
        if transition.action is None:
            continue
        s_t = transition.observation
        a_t = transition.action
        r_t = np.float32(transition.reward)
        h_t = np.float32(len(experience_replay[ep_idx][2]) - t_idx)
        batch.append((s_t, a_t, r_t, h_t))

    if len(batch) == 0:
        return None, None

    # 텐서 변환
    obs, actions, desired_return, desired_horizon = zip(*batch)
    obs = torch.tensor(np.array(obs), dtype=torch.float32).to(device)
    actions_tensor = torch.tensor(np.array(actions), dtype=torch.float32).to(device)
    desired_return = torch.tensor(np.array(desired_return), dtype=torch.float32).to(device)
    desired_horizon = torch.tensor(np.array(desired_horizon), dtype=torch.float32).unsqueeze(1).to(device)

    # return에 노이즈 추가
    desired_return = desired_return + noise * torch.normal(0, 1, size=desired_return.shape, device=desired_return.device)

    # forward & loss
    log_prob = model(obs, desired_return, desired_horizon)
    opt.zero_grad()
    l = F.mse_loss(log_prob, actions_tensor, reduction="mean")
    l.backward()

    if clip_grad_norm is not None:
        for pg in opt.param_groups:
            nn.utils.clip_grad_norm_(pg['params'], clip_grad_norm)

    opt.step()

    return l, log_prob

def update_model_original(model, opt, experience_replay, batch_size, noise=0., clip_grad_norm=None):
    '''function for updating model'''

    batch = []
    s_i = np.random.choice(np.arange(len(experience_replay)), size=batch_size, replace=True)

    for i in s_i:
        ep = experience_replay[i][2]
        # 에피소드 내 랜덤 타임스탬프
        t = np.random.randint(0, len(ep))
        # 만약 action이 None이면 다른 타임스탬프로 다시 선택하거나 continue
        if ep[t].action is None:
            continue
        s_t, a_t, r_t, h_t = ep[t].observation, ep[t].action, np.float32(ep[t].reward), np.float32(len(ep) - t)
        batch.append((s_t, a_t, r_t, h_t))

    obs, actions, desired_return, desired_horizon = zip(*batch)
    obs = np.array(obs)
    obs = torch.tensor(obs, dtype=torch.float32).to(device)
    desired_return = np.array(desired_return)
    desired_return = torch.tensor(desired_return, dtype=torch.float32).to(device)
    desired_return = desired_return + noise * torch.normal(0, 1, size=desired_return.shape, device=desired_return.device)
    desired_horizon = torch.tensor(desired_horizon, dtype=torch.float32).unsqueeze(1).to(device)
    log_prob = model(obs, desired_return, desired_horizon)
    opt.zero_grad()
    actions = np.array(actions)  # 먼저 numpy로 통합
    actions_tensor = torch.tensor(actions, dtype=torch.float32).to(device)
    l = F.mse_loss(log_prob, actions_tensor) # MSE loss function
    #l = F.smooth_l1_loss(log_prob, actions_tensor) # Smooth L1 loss (huber loss)

    l = l.mean()
    l.backward()

    if clip_grad_norm is not None:
        for pg in opt.param_groups:
            nn.utils.clip_grad_norm_(pg['params'], clip_grad_norm)

    opt.step()

    return l, log_prob

def _generate_demand_sequence(mu=0.1, sigma=1.0, S0=20, days=365):
    #sigma 값 조정하여 fluctutation 정도 조정

    dt = 1 / days
    dB = np.random.randn(days) * np.sqrt(dt)
    B = np.cumsum(dB)
    t = np.linspace(0, 1, days)
    demand_seq = S0 * np.exp((mu - 0.5 * sigma ** 2) * t + sigma * B)
    demand_seq = np.clip(demand_seq, 0.0, 60.0)

    return demand_seq

def train(version, demand_env, env, model, run_id, scaling_factor, learning_rate=1e-4, batch_size=256, total_steps=1e5, n_model_updates=10,
          n_step_episodes=10, gamma=1., max_size=500, noise=0.0, clip_grad_norm=None, logdir='runs/',
          pareto_seed_path=None, action_noise_scale=0.0, stochastic_objective_selection=False,
          raw_seed_episode=False):
    """pareto_seed_path: optional override for the evolutionary buffer-init source
    (Algorithm 2). When given, `solution` for the 'variation1'/'variation2' buffer
    initialization is built from this CSV instead of the usual MOPSO CSV -- e.g. to
    seed with CMODE's front instead. Expects the same columns MOPSO/CMODE both use:
    "Q (Order Size)", "k (Safety Stock)", "TC", "Frequency", "Amount". Default (None)
    preserves the original MOPSO-only behavior exactly.

    raw_seed_episode: when True, prepends one noise-free episode per evolutionary
    solution (the raw seed action, replayed unperturbed for all 365 days) before
    the usual n_repeat noisy episodes. Default (False) preserves the original
    Algorithm 2 buffer-init behavior exactly (every repeat, including j=1, has
    noise -- there is never a clean copy of the seed action in the buffer)."""

    step = 0
    path = pathlib.Path(__file__).parent.resolve()
    path_2 = str(path) + str(logdir)
    os.makedirs(path_2)
    logger = Logger(logdir=path_2)

    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    n_checkpoints = 0
    experience_replay = []

    MOPSO_start = time.time()

    if demand_env == 'stationary':

        MOPSO_run = False
        if MOPSO_run == True:
            device = 'cpu'
            env = Inventory(scaling_factor, demand_env)
            bounds = np.array([[0, env.D], [0, env.D]])
            particle_positions = np.array([np.random.uniform(low, high, size=200) for (low, high) in bounds]).T
            # particle_positions = lhs_initial_positions(n_particles=500, bounds=bounds) # lhs sampling
            mopso = MOPSO(num_particles=200, dim=2, bounds=bounds, num_iterations=100, initial_positions=particle_positions)
            #solution = mopso.optimize(env, demand_env)
            solution, solution2, solution3 = mopso.optimize_variation(env, demand_env)

            data = {
                "Q (Order Size)": [p[0][0] for p in solution],
                "k (Safety Stock)": [p[0][1] for p in solution],
                "TC": [p[1][0] for p in solution],
                "Frequency": [p[1][1] for p in solution],
                "Amount": [p[1][2] for p in solution]
            }

            mopso_df = pd.DataFrame(data)
            mopso_df.to_csv(f"./MOPSO/MOPSO_run_{demand_env}_{run_id}.csv", index=False)

            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            # 색상 고정: 회색으로 표시 (필요시 'blue', 'red', 등으로 바꿔도 됨)
            sc = ax.scatter(
                mopso_df["Frequency"],
                mopso_df["Amount"],
                mopso_df["TC"],
                color='blue',  # 또는 c='blue' 등
                s=40,
                edgecolor='k'
            )
            ax.set_xlabel("Stockout Frequency")
            ax.set_ylabel("Stockout Amount")
            ax.set_zlabel("Total Cost(1e3)")
            ax.set_title("Non-Dominated Solutions (MOPSO)")
            plt.savefig(f"./MOPSO/MOPSO_pareto_front_run_{demand_env}_{run_id}.png", dpi=300)
            plt.close()

        # MOPSO csv 읽어와서 사용
        # original demand (stationary)
        device = 'cpu'
        env = Inventory(scaling_factor, demand_env)
        mopso_df = pd.read_csv(f"./MOPSO/MOPSO_run_{demand_env}_{run_id}.csv")
        #mopso_df2 = pd.read_csv(f"./MOPSO/MOPSO_variation_run_{run_id}.csv")  # MOPOS_variation_run : MOPSO의 nlargest solution 사용
        solution = []
        for _, row in mopso_df.iterrows():
            action = [row["Q (Order Size)"], row["k (Safety Stock)"]]
            objective = [row["TC"], row["Frequency"], row["Amount"]]
            solution.append((action, objective))

        if pareto_seed_path is not None:
            seed_df = pd.read_csv(pareto_seed_path)
            solution = [
                ([row["Q (Order Size)"], row["k (Safety Stock)"]], [row["TC"], row["Frequency"], row["Amount"]])
                for _, row in seed_df.iterrows()
            ]

    elif demand_env == 'seasonal':

        MOPSO_run = False
        if MOPSO_run == True:
            # MOPSO 실행 : 시나리오 1(non-stationary : 3 seasons)
            device = 'cpu'
            env = Inventory(scaling_factor, demand_env)
            bounds = np.array([
                [0, env.D], [0, env.D],  # Q1, k1
                [0, env.D], [0, env.D],  # Q2, k2
                [0, env.D], [0, env.D],  # Q3, k3
            ])
            particle_positions_period0 = np.array([np.random.uniform(low, high, size=200) for (low, high) in bounds]).T
            # particle_positions_period0 = lhs_initial_positions(n_particles=num_particles, bounds=bounds) # LHS 사용 시

            mopso_period0 = MOPSO(
                num_particles=len(particle_positions_period0),
                dim=6,  # 전체 차원 (Q,r 쌍 3개 = 6차원)
                bounds=bounds,
                num_iterations=100,
                initial_positions=particle_positions_period0)

            fixed_mask = np.array([True] * 6)
            fixed_mask[0] = False  # Period 0의 Q
            fixed_mask[1] = False  # Period 0의 r

            # period_idx=0으로 optimize 함수 호출
            first_period_archive = mopso_period0.optimize_seasonal(fixed_mask, demand_env, period_idx=0, env=env)
            all_archives = []
            all_archives.append(first_period_archive)


            # --- 첫 번째 기간에서 도출된 Pareto front 저장 ---
            data_for_first_period_pareto_csv = {
                "Q1": [sol_pos[0] for sol_pos, sol_fit in first_period_archive],
                "r1": [sol_pos[1] for sol_pos, sol_fit in first_period_archive],
                "TC": [sol_fit[0] for sol_pos, sol_fit in first_period_archive],
                "Frequency": [sol_fit[1] for sol_pos, sol_fit in first_period_archive],
                "Amount": [sol_fit[2] for sol_pos, sol_fit in first_period_archive]
            }

            mopso_df_first_period_pareto = pd.DataFrame(data_for_first_period_pareto_csv)
            mopso_df_first_period_pareto.to_csv(f"./MOPSO/MOPSO_first_period_front_{run_id}.csv", index=False)
            print(f"첫 번째 기간 Pareto front를 CSV로 저장 완료: {len(mopso_df_first_period_pareto)}개 솔루션")

            # ---

            ### 2. 두 번째, 세 번째 기간에는 첫 번째 기간의 솔루션을 적용

            # 모든 파레토 솔루션에 대한 전체 기간 평가 결과를 저장할 리스트
            # 각 요소는 (원본 Period 0 솔루션, 해당 솔루션으로 3개 기간을 운영했을 때의 총 비용, 총 재고 부족 빈도, 총 재고 부족량)
            full_evaluation_results = []

            # 첫 번째 기간의 아카이브에 있는 각 파레토 솔루션을 순회합니다.
            for sol_pos_period0_only, sol_fitness_period0_only in first_period_archive:

                # 현재 파레토 솔루션의 Period 0 Q, r 값을 추출합니다.
                Q0_current_sol = sol_pos_period0_only[0]
                r0_current_sol = sol_pos_period0_only[1]

                # 이 Q0, r0 값을 모든 3개 기간에 걸쳐 적용할 솔루션을 구성합니다.
                # 즉, (Q0, r0, Q0, r0, Q0, r0) 형태의 6차원 배열을 만듭니다.
                current_full_solution = np.zeros(6)
                for i in range(3):  # Period 0, 1, 2
                    current_full_solution[i * 2] = Q0_current_sol  # 해당 Period의 Q 값
                    current_full_solution[i * 2 + 1] = r0_current_sol  # 해당 Period의 r 값

                # 이 통합 솔루션으로 각 기간의 성능을 평가하고 합산합니다.
                # 총 비용, 총 재고 부족 빈도, 총 재고 부족량을 계산하기 위함입니다.
                total_cost_overall = 0.0
                total_stockout_freq_overall = 0.0
                total_stockout_amt_overall = 0.0

                for p_idx in range(3):  # 모든 Period (0, 1, 2)에 대해 평가
                    cost_p, freq_p, amt_p = evaluate_solution_seasonal(current_full_solution, demand_env, env=env, period_idx=p_idx)

                    total_cost_overall += cost_p
                    total_stockout_freq_overall += freq_p
                    total_stockout_amt_overall += amt_p

                # 이 솔루션에 대한 전체 3개 기간의 통합 성능을 저장합니다.
                full_evaluation_results.append({
                    'period0_solution': (Q0_current_sol, r0_current_sol),
                    'overall_cost': total_cost_overall,
                    'overall_stockout_freq': total_stockout_freq_overall,
                    'overall_stockout_amt': total_stockout_amt_overall
                })

            # --- 3개 기간 통합 관점에서 새로운 Pareto front 구성 (필터링) ---
            final_integrated_pareto_front = []

            for sol_eval_result in full_evaluation_results:
                current_fitness = np.array([sol_eval_result['overall_cost'], sol_eval_result['overall_stockout_freq'], sol_eval_result['overall_stockout_amt']])

                # 현재 솔루션을 최종 프론트에 추가하기 전, 기존 프론트의 솔루션들과 비교
                new_final_front_candidates = []
                is_dominated_by_existing = False

                for existing_sol_in_front in final_integrated_pareto_front:
                    existing_fitness = np.array([existing_sol_in_front['overall_cost'], existing_sol_in_front['overall_stockout_freq'], existing_sol_in_front['overall_stockout_amt']])

                    if dominates(existing_fitness, current_fitness):  # 기존 솔루션이 현재 솔루션을 지배
                        is_dominated_by_existing = True
                        new_final_front_candidates.append(existing_sol_in_front)  # 기존 솔루션은 유지
                    elif dominates(current_fitness, existing_fitness):  # 현재 솔루션이 기존 솔루션을 지배
                        # 이 경우 existing_sol_in_front는 새 프론트에 포함되지 않음 (제거)
                        pass
                    else:  # 서로 지배 관계가 없음 (둘 다 유지)
                        new_final_front_candidates.append(existing_sol_in_front)

                if not is_dominated_by_existing:
                    new_final_front_candidates.append(sol_eval_result)  # 현재 솔루션이 어떤 기존 솔루션에도 지배되지 않으면 추가

                final_integrated_pareto_front = new_final_front_candidates

            # --- 최종 필터링된 Pareto front 결과를 CSV로 저장 ---
            data_for_final_pareto_csv = {
                "Q1": [res['period0_solution'][0] for res in final_integrated_pareto_front],
                "r1": [res['period0_solution'][1] for res in final_integrated_pareto_front],
                "Q2": [res['period0_solution'][0] for res in final_integrated_pareto_front],
                "r2": [res['period0_solution'][1] for res in final_integrated_pareto_front],
                "Q3": [res['period0_solution'][0] for res in final_integrated_pareto_front],
                "r3": [res['period0_solution'][1] for res in final_integrated_pareto_front],
                "TC": [res['overall_cost'] for res in final_integrated_pareto_front],
                "Frequency": [res['overall_stockout_freq'] for res in final_integrated_pareto_front],
                "Amount": [res['overall_stockout_amt'] for res in final_integrated_pareto_front]
            }

            mopso_df_final_pareto = pd.DataFrame(data_for_final_pareto_csv)
            mopso_df_final_pareto.to_csv(f"./MOPSO/MOPSO_run_seasonal_{run_id}.csv", index=False)
            print(f"첫 번째 기간 Pareto front를 CSV로 저장 완료: {len(mopso_df_final_pareto)}개 솔루션")


        #MOPSO 읽어와서 실행
        device = 'cpu'
        env = Inventory(scaling_factor, demand_env)
        mopso_df = pd.read_csv(f"./MOPSO/MOPSO_first_period_front_{run_id}.csv")

        solution = []
        for _, row in mopso_df.iterrows():
            action = [row["Q1"], row["r1"]]
            objective = [row["TC"], row["Frequency"], row["Amount"]]
            solution.append((action, objective))

        if pareto_seed_path is not None:
            seed_df = pd.read_csv(pareto_seed_path)
            solution = [
                ([row["Q1"], row["r1"]], [row["TC"], row["Frequency"], row["Amount"]])
                for _, row in seed_df.iterrows()
            ]

        final_df = pd.read_csv(f"./MOPSO/MOPSO_run_seasonal_{run_id}.csv")
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(
            final_df["Frequency"],
            final_df["Amount"],
            final_df["TC"],
            color='blue',  # 또는 c='blue' 등
            s=40,
            edgecolor='k'
        )

        ax.set_xlabel("Stockout Frequency")
        ax.set_ylabel("Stockout Amount")
        ax.set_zlabel("Total Cost(1e3)")
        ax.set_title("Non-Dominated Solutions (MOPSO)")

        plt.savefig(f"./MOPSO/MOPSO_pareto_front_run_seasonal_{run_id}.png", dpi=300)
        plt.close()

    elif demand_env == 'GBM':

        MOPSO_run = False
        if MOPSO_run == True:
            # MOPSO 실행
            device = 'cpu'
            fixed_seq = _generate_demand_sequence() # GBM

            # CSV로 저장
            #csv_filename = f'fixed_demand_sequence_{run_id}.csv'
            #pd.DataFrame(fixed_seq, columns=['demand']).to_csv(csv_filename, index=False)
            #fixed_seq = pd.read_csv(f'fixed_demand_sequence_{run_id}.csv')['demand'].values

            # Brownian motion
            plt.figure(figsize=(12, 4))
            plt.plot(fixed_seq, label='Generated Demand Sequence')
            plt.xlabel('Day')
            plt.ylabel('Demand')
            plt.title('Demand Sequence')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"fixed_seq_plot_{run_id}.png")
            # plt.show()

            env = Inventory(scaling_factor, demand_env, fixed_seq)
            bounds = np.array([[0, env.D], [0, env.D]])
            particle_positions = np.array([np.random.uniform(low, high, size=200) for (low, high) in bounds]).T
            #particle_positions = lhs_initial_positions(n_particles=500, bounds=bounds) # lhs sampling
            mopso = MOPSO(num_particles=200, dim=2, bounds=bounds, num_iterations=100, initial_positions=particle_positions)
            #solution = mopso.optimize(env, demand_env)
            solution, solution2, solution3 = mopso.optimize_variation(env, demand_env)

            data = {
                "Q (Order Size)": [p[0][0] for p in solution],
                "k (Safety Stock)": [p[0][1] for p in solution],
                "TC": [p[1][0] for p in solution],
                "Frequency": [p[1][1] for p in solution],
                "Amount": [p[1][2] for p in solution]
            }

            mopso_df = pd.DataFrame(data)
            mopso_df.to_csv(f"./MOPSO/MOPSO_run_GBM_{run_id}.csv", index=False)

            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')

            # 색상 고정: 회색으로 표시 (필요시 'blue', 'red', 등으로 바꿔도 됨)
            sc = ax.scatter(
                mopso_df["Frequency"],
                mopso_df["Amount"],
                mopso_df["TC"],
                color='blue',  # 또는 c='blue' 등
                s=40,
                edgecolor='k'
            )

            ax.set_xlabel("Stockout Frequency")
            ax.set_ylabel("Stockout Amount")
            ax.set_zlabel("Total Cost(1e3)")
            ax.set_title("Non-Dominated Solutions (MOPSO)")

            plt.savefig(f"./MOPSO/MOPSO_pareto_front_run_GBM_{run_id}.png", dpi=300)
            plt.close()

        # MOPSO csv 읽어와서 사용
        # original demand (stationary)
        device = 'cpu'
        fixed_seq = _generate_demand_sequence()
        env = Inventory(scaling_factor, demand_env, fixed_seq)
        mopso_df = pd.read_csv(f"./MOPSO/MOPSO_run_GBM_{run_id}.csv")
        # mopso_df = pd.read_csv(f"./MOPSO/MOPSO_solution2_run_gs_{run_id}.csv")  # MOPOS_variation_run : MOPSO의 nlargest solution 사용
        # mopso_df = pd.read_csv(f"./MOPSO/MOPSO_solution3_run_gs_{run_id}.csv")
        solution = []
        for _, row in mopso_df.iterrows():
            action = [row["Q (Order Size)"], row["k (Safety Stock)"]]
            objective = [row["TC"], row["Frequency"], row["Amount"]]
            solution.append((action, objective))

        if pareto_seed_path is not None:
            seed_df = pd.read_csv(pareto_seed_path)
            solution = [
                ([row["Q (Order Size)"], row["k (Safety Stock)"]], [row["TC"], row["Frequency"], row["Amount"]])
                for _, row in seed_df.iterrows()
            ]

    elif demand_env == 'gs':

        MOPSO_run = False
        if MOPSO_run == True:
            # MOPSO 실행 (단일 season)
            device = 'cpu'
            # 파일 경로 설정 (저장된 pickle 파일의 실제 경로에 맞게 수정해주세요)
            pickle_file_path = 'data/daily_demand_2015_good1.pkl'
            #pickle_file_path = './daily_demand_2015_good2.pkl'

            with open(pickle_file_path, 'rb') as f:  # 'rb'는 읽기 바이너리 모드
                fixed_seq_2015 = pickle.load(f)  # 불러온 데이터를 fixed_seq 변수에 바로 할당
            print(f"Demand sequence loaded successfully from {pickle_file_path}")

            # stationary (gs)
            env = Inventory(scaling_factor, demand_env, fixed_seq_2015)
            bounds = np.array([[0, env.D], [0, env.D]])
            particle_positions = np.array([np.random.uniform(low, high, size=200) for (low, high) in bounds]).T
            #particle_positions = lhs_initial_positions(n_particles=500, bounds=bounds) # lhs sampling
            mopso = MOPSO(num_particles=200, dim=2, bounds=bounds, num_iterations=100, initial_positions=particle_positions)
            #solution = mopso.optimize(env, demand_env)
            solution, solution2, solution3 = mopso.optimize_variation(env, demand_env)

            data = {
                "Q (Order Size)": [p[0][0] for p in solution],
                "k (Safety Stock)": [p[0][1] for p in solution],
                "TC": [p[1][0] for p in solution],
                "Frequency": [p[1][1] for p in solution],
                "Amount": [p[1][2] for p in solution]
            }

            mopso_df = pd.DataFrame(data)
            mopso_df.to_csv(f"./MOPSO/MOPSO_run_gs_{run_id}.csv", index=False)

            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')

            # 색상 고정: 회색으로 표시 (필요시 'blue', 'red', 등으로 바꿔도 됨)
            sc = ax.scatter(
                mopso_df["Frequency"],
                mopso_df["Amount"],
                mopso_df["TC"],
                color='blue',  # 또는 c='blue' 등
                s=40,
                edgecolor='k'
            )

            ax.set_xlabel("Stockout Frequency")
            ax.set_ylabel("Stockout Amount")
            ax.set_zlabel("Total Cost(1e6)")
            ax.set_title("Non-Dominated Solutions (MOPSO)")

            plt.savefig(f"./MOPSO/MOPSO_pareto_front_run_{demand_env}_{run_id}.png", dpi=300)
            plt.close()

        # MOPSO csv 읽어와서 사용
        # original demand (stationary)
        device = 'cpu'
        # 파일 경로 설정 (저장된 pickle 파일의 실제 경로에 맞게 수정해주세요)
        pickle_file_path = 'data/daily_demand_2015_good1.pkl'
        #pickle_file_path = './daily_demand_2015_good2.pkl'

        with open(pickle_file_path, 'rb') as f:  # 'rb'는 읽기 바이너리 모드
            fixed_seq_2015 = pickle.load(f)  # 불러온 데이터를 fixed_seq 변수에 바로 할당
        print(f"Demand sequence loaded successfully from {pickle_file_path}")

        # stationary (gs)
        env = Inventory(scaling_factor, demand_env, fixed_seq_2015)
        mopso_df = pd.read_csv(f"./MOPSO/MOPSO_run_gs_{run_id}.csv")
        #mopso_df = pd.read_csv(f"./MOPSO/MOPSO_solution2_run_gs_{run_id}.csv")  # MOPOS_variation_run : MOPSO의 nlargest solution 사용
        #mopso_df = pd.read_csv(f"./MOPSO/MOPSO_solution3_run_gs_{run_id}.csv")
        solution = []
        for _, row in mopso_df.iterrows():
            action = [row["Q (Order Size)"], row["k (Safety Stock)"]]
            objective = [row["TC"], row["Frequency"], row["Amount"]]
            solution.append((action, objective))

        if pareto_seed_path is not None:
            seed_df = pd.read_csv(pareto_seed_path)
            solution = [
                ([row["Q (Order Size)"], row["k (Safety Stock)"]], [row["TC"], row["Frequency"], row["Amount"]])
                for _, row in seed_df.iterrows()
            ]

    MOPSO_end = time.time()
    PCN_start = time.time()

    if version == 'original':
        # fill buffer with random episodes
        low = env.action_space.low
        high = env.action_space.high
        bounds = np.stack([low, high], axis=1)  # shape: (dim, 2)
        #lhs_actions = lhs_initial_positions(n_particles=max_size, bounds=bounds)

        for idx in range(max_size):
            transitions = []
            obs = env.reset(demand_env)
            day = 0
            #action = lhs_actions[idx]
            #action = np.random.uniform(low=low, high=high, size=env.action_space.shape)
            while day <= 365:
                action = np.random.uniform(low=low, high=high, size=env.action_space.shape)
                if demand_env == 'gs':
                    n_obs, reward, day, action_executed = env.step_gs(action)
                else: #stationary, seasonal, GBM
                    n_obs, reward, day, action_executed = env.step_original(action, demand_env)
                transitions.append(Transition(obs, action, np.float32(reward).copy(), n_obs, day, action_executed, weight=1.0))
                obs = n_obs
                step += 1
            add_episode(transitions, experience_replay, gamma=gamma, max_size=max_size, step=step)

    elif version == 'variation1' or version == 'variation2':
        # fill initial buffer using the evolutionary solutions (paper Algorithm 2)
        device = 'cpu'
        low = torch.tensor(env.action_space.low, dtype=torch.float32).to(device)
        high = torch.tensor(env.action_space.high, dtype=torch.float32).to(device)
        n_repeat = max(1, max_size // len(solution))
        epsilon = 0.0
        min_noise_scale = 0.01 * (high - low)

        for sol in solution:
            base_action = torch.tensor(np.array(sol[0]), dtype=torch.float32).to(device)

            if raw_seed_episode:
                transitions = []
                obs = env.reset(demand_env)
                day = 0
                while day <= 365:
                    action_np = base_action.cpu().detach().numpy()
                    if demand_env == 'gs':
                        n_obs, reward, day, action_executed = env.step_gs(action_np)
                    else:  # stationary, seasonal, GBM
                        n_obs, reward, day, action_executed = env.step_original(action_np, demand_env)
                    transitions.append(Transition(obs, action_np, np.float32(reward).copy(), n_obs, day, action_executed, weight=1.0))
                    obs = n_obs
                    step += 1
                add_episode(transitions, experience_replay, gamma=gamma, max_size=max_size, step=step)

            for j in range(1, n_repeat + 1):
                scaling = min(np.sqrt(j / n_repeat), 1.0)
                curr_epsilon = epsilon + scaling * (1.0 - epsilon)

                noise_scale = curr_epsilon * base_action.abs()
                revised_noise_scale = torch.maximum(noise_scale, min_noise_scale)
                initial_noise = torch.empty_like(base_action).uniform_(-1.0, 1.0) * revised_noise_scale
                noisy_action = torch.clamp(base_action + initial_noise, min=low, max=high)

                transitions = []
                obs = env.reset(demand_env)
                day = 0
                while day <= 365:
                    step_noise = torch.empty_like(noisy_action).uniform_(-curr_epsilon, curr_epsilon)
                    action = torch.clamp(noisy_action + step_noise * noisy_action, min=low, max=high)
                    action_np = action.cpu().detach().numpy()
                    if demand_env == 'gs':
                        n_obs, reward, day, action_executed = env.step_gs(action_np)
                    else: #stationary, seasonal, GBM
                        n_obs, reward, day, action_executed = env.step_original(action_np, demand_env)
                    transitions.append(Transition(obs, action_np, np.float32(reward).copy(), n_obs, day, action_executed, weight=1.0))
                    obs = n_obs
                    step += 1

                add_episode(transitions, experience_replay, gamma=gamma, max_size=max_size, step=step)

    print(f"(Warm up completed) Initial buffer filled with {len(experience_replay)} episodes.")
    buffer_init_end_step = step  # diagnostic: episodes with heap step < this came from buffer-init
    n_explore_episodes_total = 0
    n_explore_episodes_temp_nondom = 0
    n_target_iterations = 0
    n_target_would_be_nondom = 0

    # ✅ 경험이 충분히 쌓인 이후: ref_point 자동 계산
    rewards = np.array([e[2][0].reward for e in experience_replay])  # shape (N, D)
    worst = rewards.min(axis=0)
    range_ = rewards.max(axis=0) - worst
    ref_point = worst - 0.1 * range_
    while step < total_steps:

        rewards = np.array([t[2][0].reward for t in experience_replay])
        non_dom = get_non_dominated(rewards)

        loss = []
        entropy = []

        for _ in range(n_model_updates):
            '''l, lp = update_model(model, opt, experience_replay, batch_size=batch_size, noise=noise, clip_grad_norm=clip_grad_norm)
            loss.append(l.detach().cpu().numpy())
            lp = lp.detach().cpu().numpy()
            entropy.append(np.sum(-np.exp(lp) * lp))'''
            l, lp = update_model(model, opt, experience_replay, batch_size=batch_size, noise=noise, clip_grad_norm=clip_grad_norm, use_weight=False)
            loss.append(l.detach().cpu().numpy())
            lp = lp.detach().cpu().numpy()
            max_lp = np.max(lp)
            stable_lp = lp - max_lp
            exp_lp = np.exp(stable_lp)
            prob = exp_lp / np.sum(exp_lp)  # softmax된 확률
            entropy.append(-np.sum(prob * stable_lp))

        if version == 'original' or version == 'variation1':
            desired_return, desired_horizon = choose_commands_original(experience_replay)
        elif version == 'variation2':
            desired_return, desired_horizon = choose_commands_variation(experience_replay, stochastic_objective_selection=stochastic_objective_selection)

            # diagnostic: if the target itself (not what actually gets
            # achieved) were inserted into the current buffer as-is, would it
            # be non-dominated? Distinguishes "target design problem" from
            # "execution-gap problem".
            n_target_iterations += 1
            cur_rewards_for_target_check = np.array([t[2][0].reward for t in experience_replay])
            target_check = np.vstack([cur_rewards_for_target_check, np.asarray(desired_return)])
            target_non_dom = get_non_dominated(target_check)
            if target_non_dom[-1]:
                n_target_would_be_nondom += 1

        leaves = np.array([(len(e[2]), e[2][0].reward) for e in experience_replay[len(experience_replay) // 2:]], dtype=object)
        e_lengths, e_returns = zip(*leaves)
        e_returns = np.array(e_returns)

        try:
            if len(experience_replay) == max_size:
                logger.put('train/leaves', e_returns, step, f'{e_returns.shape[-1]}d')
            from pygmo import hypervolume
            hv = hypervolume(e_returns)
            hv_est = hv.compute(ref_point)
            logger.put('train/hypervolume', hv_est, step, 'scalar')
        except ValueError:
            pass

        returns = []
        horizons = []

        for _ in range(n_step_episodes):
            transitions = run_episode(demand_env, env, model, desired_return, desired_horizon, 365, action_noise_scale=action_noise_scale)
            step += len(transitions)
            add_episode(transitions, experience_replay, gamma=gamma, max_size=max_size, step=step)
            returns.append(transitions[0].reward)
            horizons.append(len(transitions))

            # diagnostic: was this exploration-origin episode non-dominated
            # (within the buffer as it stood right after insertion)? add_episode
            # uses a heap, so the new entry isn't necessarily at index -1 --
            # find it by its (unique, monotonically increasing) step value.
            n_explore_episodes_total += 1
            cur_rewards = np.array([t[2][0].reward for t in experience_replay])
            cur_non_dom = get_non_dominated(cur_rewards)
            just_added_i = next(i for i, e in enumerate(experience_replay) if e[1] == step)
            if cur_non_dom[just_added_i]:
                n_explore_episodes_temp_nondom += 1

        if step >= (n_checkpoints + 1) * total_steps / 10:  # total 10 models are generated
            print("\n\nSAVE : " + str(f'{logger.logdir}/model_{n_checkpoints + 1}.pt'))
            torch.save(model, f'{logger.logdir}/model_{n_checkpoints + 1}.pt')
            n_checkpoints += 1

    # 최종 시점 기준으로 non-dominated solution 추출
    rewards = np.array([t[2][0].reward for t in experience_replay])
    non_dom = get_non_dominated(rewards)
    non_dom_rewards = rewards[non_dom]
    non_dom_trajectories = [experience_replay[i][2] for i in range(len(experience_replay)) if non_dom[i]]

    # diagnostic: how many of the final non-dominated episodes trace back to
    # buffer-init (heap step < buffer_init_end_step) vs the exploration loop
    non_dom_steps = np.array([experience_replay[i][1] for i in range(len(experience_replay)) if non_dom[i]])
    n_from_init = int(np.sum(non_dom_steps < buffer_init_end_step))
    n_from_explore = int(np.sum(non_dom_steps >= buffer_init_end_step))
    print(f"[diagnostic] non-dominated origin: {n_from_init} from buffer-init, {n_from_explore} from exploration loop (out of {len(non_dom_steps)})")
    print(f"[diagnostic] exploration episodes that were non-dominated AT INSERTION TIME: {n_explore_episodes_temp_nondom} / {n_explore_episodes_total}")
    if n_target_iterations > 0:
        print(f"[diagnostic] desired_return TARGET would have been non-dominated (if perfectly achieved): {n_target_would_be_nondom} / {n_target_iterations}")

    # CSV 저장
    df = pd.DataFrame({
        "TC": [-1 * p[0] for p in non_dom_rewards],
        "Frequency": [-1 * p[1] for p in non_dom_rewards],
        "Amount": [-1 * p[2] for p in non_dom_rewards]
    })
    df.to_csv(f"./PCN/PCN_{demand_env}_{version}_run_{run_id}.csv", index=False)

    action_records = {
        "Episode": [],
        "Step": [],  # 또는 Day
        "Q (Order Size)": [],
        "r (Reorder Point)": []  # k 대신 r이면 이름도 같이 맞추기
        # "Day": []          # transition에 day 정보 있으면 추가 가능
    }

    for ep_idx, traj in enumerate(non_dom_trajectories):
        for t_idx, transition in enumerate(traj):

            # 주문이 실제로 실행된 시점만 기록
            if getattr(transition, "action_executed", False):
                curr_action = transition.action  # [Q, r] or [Q, k]

                action_records["Episode"].append(ep_idx)
                action_records["Step"].append(t_idx)  # 하루 = 한 step이면 day로 써도 됨
                # action_records["Day"].append(transition.day)  # day 속성이 따로 있으면 이렇게

                action_records["Q (Order Size)"].append(curr_action[0])
                action_records["r (Reorder Point)"].append(curr_action[1])

    df_action_traj = pd.DataFrame(action_records)
    csv_path = os.path.join(path_2, f"{version}_non_dominated_action_trajectories_{run_id}.csv")
    df_action_traj.to_csv(csv_path, index=False)

    plot_non_dominated_rewards(experience_replay, non_dom, demand_env, version, run_id, path_2)
    plot_inventory_levels(non_dom_trajectories, version, run_id, path_2)
    plot_cost(non_dom_trajectories, version, run_id, path_2)
    plot_stockout_frequency(non_dom_trajectories, version, run_id, path_2)
    plot_stockout_amount(non_dom_trajectories, version, run_id, path_2)
    plot_Q_and_r_trajectories(non_dom_trajectories, version, run_id, path_2)

    print(f"(train completed) Initial buffer filled with {len(experience_replay)} episodes.")

    print("\n--- Starting sequential evaluation with 2016 data ---")

    test = False
    if demand_env == 'gs':
        pickle_file_path = 'data/daily_demand_2016_good1.pkl'
        #pickle_file_path = './daily_demand_2016_good2.pkl'
        with open(pickle_file_path, 'rb') as f:
            fixed_seq_2016 = pickle.load(f)
        run_test(version, demand_env, model, experience_replay, path_2, run_id, fixed_seq_2016, scaling_factor)
        if test == True:
            run_test_month(version, demand_env, model, opt, experience_replay, path_2, run_id, fixed_seq_2015, fixed_seq_2016, scaling_factor)

    else: # stationary, seasonal, GBM
        pass

    #test0
    '''actions_by_episode = {}
    for ep_idx, e in enumerate(experience_replay):
        traj = e[2]  # trajectory (list of Transition)
        actions = []
        for step in traj:
            q_value = step.action[0]
            k_value = step.action[1]
            actions.append(np.array([q_value, k_value]))
        actions_by_episode[ep_idx] = actions

    # --- 2. 2016년 테스트 데이터 로드 ---
    pickle_file_path = 'data/daily_demand_2016_good1.pkl'
    #pickle_file_path = './daily_demand_2016_good2.pkl'
    with open(pickle_file_path, 'rb') as f:
        fixed_seq = pickle.load(f)

    # --- 3. 각 Action Trajectory에 대해 시뮬레이션 실행 ---
    all_trajectories_results = []

    device = 'cpu'

    for ep_idx, action_sequence in actions_by_episode.items():
        env = Inventory(scaling_factor, demand_env, fixed_seq)
        day = 0
        env.reset()

        # **수정된 부분: total_reward를 초기화하고 누적**
        total_reward = np.array([0.0, 0.0, 0.0])
        while day <= 364:
            n_obs, reward, day, last_order_day = env.step(action_sequence[day])
            total_reward += reward

        # 시뮬레이션 결과(누적된 total_reward)를 기반으로 지표 계산
        total_cost = -1 * total_reward[0]
        stockout_frequency = -1 * total_reward[1]
        stockout_amount = -1 * total_reward[2]

        all_trajectories_results.append({
            "Q": action_sequence[0][0],  # 첫 번째 Action의 Q 값 사용
            "k": action_sequence[0][1],  # 첫 번째 Action의 k 값 사용
            "TC": total_cost,
            "Frequency": stockout_frequency,
            "Amount": stockout_amount
        })

    # --- 4. 파레토 프론트 도출 및 결과 저장 ---
    results_df = pd.DataFrame(all_trajectories_results)
    objectives = results_df[['TC', 'Frequency', 'Amount']].values * -1
    is_pareto_front = get_non_dominated(objectives)
    pareto_front_df = results_df[is_pareto_front].copy()

    # === [추가] 파레토 프론트에 해당하는 액션 궤적 저장 ===
    action_records = {
        "Episode": [],
        "Step": [],
        "Q (Order Size)": [],
        "k (Safety Stock)": []
    }

    for ep_idx in pareto_front_df.index:  # Pareto front에 해당하는 episode index만 사용
        traj_actions = actions_by_episode[ep_idx]
        for t_idx, curr_action in enumerate(traj_actions):
            action_records["Episode"].append(ep_idx)
            action_records["Step"].append(t_idx)
            action_records["Q (Order Size)"].append(curr_action[0])
            action_records["k (Safety Stock)"].append(curr_action[1])

    df_action_traj = pd.DataFrame(action_records)
    csv_path = os.path.join(path_2, f"non_dominated_action_trajectories_test0_{run_id}.csv")
    df_action_traj.to_csv(csv_path, index=False)
    print(f"Non-dominated action trajectories saved to '{csv_path}'")

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(
        pareto_front_df["Frequency"],
        pareto_front_df["Amount"],
        pareto_front_df["TC"],
        color='blue',
        s=40,
        edgecolor='k'
    )

    ax.set_xlabel("Stockout Frequency")
    ax.set_ylabel("Stockout Amount")
    ax.set_zlabel("Total Cost(1e6)")
    ax.set_title("Non-Dominated Solutions (PCN_test0)")

    png_path_output = os.path.join(path_2, f"PCN_pareto_front_run_test0_{run_id}.png")
    plt.savefig(png_path_output, dpi=300)
    plt.close()

    print(f"새로운 파레토 프론트 결과가 '{png_path_output}'에 저장되었습니다.")

    # 파레토 프론트 결과를 CSV 파일로 저장

    selected_columns_df = pareto_front_df[['TC', 'Frequency', 'Amount']]
    selected_columns_df.to_csv(f"./PCN/original_PCN_run_test0_{run_id}.csv", index=False)'''

    PCN_end = time.time()
    MOPSO_elapsed_time = MOPSO_end - MOPSO_start
    PCN_elapsed_time = PCN_end - PCN_start

    return non_dom_rewards, MOPSO_elapsed_time, PCN_elapsed_time

def run_test_month(version, demand_env, model, opt, experience_replay, path_2, run_id,
                   fixed_seq_2015, fixed_seq_2016, scaling_factor ,num_month=3, device="cpu"):
    import torch, numpy as np, pandas as pd, os

    gamma = 1.0
    step = 0
    total_steps = 1e6
    max_size = 500
    n_model_updates = 10
    n_step_episodes = 10

    candidate_returns = np.array([e[2][0].reward for e in experience_replay])
    candidate_horizons = np.array([len(e[2]) for e in experience_replay])

    all_monthly_results = []
    action_records = {"Month": [], "Episode": [], "Step": [], "Q (Order Size)": [], "k (Safety Stock)": []}

    # --- num_month 개수만큼 균등 분할 ---
    month_lengths = [len(fixed_seq_2016) // num_month] * num_month
    remainder = len(fixed_seq_2016) % num_month
    for i in range(remainder):
        month_lengths[i] += 1

    start_idx = 0
    for month_idx, month_len in enumerate(month_lengths):
        month_results = []

        # --- Rolling window 설정 (항상 365일) ---
        month_end = start_idx + month_len
        train_end_2016 = month_end
        train_start_2015 = max(0, len(fixed_seq_2015) - (365 - train_end_2016))
        # rolling-window 전체 (1년)
        train_seq = np.concatenate([fixed_seq_2015[train_start_2015:], fixed_seq_2016[:train_end_2016]])
        # 평가 구간 (겹치지 않음)
        eval_seq = fixed_seq_2016[start_idx:month_end]

        # --- 공통 환경 객체 생성 ---
        env = Inventory(scaling_factor, demand_env, train_seq)
        env_eval = Inventory(scaling_factor, demand_env, eval_seq)

        # ========== [기존: PCN rolling-window update 학습] ==========
        """
        for ep_idx, (desired_return, desired_horizon) in enumerate(zip(candidate_returns, candidate_horizons)):
            obs = env.reset()
            total_reward = np.array([0.0, 0.0, 0.0])
            daily_rewards, actions = [], []
            day = 0

            while day < len(train_seq):
                action = choose_action(env, model, obs, desired_return, desired_horizon, eval=False)
                actions.append(action)
                n_obs, reward, day_in_env, _ = env.step_gs(action)
                total_reward += reward
                daily_rewards.append(reward)
                desired_return -= reward
                desired_horizon = max(desired_horizon - 1, 1.)
                obs = n_obs
                day = day_in_env

            start = month_idx * month_len
            end = min((month_idx + 1) * month_len, len(daily_rewards))
            month_reward = np.sum(daily_rewards[start:end], axis=0)

            month_results.append({
                "Q": actions[0][0],
                "k": actions[0][1],
                "TC": -month_reward[0],
                "Frequency": -month_reward[1],
                "Amount": -month_reward[2]
            })
        """

        # ========== [수정: 최신 rolling-window만 학습에 사용] ==========
        for ep_idx, (desired_return, desired_horizon) in enumerate(zip(candidate_returns, candidate_horizons)):
            obs = env.reset(demand_env)
            total_reward = np.array([0.0, 0.0, 0.0])
            daily_rewards, actions = [], []
            day = 0

            while day < len(train_seq):
                action = choose_action(env, model, obs, desired_return, desired_horizon, eval=False)
                actions.append(action)
                n_obs, reward, day_in_env, _ = env.step_gs(action)
                total_reward += reward
                daily_rewards.append(reward)
                desired_return -= reward
                desired_horizon = max(desired_horizon - 1, 1.)
                obs = n_obs
                day = day_in_env

            month_reward = np.sum(daily_rewards, axis=0)

            month_results.append({
                "Q": actions[0][0],
                "k": actions[0][1],
                "TC": -month_reward[0],
                "Frequency": -month_reward[1],
                "Amount": -month_reward[2]
            })

            for t_idx, curr_action in enumerate(actions):
                action_records["Month"].append(month_idx + 1)
                action_records["Episode"].append(ep_idx)
                action_records["Step"].append(t_idx)
                action_records["Q (Order Size)"].append(curr_action[0])
                action_records["k (Safety Stock)"].append(curr_action[1])

        # --- PCN 학습 루프 ---
        for _ in range(n_model_updates):
            update_model(model, opt, experience_replay)

        if version == 'original' or version == 'variation1':
            desired_return, desired_horizon = choose_commands_original(experience_replay)
        elif version == 'variation2':
            desired_return, desired_horizon = choose_commands_variation(experience_replay)

        for _ in range(n_step_episodes):
            transitions = run_episode(demand_env, env, model, desired_return, desired_horizon, episode_len=len(train_seq))
            add_episode(transitions, experience_replay, gamma=gamma, max_size=max_size, step=step)
            step += len(transitions)

        # --- PCN 결과 저장 및 시각화 ---
        pcn_df = pd.DataFrame(month_results[:len(candidate_returns)])
        pcn_objectives = pcn_df[['TC', 'Frequency', 'Amount']].values * -1
        pcn_is_pareto = get_non_dominated(pcn_objectives)
        pcn_pareto_df = pcn_df[pcn_is_pareto].copy()

        pcn_csv_path = os.path.join(path_2, f"PCN_month{month_idx + 1}_run{run_id}.csv")
        pcn_pareto_df.to_csv(pcn_csv_path, index=False)
        print(f"📘 PCN Month {month_idx + 1} Pareto front saved to '{pcn_csv_path}'")

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(
            pcn_pareto_df["Frequency"],
            pcn_pareto_df["Amount"],
            pcn_pareto_df["TC"],
            color='blue',
            s=40,
            edgecolor='k'
        )
        ax.set_xlabel("Stockout Frequency")
        ax.set_ylabel("Stockout Amount")
        ax.set_zlabel("Total Cost")
        ax.set_title(f"PCN Pareto Front Month {month_idx + 1}")
        png_path = os.path.join(path_2, f"PCN_pareto_front_month{month_idx + 1}_run{run_id}.png")
        plt.savefig(png_path, dpi=300)
        plt.close()

        # --- MOPSO 최적화 및 평가 ---
        print(f"\n🔁 Month {month_idx + 1}: MOPSO 최적화 시작")
        bounds = np.array([[0, env.D], [0, env.D]])
        mopso = MOPSO(num_particles=200, dim=2, bounds=bounds, num_iterations=100, initial_positions=None)
        archive, top_solutions, _ = mopso.optimize_variation(env, demand_env='gs')

        for sol in top_solutions:
            Q, k = sol[0]
            cost, freq, amt = evaluate_solution_original(Q, k, env_eval, 'gs', period=len(eval_seq))
            month_results.append({
                "Q": Q,
                "k": k,
                "TC": cost,
                "Frequency": freq,
                "Amount": amt
            })

        # --- 통합 Pareto front 도출 및 저장 ---
        month_df = pd.DataFrame(month_results)
        objectives = month_df[['TC', 'Frequency', 'Amount']].values * -1
        is_pareto = get_non_dominated(objectives)
        pareto_month_df = month_df[is_pareto].copy()
        all_monthly_results.append(pareto_month_df)

        csv_path = os.path.join(path_2, f"MOPSO_month{month_idx + 1}_run{run_id}.csv")
        pareto_month_df.to_csv(csv_path, index=False)
        print(f"✅ Month {month_idx + 1} Pareto front saved to '{csv_path}'")

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(
            pareto_month_df["Frequency"],
            pareto_month_df["Amount"],
            pareto_month_df["TC"],
            color='purple',
            s=40,
            edgecolor='k'
        )
        ax.set_xlabel("Stockout Frequency")
        ax.set_ylabel("Stockout Amount")
        ax.set_zlabel("Total Cost")
        ax.set_title(f"Non-Dominated Solutions Month {month_idx + 1}")
        png_path = os.path.join(path_2, f"PCN_MOPSO_pareto_front_month{month_idx + 1}_run{run_id}.png")
        plt.savefig(png_path, dpi=300)
        plt.close()

        start_idx = month_end

    df_action_traj = pd.DataFrame(action_records)
    df_action_traj.to_csv(os.path.join(path_2, f"monthly_action_trajectories_run{run_id}.csv"), index=False)
    print("📊 All monthly action trajectories saved.")

    return all_monthly_results, df_action_traj

def run_test(version, demand_env, model, experience_replay, path_2, run_id, fixed_seq_2016, scaling_factor):

    device = 'cpu'
    #scaling_factor = torch.tensor([[0.005, 0.1, 0.1, 0.1]]).to(device)

    # --- (B) buffer에서 desired_return 후보 추출 ---
    candidate_returns = [e[2][0].reward for e in experience_replay]
    candidate_horizons = [len(e[2]) for e in experience_replay]

    all_trajectories_results = []
    action_records = {
        "Episode": [],
        "Step": [],
        "Q (Order Size)": [],
        "k (Safety Stock)": []
    }

    # --- (C) 각 desired_return으로 1년 simulation 실행 ---
    for ep_idx, (desired_return, desired_horizon) in enumerate(zip(candidate_returns, candidate_horizons)):

        print(f"[Episode {ep_idx}] desired_return: {desired_return}, desired_horizon: {desired_horizon}")

        env = Inventory(scaling_factor, demand_env, fixed_seq_2016)
        obs = env.reset(demand_env)
        total_reward = np.array([0.0, 0.0, 0.0])
        actions = []
        day = 0

        while day < 365:
            action = choose_action(env, model, obs, desired_return, desired_horizon, eval=False)
            actions.append(action)
            if demand_env == 'gs':
                n_obs, reward, day_in_env, last_order_day = env.step_gs(action)
            else:
                n_obs, reward, day_in_env, last_order_day = env.step_original(action, demand_env)
            total_reward += reward
            desired_return -= reward
            desired_horizon = max(desired_horizon - 1, 1.)
            obs = n_obs
            day = day_in_env

        # 지표 계산
        total_cost = -1 * total_reward[0]
        stockout_frequency = -1 * total_reward[1]
        stockout_amount = -1 * total_reward[2]

        all_trajectories_results.append({
            "Q": actions[0][0],
            "k": actions[0][1],
            "TC": total_cost,
            "Frequency": stockout_frequency,
            "Amount": stockout_amount
        })

        # trajectory 기록 (Pareto filtering 전에 전체 저장)
        for t_idx, curr_action in enumerate(actions):
            action_records["Episode"].append(ep_idx)
            action_records["Step"].append(t_idx)
            action_records["Q (Order Size)"].append(curr_action[0])
            action_records["k (Safety Stock)"].append(curr_action[1])

    # --- (D) Pareto front 도출 ---
    results_df = pd.DataFrame(all_trajectories_results)
    objectives = results_df[['TC', 'Frequency', 'Amount']].values * -1
    is_pareto_front = get_non_dominated(objectives)
    pareto_front_df = results_df[is_pareto_front].copy()

    # === Pareto front action trajectory만 따로 저장 ===
    df_action_traj = pd.DataFrame(action_records)
    csv_path = os.path.join(path_2, f"non_dominated_action_trajectories_test1_{version}_{run_id}.csv")
    df_action_traj.to_csv(csv_path, index=False)
    print(f"Non-dominated action trajectories saved to '{csv_path}'")

    # --- (E) 시각화 ---
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        pareto_front_df["Frequency"],
        pareto_front_df["Amount"],
        pareto_front_df["TC"],
        color='blue',
        s=40,
        edgecolor='k'
    )
    ax.set_xlabel("Stockout Frequency")
    ax.set_ylabel("Stockout Amount")
    ax.set_zlabel("Total Cost(1e6)")
    ax.set_title("Non-Dominated Solutions (PCN_test)")
    png_path_output = os.path.join(path_2, f"PCN_pareto_front_run_test1_{version}_{run_id}.png")
    plt.savefig(png_path_output, dpi=300)
    plt.close()
    print(f"새로운 파레토 프론트 결과가 '{png_path_output}'에 저장되었습니다.")

    # --- (F) Pareto front 결과를 CSV 파일로 저장 ---
    selected_columns_df = pareto_front_df[['TC', 'Frequency', 'Amount']]
    selected_columns_df.to_csv(f"./PCN/original_PCN_run_test1_{version}_{run_id}.csv", index=False)

    return pareto_front_df

def lhs_initial_positions(n_particles, bounds):
    dim = bounds.shape[0]
    sampler = qmc.LatinHypercube(d=dim)
    sample = sampler.random(n=n_particles)
    scaled_sample = qmc.scale(sample, bounds[:, 0], bounds[:, 1])  # bounds 내로 스케일링

    return scaled_sample  # shape: (n_particles, dim)

def plot_action_trajectories_from_archive(archive, run_id, max_plot=5):
    plt.figure(figsize=(10, 6))

    for i, (position, fitness) in enumerate(archive[:max_plot]):
        Qs = [position[j] for j in range(0, 8, 2)]
        rs = [position[j] for j in range(1, 8, 2)]
        days = [0, 91, 181, 271]  # 각 기간 시작일

        plt.plot(days, Qs, label=f'Solution {i} - Q', linestyle='-', marker='o')
        plt.plot(days, rs, label=f'Solution {i} - r', linestyle='--', marker='x')

    plt.xlabel('Day')
    plt.ylabel('Action value')
    plt.title('Action Trajectories over Periods')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"./MOPSO/MOPSO_Q_r_run{run_id}.png", dpi=300)
    plt.close()

def plot_non_dominated_rewards(experience_replay, non_dom, demand_env, version, run_id, dirr):
    '''function for plotting pareto front'''

    rewards = np.array([t[2][0].reward for t in experience_replay])
    non_dom_rewards = rewards[non_dom]
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(-1 * non_dom_rewards[:, 1],  # stockout_frequency
        -1 * non_dom_rewards[:, 2],  # stockout_amount
        -1 * non_dom_rewards[:, 0],  # total_cost
        c='blue', marker='o', label='Non-dominated'
    )
    ax.set_xlabel('Stockout Frequency')
    ax.set_ylabel('Stockout Amount')
    if demand_env == 'statioanry':
        ax.set_zlabel('Total Cost(1e3)')
    elif demand_env == 'seasonal':
        ax.set_zlabel('Total Cost(1e3)')
    elif demand_env == 'GBM':
        ax.set_zlabel('Total Cost(1e3)')
    elif demand_env == 'gs':
        ax.set_zlabel('Total Cost(1e6)')
    ax.set_title('Non-Dominated Solutions(original PCN)')
    ax.legend()
    img_path = os.path.join(dirr, f"PCN_pareto_front_run_{version}_{run_id}.png")
    plt.savefig(img_path, dpi=300)
    plt.close()

def plot_inventory_levels(non_dom_trajectories, version, run_id, dirr, max_plot=5):
    plt.figure(figsize=(12, 6))

    for idx, episode in enumerate(non_dom_trajectories[:max_plot]):
        inventory_levels = [t.observation[0] for t in episode]
        days = list(range(len(inventory_levels)))
        plt.plot(days, inventory_levels, label=f'Episode {idx + 1}')

    plt.xlabel("Day")
    plt.ylabel("Inventory Level")
    plt.title("Inventory Level Over Time (Non-dominated Episodes)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    img_path = os.path.join(dirr, f"PCN_inven_run_{version}_{run_id}.png")
    plt.savefig(img_path, dpi=300)
    plt.close()

def plot_cost(non_dom_trajectories, version, run_id, dirr):
    plt.figure(figsize=(12, 6))

    max_len = max(len(ep) for ep in non_dom_trajectories)
    cost_matrix = np.full((len(non_dom_trajectories), max_len), np.nan)

    for i, episode in enumerate(non_dom_trajectories):
        cost = [-1 * t.reward[0] for t in episode]
        cost_matrix[i, :len(cost)] = cost

    avg_cost = np.nanmean(cost_matrix, axis=0)
    days = list(range(len(avg_cost)))

    plt.plot(days, avg_cost, color='blue', label='Average Daily Cost')
    plt.xlabel("Day")
    plt.ylabel("Cost")
    plt.title("Average Daily Cost Over Time (Non-dominated Episodes)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    img_path = os.path.join(dirr, f"PCN_avg_daily_cost_run_{version}_{run_id}.png")
    plt.savefig(img_path, dpi=300)
    plt.close()

def plot_stockout_frequency(non_dom_trajectories, version, run_id, dirr):
    plt.figure(figsize=(12, 6))

    max_len = max(len(ep) for ep in non_dom_trajectories)
    freq_matrix = np.full((len(non_dom_trajectories), max_len), np.nan)

    for i, episode in enumerate(non_dom_trajectories):
        freq = [-1 * t.reward[1] for t in episode]
        freq_matrix[i, :len(freq)] = freq

    avg_freq = np.nanmean(freq_matrix, axis=0)
    days = list(range(len(avg_freq)))

    plt.plot(days, avg_freq, color='darkorange', label='Average Daily Stockout Frequency')
    plt.xlabel("Day")
    plt.ylabel("Stockout Frequency")
    plt.title("Average Daily Stockout Frequency Over Time (Non-dominated Episodes)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    img_path = os.path.join(dirr, f"PCN_avg_daily_frequency_run_{version}_{run_id}.png")
    plt.savefig(img_path, dpi=300)
    plt.close()

def plot_stockout_amount(non_dom_trajectories, version, run_id, dirr):
    plt.figure(figsize=(12, 6))

    max_len = max(len(ep) for ep in non_dom_trajectories)
    amt_matrix = np.full((len(non_dom_trajectories), max_len), np.nan)

    for i, episode in enumerate(non_dom_trajectories):
        amt = [-1 * t.reward[2] for t in episode]
        amt_matrix[i, :len(amt)] = amt

    avg_amt = np.nanmean(amt_matrix, axis=0)
    days = list(range(len(avg_amt)))

    plt.plot(days, avg_amt, color='crimson', label='Average Daily Stockout Amount')
    plt.xlabel("Day")
    plt.ylabel("Stockout Amount")
    plt.title("Average Daily Stockout Amount Over Time (Non-dominated Episodes)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    img_path = os.path.join(dirr, f"PCN_avg_daily_amount_run_{version}_{run_id}.png")
    plt.savefig(img_path, dpi=300)
    plt.close()

def plot_Q_and_r_trajectories(non_dom_trajectories, version, run_id, dirr, max_plot=5):
    """
    non_dom_trajectories : 리스트[episode]
        episode는 transition들의 리스트라고 가정
        각 transition t는 t.action (길이 2: [Q, r]),
        t.action_executed (bool, 주문이 실제로 발생한 날) 을 속성으로 가짐.

    version : str   # 알고리즘 이름 (예: 'PCN_variation1')
    run_id  : int   # 실험 번호
    dirr    : str   # 저장할 디렉터리 경로
    max_plot: int   # 최대 몇 개의 Pareto 솔루션을 그릴지
    """

    # 그릴 에피소드 수 제한
    episodes_to_plot = non_dom_trajectories[:max_plot]
    max_len = max(len(ep) for ep in episodes_to_plot)

    # (솔루션 개수, 최대 기간) 크기의 행렬에 NaN으로 초기화
    Q_matrix = np.full((len(episodes_to_plot), max_len), np.nan)
    r_matrix = np.full((len(episodes_to_plot), max_len), np.nan)

    # 주문이 실제로 실행된 날만 Q, r 기록
    for i, episode in enumerate(episodes_to_plot):
        for day_idx, t in enumerate(episode):
            if getattr(t, "action_executed", False):
                Q_matrix[i, day_idx] = t.action[0]  # Q
                r_matrix[i, day_idx] = t.action[1]  # r

    days = np.arange(max_len)

    plt.figure(figsize=(12, 6))

    # 각 솔루션별로 Q(실선), r(점선) 플롯
    for i in range(len(episodes_to_plot)):
        # Q
        plt.plot(
            days, Q_matrix[i],
            marker='o',
            linestyle='-',
            label=f"Solution {i} - Q"
        )
        # r
        plt.plot(
            days, r_matrix[i],
            marker='x',
            linestyle='--',
            label=f"Solution {i} - r"
        )

    plt.xlabel("Day")
    plt.ylabel("Action value")
    plt.title(f"Action trajectories at order times ({version}, Run {run_id})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    img_path = os.path.join(dirr, f"{version}_Qr_trajectories_run_{run_id}.png")
    plt.savefig(img_path, dpi=300)
    plt.close()


'''def plot_Q_values(non_dom_trajectories, run_id, dirr, max_plot=5):
    import matplotlib.pyplot as plt
    import os

    plt.figure(figsize=(12, 6))

    for idx, episode in enumerate(non_dom_trajectories[:max_plot]):
        Q_values = []
        days = []
        for day, t in enumerate(episode):
            if getattr(t, "action_executed", False):
                Q_values.append(t.action[0])
                days.append(day)
        if Q_values:
            plt.plot(days, Q_values, marker='o', label=f'Solution {idx} - Q')

    plt.xlabel("Day")
    plt.ylabel("Q (Order Quantity)")
    plt.title("Order Quantity Over Time (Non-dominated Episodes)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    img_path = os.path.join(dirr, f"PCN_Q_run{run_id}.png")
    plt.savefig(img_path, dpi=300)
    plt.close()

def plot_k_values(non_dom_trajectories, run_id, dirr, max_plot=5):
    import matplotlib.pyplot as plt
    import os

    plt.figure(figsize=(12, 6))

    for idx, episode in enumerate(non_dom_trajectories[:max_plot]):
        k_values = []
        days = []
        for day, t in enumerate(episode):
            if getattr(t, "action_executed", False):
                k_values.append(t.action[1])
                days.append(day)
        if k_values:
            plt.scatter(days, k_values, label=f'Episode {idx + 1}', s=25)  # 점 크기 s=25 (필요시 조정)

    plt.xlabel("Day")
    plt.ylabel("k (Safety Stock Factor)")
    plt.title("Safety Stock Factor (k) Over Time (Non-dominated Episodes)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    img_path = os.path.join(dirr, f"PCN_k_run{run_id}.png")
    plt.savefig(img_path, dpi=300)
    plt.close()'''