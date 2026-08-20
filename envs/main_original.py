import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _sub in ("baselines", "envs", "models", "utils", "experiments"):
    _p = _os.path.join(_ROOT, _sub)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
_os.chdir(_ROOT)

import gym
import numpy as np
from scipy.stats import truncnorm, norm
from scipy.stats import gamma
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt
from win32comext.mapi.mapitags import pidAttachReadOnlyMin


class ContinuousHead(nn.Module):
    def __init__(self, base, demand_env):
        super(ContinuousHead, self).__init__()
        self.base = base

        if demand_env == 'stationary':
            mu = 3.22
            sigma = 0.45
            theoretical_mean = np.exp(mu + (sigma ** 2) / 2)
            # theoretical_var = (np.exp(sigma ** 2) - 1) * np.exp(2 * mu + sigma ** 2)
            # theoretical_std = np.sqrt(theoretical_var)
            self.D = theoretical_mean * 365

            '''self.mu1 = 3412 / 365
            self.std1 = 53.354 / np.sqrt(5)
            self.D = self.mu1 * 365'''

            '''self.mu1 = 10.0
            self.std1 = 2.0
            self.D = self.mu1 * 365'''

        elif demand_env == 'seasonal':
            self.mu1 = 10.0
            self.std1 = 2.0
            self.D = self.mu1 * 365

        elif demand_env == 'GBM':
            self.D = 20 * 365

        elif demand_env == 'gs':
            # GS retail good1 = max(daily demand) * 365
            self.D = 10 * 365
            # GS retail good2 = max(daily demand) * 365
            #self.D = 5 * 365

    def forward(self, state, desired_return, desired_horizon):
        '''function for embedding'''

        x = self.base(state, desired_return, desired_horizon)
        x = torch.tanh(x) # x ∈ [-1, 1]
        low = torch.tensor([0.0, 0.0], dtype=x.dtype, device=x.device)
        high = torch.tensor([self.D, self.D], dtype=x.dtype, device=x.device)
        scaled = 0.5 * (x + 1) * (high - low) + low # scaling : Q ∈ [0, D], k ∈ [0, D/sigma_L]

        return scaled

class Inventory(nn.Module):
    def __init__(self, scaling_factor, demand_env, fixed_demand_seq=None):
        super(Inventory, self).__init__()
        self.demand_log = []
        self._demand_plot_saved = False
        self.fixed_demand_seq = fixed_demand_seq

        if demand_env == 'gs':
            # GS retail (good 1)
            self.max_L = 5 # lead time
            self.A = 14250  # ordering cost per unit
            self.p = 19000  # the selling price per unit
            self.K = 38000  # setup cost
            self.hc = 9.5  # the unit inventory holding cost per unit time
            self.D = 10 * 365

            # GS retail (good2)
            '''self.max_L = 5  # lead time
            self.A = 8625  # ordering cost per unit (100)
            self.p = 11500  # selling price per unit 
            self.K = 23000 # setup cost
            self.hc = 5.75  # the unit inventory holding cost per unit time
            self.D = 5 * 365'''

        else: #stationary, seasonal, GBM
            self.A = 80  # ordering cost (100)
            self.p = 10  # the purchase price per unit
            self.c = 27.5  # unit item cost
            self.h = 0.26  # inventory carrying rate
            self.hc = 5  # the unit inventory holding cost per unit time

            if demand_env == 'stationary':
                '''self.mu1 = 10
                self.std1 = 2.0
                self.D = self.mu1 * 365
                self.max_L = 5  # lead time'''

                '''self.mu1 = 3412 / 365
                self.std1 = 53.354 / np.sqrt(5)
                self.D = self.mu1 * 365
                self.max_L = 30  # lead time'''

                mu = 3.22
                sigma = 0.45
                theoretical_mean = np.exp(mu + (sigma ** 2) / 2)
                #theoretical_var = (np.exp(sigma ** 2) - 1) * np.exp(2 * mu + sigma ** 2)
                #theoretical_std = np.sqrt(theoretical_var)
                self.D = theoretical_mean * 365
                self.max_L = 5  # lead time

            elif demand_env == 'seasonal':
                self.mu1 = 10.0
                self.std1 = 2.0
                self.D = self.mu1 * 365
                self.max_L = 5  # lead time

            elif demand_env == 'GBM':
                self.D = 20 * 365
                self.max_L = 5  # lead time

        self.action_space = gym.spaces.Box(low=np.array([0.0, 0.0], dtype=np.float32), high=np.array([self.D, self.D], dtype=np.float32), dtype=np.float32)
        self.scaling_factor = scaling_factor

        if demand_env == 'gs':
            self.s_emb = nn.Sequential(nn.Linear(2, 64), nn.Sigmoid())  # 2
        else:
            self.s_emb = nn.Sequential(nn.Linear(self.max_L + 1, 64), nn.Sigmoid()) # 2

        self.c_emb = nn.Sequential(nn.Linear(4, 64), nn.Sigmoid()) # [desired_return, desired_horizon]
        self.fc = nn.Sequential(nn.Linear(64, 2))  #fully connected layer (latent representation)

    def forward(self, state, desired_return, desired_horizon):
        '''function for embedding'''

        c = torch.cat((desired_return, desired_horizon), dim=-1)
        c = c * self.scaling_factor
        c = self.c_emb(c)
        s = self.s_emb(state.float())  # (batch_size, 64)
        hidden = self.fc(s*c)

        return hidden

    def get_demand_seasonal(self):
        '''function for generating demand from a truncated normal distribution (seasonal)'''

        if self.day <= 120: # period 1 : truncated normal
            mu = self.mu1
            std = self.std1
            truncated_point = mu * 3
            while True:
                d = np.random.normal(mu, std)
                if 0.0 <= d < truncated_point:
                    return d
        elif self.day <= 240: # period 2 : truncated gamma
            alpha = 10.0
            beta = 1.0
            truncated_point = alpha / beta * 3

            while True:
                d = np.random.gamma(alpha, 1. / beta)
                if 0.0 <= d < truncated_point:
                    return d
        else: # period 3 : poisson
            lamb = 10
            truncated_point = lamb * 3
            while True:
                d = np.random.poisson(lamb)
                if 0.0 <= d < truncated_point:
                    return d

    def get_demand_stationary(self):
        '''function for generating demand from a truncated normal distribution (stationary)'''

        mu = 3.22 # underlying normal mean
        sigma = 0.45  # underlying normal std
        d = np.random.lognormal(mean=mu, sigma=sigma)
        return d

        '''mu = self.mu1
        std = self.std1

        truncated_point = mu * 3

        while True:
            d = np.random.normal(mu, std)
            if 0.0 <= d < truncated_point:
                return d'''

        '''alpha = 10.0
        beta = 1.0
        truncated_point = alpha / beta * 3

        while True:
            d = np.random.gamma(alpha, 1. / beta)
            if 0.0 <= d < truncated_point:
                return d'''

        '''lamb = 10
        d = np.random.poisson(lamb)
        return d'''

    def get_demand_fixed(self):
        '''function for generating demand (GS retail, GBM)'''

        if self.day >= len(self.fixed_demand_seq):
            return 0.0

        demand = self.fixed_demand_seq[self.day]
        self.demand_log.append(demand)

        return demand

    def _save_demand_plot(self):
        # 로그 디렉토리 생성
        import os
        os.makedirs("logs", exist_ok=True)

        # Plot 생성 및 저장
        plt.figure(figsize=(10, 4))
        plt.plot(self.demand_log, label='Daily Demand')
        plt.xlabel('Day')
        plt.ylabel('Demand')
        plt.title(f'Simulated Daily Demand (ID: {id(self)})')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"./demand_plot_{id(self)}.png")
        plt.close()

    def get_lead_time(self):
        '''function for generating lead time as a discrete random variable'''

        lead_time_options = [5]
        probabilities = [1.0]

        return np.random.choice(lead_time_options, p=probabilities)

    def get_lead_time_gs(self):
        '''function for generating lead time as a discrete random variable'''

        lead_time_options = [3, 4, 5]
        probabilities = [0.3, 0.4, 0.3]

        return np.random.choice(lead_time_options, p=probabilities)

    def reset(self, demand_env, day=0):
        '''function for setting the initial environment'''

        self.inventory_level_list = np.array([0] * 400) # inventory level during 1 year
        self.on_order_list = np.array([0] * 400) # expected arrival order(future)
        self.order_list = np.array([0] * 400) # order history(past)
        self.day = day # day index
        self.max_change = 365 / self.max_L # budget
        #self.max_change = 365
        self.num_change = 0
        self.current_action = (0, 0)
        self.action_log = []  # 🔍 매일의 action 기록용 리스트 추가

        if demand_env == 'gs':
            obs = np.concatenate([[self.inventory_level_list[self.day]],  # → shape (1,)
            [np.sum(self.on_order_list[self.day + 1: self.day + self.max_L + 1])]], axis=0)  # [np.sum(self.on_order_list[self.day + 1: self.day + self.max_L + 1])]], axis=0)
        else:
            obs = np.concatenate([[self.inventory_level_list[self.day]],  # → shape (1,)
            self.on_order_list[self.day + 1: self.day + self.max_L + 1]], axis=0)  # [np.sum(self.on_order_list[self.day + 1: self.day + self.max_L + 1])]], axis=0)

        self.demand_log = []
        self._demand_plot_saved = False
        #[(self.max_change - self.num_change) / (367 - self.day)]

        return obs

    def step_original(self, action, demand_env):
        '''function for continuous review system(Q,r) : duplicated order is allowed'''
        #stationary, seasonal, GBM

        if action is not None: # when action is changed
            self.current_action = action # update the current action
            self.num_change += 1 # count the number of actions changes

        Q, r = self.current_action # use the current action if action is not changed
        self.action_log.append((self.day, self.current_action))  # 🔍 현재 day의 action 기록
        self.action_executed = False
        cost = 0
        stockout_frequency = 0
        stockout_amount = 0

        L = self.get_lead_time()

        self.day += 1  # update the date
        self.inventory_level_list[self.day] = self.inventory_level_list[self.day - 1]
        self.inventory_level_list[self.day] += self.on_order_list[self.day] # replenishment at the start of the day
        cost += self.h / 365 * self.c * max(0, self.inventory_level_list[self.day])  # compute holding cost right after replenishment
        #cost += self.hc * max(0, self.inventory_level_list[self.day])

        if demand_env == 'stationary':
            demand = self.get_demand_stationary()
        elif demand_env == 'seasonal':
            demand = self.get_demand_seasonal()
        elif demand_env == 'GBM':
            demand = self.get_demand_fixed()

        self.inventory_level_list[self.day] -= demand  # reflect daily demand to inventory level

        if self.day > self.max_L and self.inventory_level_list[self.day] < 0: # compute frequency of stockout and amount of stockout
            stockout_frequency += 1
            stockout_amount += self.inventory_level_list[self.day] * -1

        #and all(abs(x) < 1e-6 for x in self.on_order_list[self.day + 1 : self.day + L])
        #if self.day + L < 365 and self.day - self.last_order_day >= L and self.inventory_level_list[self.day] < (self.D / 365) * L + k * self.sigma_L: # continuous review system(Q,k)
        if self.day + L < 365 and self.inventory_level_list[self.day] + sum(self.on_order_list[self.day + 1 : self.day + self.max_L + 1]) < r: #PCN(Q,r)
            self.on_order_list[self.day + L] += Q  # order if the current inventory position(=current inventory level + sum of on_order_list) is lower than safety stock
            cost += self.A  # fixed ordering cost
            #cost += self.p * Q # ordering cost per unit
            self.action_executed = True

        n_obs = np.concatenate([[self.inventory_level_list[self.day]],  # → shape (1,)
        self.on_order_list[self.day + 1: self.day + self.max_L + 1]], axis=0)  # [np.sum(self.on_order_list[self.day + 1: self.day + self.max_L + 1])]]

        #[(self.max_change - self.num_change) / (367 - self.day)]

        if demand_env == 'stationary':
            reward = np.array([cost * -1 / 1000, stockout_frequency * -1, stockout_amount * -1]) # reward
        elif demand_env == 'seasonal':
            reward = np.array([cost * -1 / 1000, stockout_frequency * -1, stockout_amount * -1])  # reward
        elif demand_env == 'GBM':
            reward = np.array([cost * -1 / 1000, stockout_frequency * -1, stockout_amount * -1])  # reward

        return n_obs, reward, self.day, self.action_executed

    def step_gs(self, action):
        '''function for continuous review system(Q,r) : GS retail'''

        if action is not None: # when action is changed
            self.current_action = action # update the current action
            self.num_change += 1 # count the number of actions changes

        Q, r = self.current_action # use the current action if action is not changed
        self.action_log.append((self.day, self.current_action))  # 🔍 현재 day의 action 기록
        self.action_executed = False
        cost = 0
        stockout_frequency = 0
        stockout_amount = 0

        L = self.get_lead_time_gs()
        #L = self.get_deterministic_lead_time()

        self.day += 1  # update the date
        self.inventory_level_list[self.day] = self.inventory_level_list[self.day - 1]
        self.inventory_level_list[self.day] += self.on_order_list[self.day] # replenishment at the start of the day
        cost += self.hc * max(0, self.inventory_level_list[self.day])  # compute holding cost right after replenishment
        demand = self.get_demand_fixed()
        #cost -= self.p * min(demand, max(0, self.inventory_level_list[self.day])) # revenue
        self.inventory_level_list[self.day] -= demand # reflect daily demand to inventory level
        if self.day > self.max_L and self.inventory_level_list[self.day] < 0: # compute frequency of stockout and amount of stockout
            stockout_frequency += 1
            stockout_amount += self.inventory_level_list[self.day] * -1

        #and all(abs(x) < 1e-6 for x in self.on_order_list[self.day + 1 : self.day + L])
        #if self.day + L < 365 and self.day - self.last_order_day >= L and self.inventory_level_list[self.day] < (self.D / 365) * L + k * self.sigma_L: # continuous review system(Q,k)
        if self.day + L < 365 and self.inventory_level_list[self.day] + sum(self.on_order_list[self.day + 1 : self.day + self.max_L + 1]) < r: #PCN(Q,r)
            self.on_order_list[self.day + L] += Q  # order if the current inventory position(=current inventory level + sum of on_order_list) is lower than safety stock
            cost += self.K # setup cost
            cost += Q * self.A # ordering cost
            self.action_executed = True

        n_obs = np.concatenate([[self.inventory_level_list[self.day]],  # → shape (1,)
        [np.sum(self.on_order_list[self.day + 1: self.day + self.max_L + 1])]], axis=0)  # [np.sum(self.on_order_list[self.day + 1: self.day + self.max_L + 1])]

        #[(self.max_change - self.num_change) / (367 - self.day)]

        reward = np.array([cost * -1 / 1e6, stockout_frequency * -1, stockout_amount * -1]) # cost is reward

        return n_obs, reward, self.day, self.action_executed

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

def compute_hypervolume(pcn_front, ref_point = [10000, 100, 1000]):

    """
    pcn_front: shape (N, 3), where each row is an objective vector
    ref_point: list or array of size 3, representing the reference point for HV calculation
    """
    # Hypervolume expects minimization; no need to invert if already using -cost etc.
    from pygmo import hypervolume
    hv = hypervolume(pcn_front)
    hv_value = hv.compute(ref_point)

    return hv_value

if __name__ == "__main__":
    import torch
    from datetime import datetime
    import uuid
    from pcn_continuous import train # PCN original
    import time
    import random
    import pandas as pd
    import pickle
    from MOPSO import *
    from pcn_continuous import *

    device = 'cpu'

    n_runs = 10
    version = 'original'
    demand_env = 'stationary'

    pcn_results = []
    spacings_pcn = []
    spreads_pcn = []
    hv_pcn = []

    total_pcn_time = 0
    total_MOPSO_time = 0

    for run_id in range(1, n_runs + 1):
        print(f"\n[Run {run_id}/{n_runs}]")
        seed = 42 + run_id  # 각 run마다 다르게 설정

        # 고정 시드 설정: run_id 기준
        random.seed(run_id)
        np.random.seed(run_id)
        torch.manual_seed(run_id)

        ## --------------------- PCN 실행 ---------------------
        device = 'cpu'
        if demand_env == 'stationary':
            scaling_factor = torch.tensor([[0.000125, 0.1, 1.0, 0.1]]).to(device)
        elif demand_env == 'seasonal':
            scaling_factor = torch.tensor([[0.0001, 0.1, 0.1, 0.1]]).to(device)
        elif demand_env == 'GBM':
            scaling_factor = torch.tensor([[0.0001, 0.1, 0.1, 0.1]]).to(device)
        elif demand_env == 'gs':
            scaling_factor = torch.tensor([[0.0001, 0.1, 0.1, 0.1]]).to(device)

        env = Inventory(scaling_factor, demand_env)

        nA = np.prod(env.action_space.shape)
        model = Inventory(scaling_factor, demand_env).to(device)
        model = ContinuousHead(model, demand_env)
        env.nA = nA

        logdir = f'/PCN/PCN_{demand_env}_{version}'
        logdir += datetime.now().strftime('%Y-%m-%d_%H-%M-%S_') + str(uuid.uuid4())[:4] + '/'
        print(f"[PCN logdir: {logdir}]")

        non_dom_rewards, MOPSO_elapsed_time, PCN_elapsed_time = train(
            version,
            demand_env,
            env,
            model,
            run_id,
            scaling_factor,
            learning_rate=1e-4,
            batch_size=256,
            total_steps=1e6,
            n_model_updates=100,
            n_step_episodes=10,
            max_size=500,
            noise=0.0,
            logdir=logdir,
            clip_grad_norm= None
        )
        total_pcn_time += PCN_elapsed_time
        total_MOPSO_time += MOPSO_elapsed_time

        df = pd.DataFrame({
            "TC": [-1 * p[0] for p in non_dom_rewards],
            "Frequency": [-1 * p[1] for p in non_dom_rewards],
            "Amount": [-1 * p[2] for p in non_dom_rewards]
        })
        #df.to_csv(f"./PCN/original_PCN_run_{run_id}.csv", index=False)

        pcn_front = np.array([[-1 * r[0], -1 * r[1], -1 * r[2]] for r in non_dom_rewards])
        pcn_results.append(pcn_front)
        spacings_pcn.append(spacing(pcn_front))
        spreads_pcn.append(max_spread(pcn_front))

        if demand_env == 'gs':
            mopso_df = pd.read_csv(f"./MOPSO/MOPSO_run_{demand_env}_test1_{run_id}.csv")

            solution = []
            for _, row in mopso_df.iterrows():
                action = [row["Q (Order Size)"], row["k (Safety Stock)"]]
                objective = [row["TC"], row["Frequency"], row["Amount"]]
                solution.append((action, objective))

            pickle_file_path = 'data/daily_demand_2016_good1.pkl'
            #pickle_file_path = './daily_demand_2016_good2.pkl'

            with open(pickle_file_path, 'rb') as f:
                fixed_seq = pickle.load(f)
            print(f"Demand sequence loaded successfully from {pickle_file_path}")

            scaling_factor = torch.tensor([[0.0001, 0.1, 0.1, 0.1]]).to(device)
            env = Inventory(scaling_factor, demand_env, fixed_demand_seq=fixed_seq)

            test_results_list = []

            for sol_idx, sol in enumerate(solution):
                Q, k = sol[0]

                # evaluate_solution 함수를 호출하여 새로운 지표를 계산
                if demand_env == 'seasonal':
                    TC, Frequency, Amount = evaluate_solution_seasonal(Q, k, fixed_seq)
                else: # stationary, GBM, gs
                    TC, Frequency, Amount = evaluate_solution_original(Q, k, env, demand_env)

                # 결과를 기존 형식의 딕셔너리로 구성
                test_result = {
                    "Q (Order Size)": Q,
                    "k (Safety Stock)": k,
                    "TC": TC,
                    "Frequency": Frequency,
                    "Amount": Amount
                }

                test_results_list.append(test_result)

            # --- 2. 시뮬레이션 결과를 DataFrame으로 변환하고 Plotting ---

            # MOPSO 폴더가 없으면 생성

            # test_results_list를 DataFrame으로 변환
            test_df = pd.DataFrame(test_results_list)

            # TC, Frequency, Amount 세 가지 목적 함수를 사용하여 파레토 프론트 추출
            # 목적 함수: TC, Frequency, Amount 모두 작을수록 좋으므로,
            # 지표를 모두 -1을 곱하여 '클수록 좋은' 문제로 바꿔서 get_non_dominated 함수에 전달
            objectives = test_df[['TC', 'Frequency', 'Amount']].values * -1
            is_pareto_front = get_non_dominated(objectives)

            # 파레토 프론트에 해당하는 솔루션만 필터링
            pareto_front_df = test_df[is_pareto_front].copy()

            # 기존 형식과 동일하게 CSV 파일로 저장

            pareto_front_df.to_csv(f"./MOPSO/MOPSO_run_{demand_env}_test1_{run_id}.csv", index=False)

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
            ax.set_title("Non-Dominated Solutions (MOPSO)")

            plt.savefig(f"./MOPSO/MOPSO_pareto_front_run_{demand_env}_test1_{run_id}.png", dpi=300)

            plt.close()


    print("n_runs, demand_env, MOPSO_elapsed_time(1 run), PCN_elapsed_time(1 run)", n_runs, demand_env, total_MOPSO_time, total_pcn_time)
    ###test(GS retail)




'''    print("\n================= Summary =================")
    print(f"Spacing (PCN): {np.mean(spacings_pcn):.4f} ± {np.std(spacings_pcn):.4f}")
    print(f"Max Spread (PCN): {np.mean(spreads_pcn):.4f} ± {np.std(spreads_pcn):.4f}")
    #print(f"HV (PCN): {np.mean(hv_pcn):.4f} ± {np.std(hv_pcn):.4f}")'''