"""Runs the CAPQL PAPER AUTHORS' OWN, UNMODIFIED classes (models/capql.py,
models/networks.py, utils/replay_memory.py, utils/utils.py from
https://github.com/haoyelu/CAPQL, cloned into ./_capql_reference/) against
our Inventory environment, instead of our own reimplementation in
capql_baseline.py.

Purpose: capql_baseline.py is a from-scratch port of the reference repo's
logic (same math, rewritten into our own file with a kwargs-based
constructor instead of the reference's Hydra config object). A manual
line-by-line comparison found no discrepancies, but that is not the same
guarantee as literally running the authors' code. This script removes that
remaining doubt by importing their classes verbatim and only writing the
minimal glue needed: (1) our own InventoryMOEnv wrapper (unavoidable --
their repo has no equivalent for this problem), and (2) a plain object
standing in for their Hydra `config` (CAPQL.__init__ only reads a few
attributes off it; no actual Hydra/OmegaConf machinery is required).

Run in the `pcn` conda env:

    conda activate pcn
    python capql_verbatim_check.py
"""

import random
import sys
import time
import types

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "./_capql_reference")
from models.capql import CAPQL as ReferenceCAPQL  # noqa: E402
from utils.replay_memory import ReplayMemory as ReferenceReplayMemory  # noqa: E402
from utils.utils import Weight_Sampler_pos as ReferenceWeightSampler  # noqa: E402
sys.path.pop(0)

from capql_baseline import InventoryMOEnv, calibrate_reward_scale_from_baseline, non_dominated_front  # noqa: E402


def make_reference_config(gamma, tau, lr, alpha, hidden_size, target_update_interval, cuda):
    cfg = types.SimpleNamespace()
    cfg.training = types.SimpleNamespace(
        gamma=gamma, tau=tau, lr=lr, alpha=alpha, target_update_interval=target_update_interval,
    )
    cfg.model = types.SimpleNamespace(type="CAPQL", hidden_size=hidden_size)
    cfg.gpu = types.SimpleNamespace(cuda=cuda)
    return cfg


def train_capql_verbatim(env, total_steps, reward_scale, start_steps=10000, batch_size=256, gamma=0.99,
                          tau=0.005, lr=3e-4, alpha=0.2, replay_size=1_000_000,
                          updates_per_step=1, device="cpu", verbose=True, diag_every=500):
    num_inputs = env.reset().shape[0]
    cfg = make_reference_config(gamma, tau, lr, alpha, hidden_size=256, target_update_interval=1,
                                 cuda=(device == "cuda"))
    agent = ReferenceCAPQL(num_inputs, env.action_space, env.rwd_dim, cfg)
    memory = ReferenceReplayMemory(replay_size, seed=0)
    sampler = ReferenceWeightSampler(env.rwd_dim)

    if verbose:
        print(f"  [verbatim] reward_scale = {reward_scale}")

    total_numsteps = 0
    updates = 0
    i_episode = 0
    recent_actions = []
    recent_scaled_rewards = []

    while total_numsteps < total_steps:
        state = env.reset()
        episode_steps = 0
        done = False
        i_episode += 1

        while not done:
            w = sampler.sample(1).view(-1)
            if start_steps > total_numsteps:
                action = np.random.uniform(env.action_space.low, env.action_space.high)
            else:
                action = agent.select_action(state, w.numpy())
            recent_actions.append(action.copy())

            if len(memory) > batch_size:
                for _ in range(updates_per_step):
                    qf1_loss, qf2_loss, policy_loss = agent.update_parameters(memory, batch_size, updates)
                    updates += 1
                    if verbose and updates % diag_every == 0:
                        act_arr = np.array(recent_actions[-diag_every:])
                        rew_arr = np.array(recent_scaled_rewards[-diag_every:]) if recent_scaled_rewards else np.zeros((1, env.rwd_dim))
                        print(f"    [update {updates}] qf1_loss={qf1_loss:.4g} qf2_loss={qf2_loss:.4g} policy_loss={policy_loss:.4g} "
                              f"action_mean={act_arr.mean(axis=0)} action_std={act_arr.std(axis=0)} "
                              f"scaled_reward_mean={rew_arr.mean(axis=0)}")

            next_state, reward, done, _ = env.step(action)
            recent_scaled_rewards.append(reward * reward_scale)
            episode_steps += 1
            total_numsteps += 1

            mask = 1.0 if episode_steps == env._max_episode_steps else float(not done)
            memory.push(state, action, w.numpy(), reward * reward_scale, next_state, mask)
            state = next_state

        if verbose and i_episode % 20 == 0:
            print(f"  [episode {i_episode}] total_steps={total_numsteps}/{total_steps} updates={updates}")

    return agent


def evaluate_front_verbatim(agent, env, n_eval_weights=10, n_episodes_per_weight=1):
    sampler = ReferenceWeightSampler(env.rwd_dim)
    eval_weights = sampler.sample(n_eval_weights).numpy()
    vec_returns = []
    for w in eval_weights:
        rep_returns = []
        for _ in range(n_episodes_per_weight):
            state = env.reset()
            done = False
            ep_reward = np.zeros(env.rwd_dim)
            while not done:
                action = agent.select_action(state, w, evaluate=True)
                state, reward, done, _ = env.step(action)
                ep_reward += reward
            rep_returns.append(ep_reward)
        vec_returns.append(np.mean(rep_returns, axis=0))
    return non_dominated_front(np.array(vec_returns))


if __name__ == "__main__":
    SCALING_FACTOR = torch.tensor([[0.0001, 0.1, 0.1, 0.1]])
    from pcn_continuous import _generate_demand_sequence

    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

    fixed_seq = _generate_demand_sequence()
    env = InventoryMOEnv(SCALING_FACTOR, "GBM", fixed_demand_seq=fixed_seq)
    reward_scale = calibrate_reward_scale_from_baseline("./MOPSO/MOPSO_GBM/MOPSO_run_GBM_1.csv")

    print("=== CAPQL VERBATIM (authors' own classes): GBM, total_steps=100000, start_steps=20000, alpha=0.2 ===")
    start = time.time()
    agent = train_capql_verbatim(env, total_steps=100_000, reward_scale=reward_scale, start_steps=20_000,
                                  alpha=0.2, device="cpu", diag_every=500)
    print(f"Training done in {time.time() - start:.1f}s")

    front = evaluate_front_verbatim(agent, env, n_eval_weights=10, n_episodes_per_weight=1)
    print(f"Non-dominated front: {len(front)} points")
    print(front)

    df = pd.DataFrame({"TC": -front[:, 0], "Frequency": -front[:, 1], "Amount": -front[:, 2]})
    out_path = "capql_verbatim_check_GBM.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} non-dominated points to {out_path}")
