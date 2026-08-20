import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _sub in ("baselines", "envs", "models", "utils", "experiments"):
    _p = _os.path.join(_ROOT, _sub)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
_os.chdir(_ROOT)

"""CAPQL baseline for the (Q, r) continuous-review inventory problem.

Algorithm: Concave-Augmented Pareto Q-Learning (CAPQL), from Lu, Herman & Yu,
"Multi-Objective Reinforcement Learning: Convexity, Stationarity and Pareto
Optimality," ICLR 2023. CAPQL extends SAC to a vector-reward, preference-
conditioned setting: twin Q-networks and a Gaussian policy additionally take a
preference vector w as input; the SAC entropy term (already part of the
standard soft-actor-critic target) plays the role of the paper's strongly
concave reward augmentation (Sec 4.1/5.2, Eq 8-10), which the paper shows is
what makes the preference-to-solution mapping g_s(w) continuous and hence
learnable. Reference implementation: https://github.com/haoyelu/CAPQL
(models/capql.py, models/networks.py, training_pipeline.py, utils/*) --
networks/agent/replay-buffer/weight-sampler logic below are ported from that
repo essentially as-is; only the environment wrapper and multi-setting/CSV
I/O layer are new (adapted from gpils_baseline.py's InventoryMOEnv pattern).

alpha (entropy augmentation strength) has no paper-given value for this
problem (the paper's own alpha values are per-MuJoCo-env, e.g. Hopper=0.2,
Walker2D=0.05, HalfCheetah=0.1, Humanoid=0.005) -- 0.1 is used as a starting
point (mid-range of the paper's reported values), pending a proper grid
search once the implementation is verified.

Run in the `pcn` conda env:

    conda activate pcn
    python capql_baseline.py
"""

import copy
import os
import random
import time

import gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam

from main_original import Inventory

# ---------------------------------------------------------------------------
# Networks (ported from CAPQL reference models/networks.py)
# ---------------------------------------------------------------------------
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPS = 1e-6


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, rwd_dim, hidden_dim):
        super().__init__()
        self.Q1 = nn.Sequential(
            nn.Linear(num_inputs + num_actions + rwd_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, rwd_dim),
        )
        self.Q2 = copy.deepcopy(self.Q1)
        self.rwd_dim = rwd_dim
        self.apply(weights_init_)

    def forward(self, state, action, w):
        xu = torch.cat([state, action, w], 1)
        return self.Q1(xu), self.Q2(xu)


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, rwd_dim, hidden_dim, action_space):
        super().__init__()
        self.linear1 = nn.Linear(num_inputs + rwd_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)
        self.apply(weights_init_)

        self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.)
        self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.)

    def forward(self, state, w):
        x = torch.cat((state, w), dim=1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = torch.clamp(self.log_std_linear(x), min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state, w):
        mean, log_std = self.forward(state, w)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + EPS)
        log_prob = log_prob.sum(1, keepdim=True).clamp(-1e3, 1e3)

        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)


def update_target(target, source, tau):
    for tp, p in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tp.data * (1.0 - tau) + p.data * tau)


class Weight_Sampler_pos:
    """Samples w on the positive simplex (w >= 0, sum(w) = 1), uniform-ish
    over directions (CAPQL reference utils/utils.py Weight_Sampler_pos)."""

    def __init__(self, rwd_dim):
        self.rwd_dim = rwd_dim

    def sample(self, n_sample):
        s = torch.abs(torch.normal(torch.zeros(n_sample, self.rwd_dim)))
        return s / torch.norm(s, dim=1, keepdim=True, p=1)


class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, w, reward, next_state, mask):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, w, reward, next_state, mask)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, w, reward, next_state, mask = map(np.stack, zip(*batch))
        return state, action, w, reward, next_state, mask

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# CAPQL agent (ported from CAPQL reference models/capql.py)
# ---------------------------------------------------------------------------
class CAPQL:
    def __init__(self, num_inputs, action_space, rwd_dim, gamma=0.99, tau=0.005,
                 lr=3e-4, alpha=0.1, hidden_size=256, target_update_interval=1,
                 device="cpu"):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.target_update_interval = target_update_interval
        self.device = torch.device(device)

        self.critic = QNetwork(num_inputs, action_space.shape[0], rwd_dim, hidden_size).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr)
        self.critic_target = copy.deepcopy(self.critic)

        self.policy = GaussianPolicy(num_inputs, action_space.shape[0], rwd_dim, hidden_size, action_space).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=lr)

    def select_action(self, state, w, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        w = torch.FloatTensor(w).to(self.device).unsqueeze(0)
        if not evaluate:
            action, _, _ = self.policy.sample(state, w)
        else:
            _, _, action = self.policy.sample(state, w)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates):
        state_batch, action_batch, w_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        w_batch = torch.FloatTensor(w_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_action, next_log_pi, _ = self.policy.sample(next_state_batch, w_batch)
            qf1_next, qf2_next = self.critic_target(next_state_batch, next_action, w_batch)
            min_qf_next = torch.min(qf1_next, qf2_next) - self.alpha * next_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * min_qf_next

        qf1, qf2 = self.critic(state_batch, action_batch, w_batch)
        qf_loss = F.mse_loss(qf1, next_q_value) + F.mse_loss(qf2, next_q_value)

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch, w_batch)
        qf1_pi, qf2_pi = self.critic(state_batch, pi, w_batch)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        min_qf_pi = (min_qf_pi * w_batch).sum(dim=-1, keepdim=True)
        policy_loss = (self.alpha * log_pi - min_qf_pi).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if updates % self.target_update_interval == 0:
            update_target(self.critic_target, self.critic, self.tau)

        return qf_loss.item(), policy_loss.item()


# ---------------------------------------------------------------------------
# Environment wrapper (adapted from gpils_baseline.py's InventoryMOEnv)
# ---------------------------------------------------------------------------
class InventoryMOEnv:
    def __init__(self, scaling_factor, demand_env, episode_len=365, fixed_demand_seq=None):
        self.demand_env = demand_env
        self.episode_len = episode_len
        self.sim = Inventory(scaling_factor, demand_env, fixed_demand_seq=fixed_demand_seq)
        self.action_space = self.sim.action_space
        self.rwd_dim = 3
        self._max_episode_steps = episode_len

    def reset(self):
        obs = self.sim.reset(self.demand_env, day=0)
        return np.asarray(obs, dtype=np.float32)

    def step(self, action):
        if self.demand_env == "gs":
            n_obs, reward, day, action_executed = self.sim.step_gs(action)
        else:
            n_obs, reward, day, action_executed = self.sim.step_original(action, self.demand_env)
        done = bool(day >= self.episode_len)
        return np.asarray(n_obs, dtype=np.float32), np.asarray(reward, dtype=np.float32), done, {"action_executed": bool(action_executed)}


def non_dominated_front(points):
    points = np.asarray(points)
    keep = np.ones(len(points), dtype=bool)
    for i, p in enumerate(points):
        for j, q in enumerate(points):
            if i != j and np.all(q >= p) and np.any(q > p):
                keep[i] = False
                break
    return points[keep]


# ---------------------------------------------------------------------------
# Reward scale calibration
# ---------------------------------------------------------------------------
class RunningMeanStd:
    """Welford-style running mean/variance tracker (same formula as OpenAI
    Baselines' RunningMeanStd), used to normalize each objective's reward
    scale from CAPQL's own collected experience only -- no other algorithm's
    output (e.g. an existing MOPSO baseline front) is used as a reference.

    var is initialized to 1 (not 0) and count to a small epsilon, so the
    scale starts near neutral (1/(1+eps) ~= 1) instead of exploding on the
    first few samples, and gets refined as more experience comes in.
    """

    def __init__(self, shape, epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x):
        delta = x - self.mean
        tot_count = self.count + 1

        new_mean = self.mean + delta / tot_count
        m_a = self.var * self.count
        M2 = m_a + np.square(delta) * self.count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    @property
    def std(self):
        return np.sqrt(self.var)


class ObsNormalizer:
    """Wraps an InventoryMOEnv-like env's reset()/step() to normalize
    observations with a RunningMeanStd computed only from the agent's own
    collected experience (same precedent as the reward-scale RunningMeanStd
    above -- no other algorithm's output is used as a reference).

    Motivation: raw Inventory observations (inventory level, on-order
    amounts) are on the same O(1000s) scale as the action space, unlike
    MuJoCo's typically small (O(1-10)) native observations that CAPQL's
    published hyperparameters (lr, init scale, etc.) were tuned against.
    Unnormalized, large inputs against xavier-initialized weights can push
    the policy's pre-tanh mean/log_std into saturation from the very start
    of training, independent of alpha/start_steps/lr.

    Stats keep updating on every reset()/step() call, including during
    evaluation -- after total_steps of training this changes the running
    mean/std negligibly, and eval must normalize with (approximately) the
    same statistics the policy was trained under.
    """

    def __init__(self, env, eps=1e-6):
        self.env = env
        self.eps = eps
        self.obs_rms = None  # sized lazily from the first observation

    def _normalize(self, obs):
        obs = np.asarray(obs, dtype=np.float64)
        if self.obs_rms is None:
            self.obs_rms = RunningMeanStd(shape=obs.shape)
        self.obs_rms.update(obs)
        return ((obs - self.obs_rms.mean) / (self.obs_rms.std + self.eps)).astype(np.float32)

    def reset(self):
        return self._normalize(self.env.reset())

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._normalize(obs), reward, done, info

    def __getattr__(self, name):
        return getattr(self.env, name)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def save_checkpoint(path, agent, total_numsteps, updates, i_episode, reward_rms, env,
                     gamma, tau, alpha, lr):
    """Saves everything needed to resume train_capql() later: both networks'
    weights, both optimizers' state, step/update counters, the reward-scale
    RunningMeanStd, and (if env is ObsNormalizer-wrapped) its obs_rms too --
    resuming without these would abruptly shift the reward/obs scale the
    policy was trained under. The replay buffer itself is NOT saved (would
    dominate checkpoint size); a resumed run starts with an empty buffer,
    which refills past batch_size within a few hundred steps."""
    ckpt = {
        "policy_state_dict": agent.policy.state_dict(),
        "critic_state_dict": agent.critic.state_dict(),
        "critic_target_state_dict": agent.critic_target.state_dict(),
        "policy_optim_state_dict": agent.policy_optim.state_dict(),
        "critic_optim_state_dict": agent.critic_optim.state_dict(),
        "total_numsteps": total_numsteps, "updates": updates, "i_episode": i_episode,
        "reward_rms_mean": reward_rms.mean, "reward_rms_var": reward_rms.var,
        "reward_rms_count": reward_rms.count,
        "gamma": gamma, "tau": tau, "alpha": alpha, "lr": lr,
    }
    obs_rms = getattr(env, "obs_rms", None)
    if obs_rms is not None:
        ckpt["obs_rms_mean"] = obs_rms.mean
        ckpt["obs_rms_var"] = obs_rms.var
        ckpt["obs_rms_count"] = obs_rms.count
    torch.save(ckpt, path)


def load_checkpoint(path, device):
    return torch.load(path, map_location=device)


def train_capql(env, total_steps, start_steps=10000, batch_size=256, gamma=0.99,
                 tau=0.005, lr=3e-4, alpha=0.1, replay_size=1_000_000,
                 updates_per_step=1, device="cpu", verbose=True, diag_every=200,
                 reward_scale_eps=1e-6, eval_every_steps=None, eval_n_weights=50,
                 eval_episodes_per_weight=1, resume_from=None, checkpoint_path=None):
    """eval_every_steps (optional, off by default): if set, runs evaluate_front()
    at the end of each episode that crosses a multiple of eval_every_steps, for
    building a training-progress/convergence curve. Diagnostic only -- these
    extra rollouts share the training env, so if it's wrapped in ObsNormalizer
    they DO feed its running obs mean/std (unlike the current single
    end-of-training evaluate_front() call convention), a minor difference from
    the final-eval-only setup used everywhere else in this project. When
    eval_every_steps is None (default), behavior/return value is unchanged
    from before.

    resume_from (optional path): loads a checkpoint saved by save_checkpoint()
    and continues training from its total_numsteps/updates instead of
    starting fresh (agent weights, optimizer state, reward_rms, and obs_rms
    if present are all restored; the replay buffer starts empty regardless).

    checkpoint_path (optional path): if set, saves a checkpoint (via
    save_checkpoint()) at every eval_every_steps checkpoint (when that's also
    set) and always once more at the very end of training, so a run can be
    resumed later even if interrupted mid-training."""
    num_inputs = env.reset().shape[0]
    agent = CAPQL(num_inputs, env.action_space, env.rwd_dim, gamma=gamma, tau=tau,
                  lr=lr, alpha=alpha, device=device)
    memory = ReplayMemory(replay_size, seed=0)
    sampler = Weight_Sampler_pos(env.rwd_dim)
    reward_rms = RunningMeanStd(shape=(env.rwd_dim,))

    total_numsteps = 0
    updates = 0
    i_episode = 0
    recent_actions = []
    recent_scaled_rewards = []
    last_qf_loss, last_policy_loss = None, None
    eval_history = []

    if resume_from is not None:
        ckpt = load_checkpoint(resume_from, device)
        agent.policy.load_state_dict(ckpt["policy_state_dict"])
        agent.critic.load_state_dict(ckpt["critic_state_dict"])
        agent.critic_target.load_state_dict(ckpt["critic_target_state_dict"])
        agent.policy_optim.load_state_dict(ckpt["policy_optim_state_dict"])
        agent.critic_optim.load_state_dict(ckpt["critic_optim_state_dict"])
        total_numsteps = ckpt["total_numsteps"]
        updates = ckpt["updates"]
        i_episode = ckpt.get("i_episode", 0)
        reward_rms.mean = ckpt["reward_rms_mean"]
        reward_rms.var = ckpt["reward_rms_var"]
        reward_rms.count = ckpt["reward_rms_count"]
        obs_rms = getattr(env, "obs_rms", None)
        if obs_rms is not None and "obs_rms_mean" in ckpt:
            obs_rms.mean = ckpt["obs_rms_mean"]
            obs_rms.var = ckpt["obs_rms_var"]
            obs_rms.count = ckpt["obs_rms_count"]
        if verbose:
            print(f"  resumed from {resume_from}: total_numsteps={total_numsteps}, updates={updates}")

    next_eval_step = eval_every_steps
    if eval_every_steps is not None and total_numsteps > 0:
        next_eval_step = ((total_numsteps // eval_every_steps) + 1) * eval_every_steps

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
                    qf_loss, policy_loss = agent.update_parameters(memory, batch_size, updates)
                    last_qf_loss, last_policy_loss = qf_loss, policy_loss
                    updates += 1
                    if verbose and updates % diag_every == 0:
                        act_arr = np.array(recent_actions[-diag_every:])
                        rew_arr = np.array(recent_scaled_rewards[-diag_every:]) if recent_scaled_rewards else np.zeros((1, env.rwd_dim))
                        print(f"    [update {updates}] qf_loss={qf_loss:.4g} policy_loss={policy_loss:.4g} "
                              f"action_mean={act_arr.mean(axis=0)} action_std={act_arr.std(axis=0)} "
                              f"scaled_reward_mean={rew_arr.mean(axis=0)} reward_scale={1.0 / (reward_rms.std + reward_scale_eps)}")

            next_state, reward, done, _ = env.step(action)
            reward_rms.update(reward)
            reward_scale = 1.0 / (reward_rms.std + reward_scale_eps)
            recent_scaled_rewards.append(reward * reward_scale)
            episode_steps += 1
            total_numsteps += 1

            mask = 1.0 if episode_steps == env._max_episode_steps else float(not done)
            memory.push(state, action, w.numpy(), reward * reward_scale, next_state, mask)
            state = next_state

        if eval_every_steps is not None and total_numsteps >= next_eval_step:
            eval_front = evaluate_front(agent, env, n_eval_weights=eval_n_weights,
                                         n_episodes_per_weight=eval_episodes_per_weight)
            eval_history.append({"step": total_numsteps, "front": eval_front.tolist()})
            if verbose:
                print(f"  [eval @ step {total_numsteps}] front_size={len(eval_front)}")
            if checkpoint_path is not None:
                save_checkpoint(checkpoint_path, agent, total_numsteps, updates, i_episode,
                                 reward_rms, env, gamma, tau, alpha, lr)
                if verbose:
                    print(f"  [checkpoint saved @ step {total_numsteps}] -> {checkpoint_path}")
            next_eval_step += eval_every_steps

        if verbose and i_episode % 20 == 0:
            print(f"  [episode {i_episode}] total_steps={total_numsteps}/{total_steps} updates={updates}")

    if checkpoint_path is not None:
        save_checkpoint(checkpoint_path, agent, total_numsteps, updates, i_episode,
                         reward_rms, env, gamma, tau, alpha, lr)
        if verbose:
            print(f"  [final checkpoint saved @ step {total_numsteps}] -> {checkpoint_path}")

    if eval_every_steps is not None:
        return agent, eval_history
    return agent


def evaluate_at_weights(agent, env, weights, n_episodes_per_weight=1):
    """Same rollout logic as evaluate_front, but for an explicit list of
    weight vectors instead of ones randomly sampled from Weight_Sampler_pos.

    Weight_Sampler_pos draws from abs(Normal) then L1-normalizes, so it
    essentially never lands exactly on a simplex corner or edge midpoint by
    chance -- this lets us check those specific points directly, to rule out
    "the random sample just missed the diversity" as an explanation for a
    collapsed front. Returns the raw per-weight vector returns (not filtered
    to the non-dominated subset) so each weight's result can be inspected
    individually.
    """
    vec_returns = []
    for w in weights:
        w = np.asarray(w, dtype=np.float32)
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
    return np.array(vec_returns)


def rollout_action_trajectory(agent, env, w, out_path=None):
    """Rolls out one evaluation episode at a fixed preference vector w and
    records the (Q, r) action only on days an order was actually executed --
    the CAPQL analogue of PCN's Episode==0 trajectory dump in
    pcn_continuous.py (train()'s
    "{version}_non_dominated_action_trajectories_{run_id}.csv", which itself
    only logs a row when `transition.action_executed` is True). Matching
    that convention here (rather than logging the queried action on every
    day regardless of whether an order was placed) keeps the two algorithms'
    trajectories comparable in the "representative action trajectories"
    figure (paper Figure 6): unlike MOPSO/MOPSO-AMS's single static (Q, r)
    applied every day, CAPQL's action is state-conditioned, so the sequence
    of executed orders still varies over the year.

    Column names ("Step", "Q (Order Size)", "k (Safety Stock)") match
    main.py's load_mopso_policy/find_column conventions so the resulting CSV
    is a drop-in source for plot_representative_action_trajectories().
    """
    w = np.asarray(w, dtype=np.float32)
    state = env.reset()
    done = False
    step = 0
    records = {"Step": [], "Q (Order Size)": [], "k (Safety Stock)": []}
    while not done:
        action = agent.select_action(state, w, evaluate=True)
        state, _, done, info = env.step(action)
        if info.get("action_executed", False):
            records["Step"].append(step)
            records["Q (Order Size)"].append(float(action[0]))
            records["k (Safety Stock)"].append(float(action[1]))
        step += 1

    df = pd.DataFrame(records)
    if out_path is not None:
        df.to_csv(out_path, index=False)
    return df


def evaluate_front(agent, env, n_eval_weights=50, n_episodes_per_weight=1):
    sampler = Weight_Sampler_pos(env.rwd_dim)
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
    # --- Smoke test: GBM, 1 run, small step budget ---
    import time as _time

    SCALING_FACTOR = torch.tensor([[0.0001, 0.1, 0.1, 0.1]])
    from pcn_continuous import _generate_demand_sequence

    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

    fixed_seq = _generate_demand_sequence()
    env = InventoryMOEnv(SCALING_FACTOR, "GBM", fixed_demand_seq=fixed_seq)

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== CAPQL smoke test: GBM, total_steps=50000, device={_device} ===")
    start = _time.time()
    agent = train_capql(env, total_steps=50_000, start_steps=2000, alpha=0.1, device=_device)
    print(f"Training done in {_time.time() - start:.1f}s")

    front = evaluate_front(agent, env, n_eval_weights=30, n_episodes_per_weight=1)
    print(f"Non-dominated front: {len(front)} points")
    df = pd.DataFrame({"TC": -front[:, 0], "Frequency": -front[:, 1], "Amount": -front[:, 2]})
    print(df)
    df.to_csv("capql_smoketest_GBM.csv", index=False)
