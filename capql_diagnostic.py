"""Short diagnostic run for CAPQL: total_steps=5000 (vs the 50k smoke test),
with loss/action/reward logging every 200 updates, to see whether training is
diverging (NaN/exploding loss, actions pinned to a boundary) or just slowly
converging to a mediocre point, before committing to another full 50k run.

Run in the `pcn` conda env:

    conda activate pcn
    python capql_diagnostic.py
"""

import random
import time

import numpy as np
import torch

from capql_baseline import (
    InventoryMOEnv, calibrate_reward_scale_from_baseline, train_capql,
    evaluate_front,
)

SCALING_FACTOR = torch.tensor([[0.0001, 0.1, 0.1, 0.1]])

if __name__ == "__main__":
    from pcn_continuous import _generate_demand_sequence

    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

    fixed_seq = _generate_demand_sequence()
    env = InventoryMOEnv(SCALING_FACTOR, "GBM", fixed_demand_seq=fixed_seq)
    reward_scale = calibrate_reward_scale_from_baseline("./MOPSO/MOPSO_GBM/MOPSO_run_GBM_1.csv")

    print(f"=== CAPQL diagnostic: GBM, total_steps=100000, start_steps=20000 ===")
    start = time.time()
    agent = train_capql(env, total_steps=100000, reward_scale=reward_scale, start_steps=20000,
                         alpha=0.1, device="cpu", diag_every=500)
    print(f"Training done in {time.time() - start:.1f}s")

    front = evaluate_front(agent, env, n_eval_weights=10, n_episodes_per_weight=1)
    print(f"Non-dominated front: {len(front)} points")
    print(front)

    # --- Probe: does the critic overestimate action=(0,0) vs a known-good MOPSO action? ---
    print("\n=== Critic Q-value probe: (0,0) vs MOPSO-optimal action ===")
    import torch as _torch
    state0 = env.reset()
    mopso_action = np.array([417.763058, 80.539952], dtype=np.float32)  # MOPSO's lowest-TC GBM solution
    zero_action = np.array([0.0, 0.0], dtype=np.float32)
    test_weights = [np.array([1., 0., 0.]), np.array([0., 1., 0.]), np.array([0., 0., 1.]), np.array([1/3, 1/3, 1/3])]

    with _torch.no_grad():
        s_t = _torch.FloatTensor(state0).unsqueeze(0)
        for w in test_weights:
            w_t = _torch.FloatTensor(w).unsqueeze(0)
            a_mopso = _torch.FloatTensor(mopso_action).unsqueeze(0)
            a_zero = _torch.FloatTensor(zero_action).unsqueeze(0)
            q1_mopso, q2_mopso = agent.critic(s_t, a_mopso, w_t)
            q1_zero, q2_zero = agent.critic(s_t, a_zero, w_t)
            print(f"  w={w}: Q(MOPSO action)={q1_mopso.numpy()[0]}  Q(action=0,0)={q1_zero.numpy()[0]}  "
                  f"w^T Q(MOPSO)={(w_t*q1_mopso).sum().item():.4f}  w^T Q(zero)={(w_t*q1_zero).sum().item():.4f}")
