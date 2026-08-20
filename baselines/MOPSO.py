import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _sub in ("baselines", "envs", "models", "utils", "experiments"):
    _p = _os.path.join(_ROOT, _sub)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
_os.chdir(_ROOT)

import random
import numpy as np
from scipy.stats import truncnorm, norm
import matplotlib.pyplot as plt
import pandas as pd
import time
from main_original import Inventory
import os
import torch
from scipy.stats import qmc
import heapq

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class Particle:
    def __init__(self, position, dim, bounds):
        self.position = position.copy()
        self.fitness = np.array([np.inf, np.inf, np.inf])
        self.velocity = np.array([np.random.uniform(-bounds[i][1] / 300, bounds[i][1] / 300) for i in range(dim)]) # stationary : 300, seasonal : 300, GBM : 300, gs : 300
        self.best_position = self.position.copy()
        self.best_fitness = self.fitness.copy()

class MOPSO:
    def __init__(self, num_particles, dim, bounds, num_iterations, initial_positions=None, max_archive_size=None,
                 tsou_pbest_update=False, tsou_archive_removal=False, tsou_interleaved_loop=False):
        self.num_particles = num_particles
        self.dim = dim
        self.bounds = np.array(bounds)
        self.num_iterations = num_iterations
        self.archive = []
        self.w = 0.8 # usually 0.8 ~ 1.2 (stationary : 1.0?, seasonal : ?, GBM : 0.8?, gs : 1.0)
        self.c1 = 1.0 # local best : usually 1.0 ~ 4.0 (stationary : 1.0?, seasonal : ?, GBM : 1.0?, gs : 3.0)
        self.c2 = 1.0 # global best : usually 1.0 ~ 4.0 (stationary : 1.0?, seasonal : ?, GBM : 1.0?, gs : 1.0)
        # opt-in only: default None preserves the original unbounded archive
        # behavior (all existing baseline CSVs were generated this way).
        # Pass e.g. max_archive_size=num_particles to enable the adaptive-grid
        # style capacity+crowding-deletion mechanism from the original MOPSO
        # (Coello, Pulido & Lechuga 2004; inherited by Tsou 2008's inventory
        # MOPSO, which this project's baseline is descended from).
        self.max_archive_size = max_archive_size
        self.peak_archive_size = 0  # diagnostic: largest archive size reached at any point (pre-cap)
        # Tsou (2008) Table 1, lines 17-19: update pbest whenever the new
        # position weakly dominates pbest OR is incomparable to it (i.e.
        # keep the old pbest only when it weakly dominates the new
        # position). Default False preserves the original behavior: update
        # only on strict dominance, keep old pbest on ties/incomparable.
        self.tsou_pbest_update = tsou_pbest_update
        # Tsou (2008) Table 1 text: the archive REJECTS a new particle using
        # weak dominance (standard Pareto dominance), but only REMOVES
        # existing archive members that the new particle *strongly*
        # dominates (strict < in every objective) -- a narrower removal
        # criterion than weak dominance. Default False preserves the
        # original behavior (weak dominance used for both add and remove).
        self.tsou_archive_removal = tsou_archive_removal
        # Tsou (2008) Table 1, lines 04-19: a single per-particle loop where
        # leader selection (line 05) uses the archive as updated by
        # earlier particles processed in the *same* iteration (steady-state/
        # asynchronous update). Default False preserves the original
        # behavior (two separate loops per iteration; all particles in an
        # iteration move using the archive snapshot from the *previous*
        # iteration -- generational/synchronous update).
        self.tsou_interleaved_loop = tsou_interleaved_loop

        # 🟡 LHS로 초기 위치 샘플링
        '''initial_positions = lhs_initial_positions(num_particles, self.bounds)
        self.particles = [Particle(position=initial_positions[i], dim=self.dim, bounds=self.bounds) for i in range(num_particles)]
        self.fixed_mask = np.array([False] * dim)'''

        # random하게 initial position sampling
        initial_positions = np.array([np.random.uniform(low, high, size=num_particles) for (low, high) in self.bounds]).T
        #self.particles = [Particle(position=initial_positions[i], bounds = self.bounds) for i in range(num_particles)]
        self.particles = [Particle(position=initial_positions[i], dim=self.dim, bounds=self.bounds) for i in range(num_particles)]

    def update_archive(self, particle):
        '''function for updating archive'''

        dominated = [] # index of existing non-dominated solutions that dominated by new particle
        non_dominated = True # check if new particle is non-dominated or not
        removal_check = strongly_dominates if self.tsou_archive_removal else dominates
        for pos, fit in enumerate(self.archive):
            if dominates(fit[1], particle[1]): # check if existing non-dominated solution dominates new particle
                non_dominated = False
                break
            elif removal_check(particle[1], fit[1]): # check if new particle dominates extisting non-dominated solution
                dominated.append(pos)
        if non_dominated: # update archive if new particle is non-dominated
            self.archive = [self.archive[i] for i in range(len(self.archive)) if i not in dominated]
            self.archive.append(particle)
            if len(self.archive) > self.peak_archive_size:
                self.peak_archive_size = len(self.archive)
            if self.max_archive_size is not None:
                self._prune_archive()

    def _prune_archive(self):
        '''repeatedly remove the most crowded archive member (smallest
        nearest-neighbor distance in normalized objective space) until the
        archive is back down to max_archive_size -- a simplified proxy for
        the adaptive-grid crowding deletion in the original MOPSO (Coello,
        Pulido & Lechuga 2004).'''
        while len(self.archive) > self.max_archive_size:
            fitness = np.array([fit for _, fit in self.archive])
            lo, hi = fitness.min(axis=0), fitness.max(axis=0)
            denom = np.where(hi - lo < 1e-12, 1.0, hi - lo)
            norm_fitness = (fitness - lo) / denom

            dmat = np.linalg.norm(norm_fitness[:, None, :] - norm_fitness[None, :, :], axis=-1)
            np.fill_diagonal(dmat, np.inf)
            nearest_dist = dmat.min(axis=1)
            to_remove = int(np.argmin(nearest_dist))

            self.archive = [self.archive[i] for i in range(len(self.archive)) if i != to_remove]

    def _should_update_pbest(self, particle):
        if self.tsou_pbest_update:
            # Tsou (2008): update unless old pbest weakly dominates the new position.
            return not dominates(particle.best_fitness, particle.fitness)
        return dominates(particle.fitness, particle.best_fitness)

    def optimize_seasonal(self, fixed_mask, demand_env, period_idx, env):
        """
        fixed_mask: bool array of length self.dim
          - True  → 해당 차원은 고정
          - False → 해당 차원은 업데이트 대상

        period_idx: 0 ~ 3 → 어떤 period의 (Q, r) pair를 업데이트하는지
        """
        fixed_mask = np.array(fixed_mask)

        for _ in range(self.num_iterations):
            leaders = [pos[0] for pos in self.archive]

            for particle in self.particles:
                total_cost, stockout_freq, stockout_amt = evaluate_solution_seasonal(particle.position, demand_env, env=env, period_idx=period_idx)

                particle.fitness = np.array([total_cost, stockout_freq, stockout_amt])
                self.update_archive([particle.position.copy(), particle.fitness.copy()])

                if self._should_update_pbest(particle):
                    particle.best_position = particle.position.copy()
                    particle.best_fitness = particle.fitness.copy()

            for particle in self.particles:
                leader = random.choice(leaders) if leaders else particle.best_position
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocity_candidate = (self.w * particle.velocity + self.c1 * r1 * (particle.best_position - particle.position) + self.c2 * r2 * (leader - particle.position))

                # 🔒 고정 차원은 velocity = 0
                velocity_candidate = np.where(fixed_mask, 0.0, velocity_candidate)
                particle.velocity = velocity_candidate

                # 📍 고정 차원은 position도 그대로 유지
                new_position = particle.position + particle.velocity
                new_position = np.clip(new_position, self.bounds[:, 0], self.bounds[:, 1])
                particle.position = np.where(fixed_mask, particle.position, new_position)

        return self.archive

    def optimize_variation(self, env, demand_env):
        '''function for optimizing the particles (variation)'''

        for _ in range(self.num_iterations):  # repeat updates for number of iterations
            if self.tsou_interleaved_loop:
                self._step_interleaved(env, demand_env)
            else:
                self._step_generational(env, demand_env)

        # After all iterations, generate the top N solutions
        # by applying the nlargest logic to all particles
            # 모든 반복이 끝난 후, 상위 n개의 솔루션을 선택하여 반환
        top_n_solutions = get_top_n_solutions(self.particles, n=100)

        # archive와 상위 n개의 솔루션을 함께 반환
        return self.archive, top_n_solutions, [[p.position.copy(), p.fitness.copy()] for p in self.particles]

    def _step_generational(self, env, demand_env):
        '''Original two-pass update: all particles evaluate against the
        archive snapshot from the *previous* iteration, then all particles
        move using that same (now stale) snapshot.'''
        leaders = [pos[0] for pos in self.archive]

        for particle in self.particles:
            total_cost, stockout_frequency, stockout_amount = evaluate_solution_original(particle.position[0], particle.position[1], env, demand_env)
            particle.fitness = np.array([total_cost, stockout_frequency, stockout_amount])
            self.update_archive([particle.position.copy(), particle.fitness.copy()])

            if self._should_update_pbest(particle):
                particle.best_position = particle.position.copy()
                particle.best_fitness = particle.fitness.copy()

        for particle in self.particles:
            leader = random.choice(leaders) if leaders else particle.best_position
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            particle.velocity = (self.w * particle.velocity + self.c1 * r1 * (particle.best_position - particle.position) + self.c2 * r2 * (leader - particle.position))
            particle.velocity = np.clip(particle.velocity, -self.bounds[0][1] / 300, self.bounds[0][1] / 300)
            particle.position += particle.velocity
            particle.position = np.clip(particle.position, self.bounds[0][0], self.bounds[0][1])

    def _step_interleaved(self, env, demand_env):
        '''Tsou (2008) Table 1, lines 04-19: single per-particle pass --
        pick leader from the archive as updated by earlier particles in
        this same iteration, move, evaluate the new position, then update
        the archive and pbest, before moving on to the next particle.'''
        for particle in self.particles:
            leaders = [pos[0] for pos in self.archive]
            leader = random.choice(leaders) if leaders else particle.best_position
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            particle.velocity = (self.w * particle.velocity + self.c1 * r1 * (particle.best_position - particle.position) + self.c2 * r2 * (leader - particle.position))
            particle.velocity = np.clip(particle.velocity, -self.bounds[0][1] / 300, self.bounds[0][1] / 300)
            particle.position += particle.velocity
            particle.position = np.clip(particle.position, self.bounds[0][0], self.bounds[0][1])

            total_cost, stockout_frequency, stockout_amount = evaluate_solution_original(particle.position[0], particle.position[1], env, demand_env)
            particle.fitness = np.array([total_cost, stockout_frequency, stockout_amount])
            self.update_archive([particle.position.copy(), particle.fitness.copy()])

            if self._should_update_pbest(particle):
                particle.best_position = particle.position.copy()
                particle.best_fitness = particle.fitness.copy()

def lhs_initial_positions(n_particles, bounds):
    dim = bounds.shape[0]
    sampler = qmc.LatinHypercube(d=dim)
    sample = sampler.random(n=n_particles)
    scaled_sample = qmc.scale(sample, bounds[:, 0], bounds[:, 1])  # bounds 내로 스케일링

    return scaled_sample  # shape: (n_particles, dim)

def evaluate_solution_seasonal(solution_vector, demand_env, env=None, period_idx=None):
    """
    solution_vector: shape (8,)
    env: 이미 생성된 환경 인스턴스 (Inventory)
    period_idx: 0 ~ 3 or None
    """
    Qs = [solution_vector[i] for i in range(0, 6, 2)]
    rs = [solution_vector[i] for i in range(1, 6, 2)]

    total_reward = np.array([0.0, 0.0, 0.0])
    env.reset(demand_env)

    if period_idx is None:
        end_day = 365
    else:
        end_day = (period_idx + 1) * 120 if period_idx < 2 else 365

    while env.day <= end_day:
        if period_idx is None:
            period = (env.day - 1) // 90 if env.day > 0 else 0
        else:
            period = period_idx

        current_Q = Qs[period]
        current_r = rs[period]

        _, reward, _, _ = env.step_original([current_Q, current_r], demand_env)
        total_reward += reward

    return -1 * total_reward[0], -1 * total_reward[1], -1 * total_reward[2]

def evaluate_solution_original(Q, k, env, demand_env, period=365):
    '''function for evaluating fitness of solution (stationary demand)'''
    #stationary, GBM, gs

    total_reward = np.array([0.0, 0.0, 0.0])
    day = 0
    env.reset(demand_env)
    while day <= period:
        action = [Q, k]
        if demand_env == 'gs':
            n_obs, reward, day, last_order_day = env.step_gs(action)
        else:
            n_obs, reward, day, last_order_day = env.step_original(action, demand_env)
        total_reward += reward

    return -1 * total_reward[0], -1 * total_reward[1], -1 * total_reward[2]  # 원래 스케일로 복구

def dominates(p1, p2, epilson = 0.0):
    '''function for checking dominance'''

    return np.all(p1 <= p2 + epilson) and np.any(p1 < p2 - epilson)

def strongly_dominates(p1, p2):
    '''Tsou (2008)'s "strongly dominate": strict < in every objective.'''
    return bool(np.all(p1 < p2))

def get_top_n_solutions(particles, n=100, percentile=10):
    '''
    Selects the top n solutions from a list of particles based on multi-objective ranking.
    The ranking considers non-domination and crowding distance.

    Args:
        particles (list): A list of particle objects.
        n (int): The number of top solutions to return.
        threshold (float): A crowding distance threshold for penalizing crowded solutions.

    Returns:
        list: A list of the top n solutions, in the same format as the input particles.
    '''
    # Extract fitness from all particles
    returns = np.array([p.fitness for p in particles])

    # 1. Calculate crowding distance for all points
    # (crowding_distance and get_non_dominated functions must be defined and accessible)
    distances = crowding_distance(returns)
    threshold = np.percentile(distances, percentile)
    sma = np.argwhere(distances <= threshold).flatten()

    # Identify solutions that are too close together (crowding distance <= threshold)
    sma_indices = np.argwhere(distances <= threshold).flatten()

    # 2. Identify non-dominated solutions
    nd_indices = get_non_dominated(returns)
    nd_solutions = returns[nd_indices]

    # 3. Compute ranking score (l2)
    # Distance to the closest non-dominated point
    # We first expand returns to match the dimensions of nd_solutions for vectorized calculation
    returns_exp = np.tile(np.expand_dims(returns, 1), (1, len(nd_solutions), 1))
    l2 = np.min(np.linalg.norm(returns_exp - nd_solutions, axis=-1), axis=-1) * -1
    l2_min, l2_max = l2.min(), l2.max()
    if l2_max - l2_min != 0:
        l2 = (l2 - l2_min) / (l2_max - l2_min)
    # Apply penalties
    # Penalty for duplicate solutions
    non_dominated_unique_indices = np.nonzero(nd_indices)[0]
    _, unique_indices = np.unique(nd_solutions, axis=0, return_index=True)
    unique_indices = non_dominated_unique_indices[unique_indices]

    duplicates = np.ones(len(l2), dtype=bool)
    duplicates[unique_indices] = False
    '''l2[duplicates] -= 1e-5
    # Penalty for solutions with low crowding distance
    l2[sma_indices] *= 2'''

    l2[duplicates] += (l2.max() - l2.min()) * 0.01  # 점수 범위의 1%만큼 페널티 부여
    #l2[sma] += (l2.max() - l2.min()) * 0.5  # 점수 범위의 50%만큼 페널티 부여 (매우 강한 페널티)
    l2[sma] *= 2

    # 4. Sort and select top n
    sorted_indices = np.argsort(l2)
    # 파티클 객체 대신 [position, fitness] 형식의 리스트로 반환
    top_n_solutions = [[particles[i].position, particles[i].fitness] for i in sorted_indices[:n]]

    return top_n_solutions

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

if __name__ == "__main__":
    pass

    