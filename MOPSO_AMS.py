"""MOPSO-AMS: multi-swarm MOPSO with an adaptive multiple selection strategy.

Implements Han, Zhang, Yinga & Qiao (2023), "Adaptive multiple selection
strategy for multi-objective particle swarm optimization", Information
Sciences 624:235-251, adapted to this project's 2D (Q, r) continuous
action space and 3-objective (TC, Frequency, Amount) inventory problem.

Reused from MOPSO.py (identical, unmodified): dominates(), get_non_dominated(),
get_top_n_solutions(), crowding_distance(), evaluate_solution_original(),
evaluate_solution_seasonal().

Adaptation notes (paper is ambiguous or benchmark-specific on these points,
so a concrete choice was made -- see paper Sections 4.1-4.3):
  - Direction vectors U (Eq. 9-11): the paper does not fix K for ZDT/DTLZ
    benchmarks. Uses a Das-Dennis simplex-lattice with p=4 subdivisions for
    M=3 objectives (K=15 directions), the standard NSGA-III-style choice.
    Objective vectors are min-max normalized against the *current archive's*
    range before computing direction cosines / cosine similarity to u_k
    (equivalent to the paper's arccos-based angle, but robust to TC/Freq/
    Amount having wildly different raw scales).
  - sigma_h(t) (Eq. 12): interpreted as the count of non-dominated solutions
    assigned to the h-th (non-empty) sub-space, matching Remark 1's
    description ("small value -> better distribution").
  - Hierarchical clustering into leader groups A_n(t): scipy's agglomerative
    clustering (ward linkage) into N=num_subswarms clusters of the archive.
  - Ideal-state leader (Eq. 18): the paper says "the non-dominated solutions
    in A_n(t)" without specifying which one; the candidate in each cluster
    maximizing (normalized sparsity + normalized dominance rank) is used,
    consistent with Remark 3's "candidates with good diversity and
    convergence".
  - Eq. 23's inner sum index "N" is treated as the sub-swarm size I (a
    notational slip in the paper -- summing over the whole population N
    would double the same quantity for every sub-swarm and contradicts the
    per-sub-swarm crowding intent described in the text).
  - w/c1/c2 are clipped to [0.1, 2.0] / [0.1, 4.0] / [0.1, 4.0] since the
    multiplicative Eq. 24-26 update has no stated bound and could otherwise
    drift unboundedly over 100 iterations.
  - External archive size Q (Algorithm 1 step 5, "Delete the crowded
    solutions in the archive S(t)"): defaults to Q = I (population size),
    matching the paper's own "I and Q are set to..." convention. When the
    archive exceeds Q, the most crowded direction-vector sub-space (reusing
    the Eq. 9-11 assignment) is identified and its most crowded member
    (smallest nearest-neighbor distance within that sub-space) is removed,
    repeated until |S(t)| == Q.

Run in the `pcn` or `gpils` conda env (only needs numpy/scipy/pandas/torch):

    conda activate pcn
    python MOPSO_AMS.py
"""

import random
import itertools

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster

from MOPSO import (
    dominates,
    get_non_dominated,
    get_top_n_solutions,
    evaluate_solution_original,
    evaluate_solution_seasonal,
)

EPS = 1e-12


def das_dennis(p, M):
    """Simplex-lattice reference directions with p divisions in M dimensions."""

    def _gen(remaining, dims_left):
        if dims_left == 1:
            return [[remaining]]
        pts = []
        for i in range(remaining + 1):
            for rest in _gen(remaining - i, dims_left - 1):
                pts.append([i] + rest)
        return pts

    pts = np.array(_gen(p, M), dtype=float) / p
    return pts


class ParticleAMS:
    def __init__(self, position, dim, bounds, w, c1, c2):
        self.position = position.copy()
        self.fitness = np.array([np.inf, np.inf, np.inf])
        self.velocity = np.array([np.random.uniform(-bounds[i][1] / 300, bounds[i][1] / 300) for i in range(dim)])
        self.best_position = self.position.copy()
        self.best_fitness = self.fitness.copy()
        self.w = w
        self.c1 = c1
        self.c2 = c2


class MOPSO_AMS:
    def __init__(self, num_particles, dim, bounds, num_iterations, num_subswarms=4,
                 w0=0.8, c1_0=1.0, c2_0=1.0, das_dennis_p=4, M=3, max_archive_size=None):
        self.dim = dim
        self.bounds = np.array(bounds)
        self.num_iterations = num_iterations
        self.num_subswarms = num_subswarms
        self.subswarm_size = max(1, num_particles // num_subswarms)
        self.M = M
        self.directions = das_dennis(das_dennis_p, M)  # (K, M)
        # paper: "the size of external archive is Q" with Q = I (population size)
        self.max_archive_size = num_particles if max_archive_size is None else max_archive_size

        initial_positions = np.array(
            [np.random.uniform(low, high, size=num_particles) for (low, high) in self.bounds]
        ).T
        self.subswarms = []
        idx = 0
        for _ in range(num_subswarms):
            sw = [
                ParticleAMS(initial_positions[idx + i], dim, self.bounds, w0, c1_0, c2_0)
                for i in range(self.subswarm_size)
            ]
            self.subswarms.append(sw)
            idx += self.subswarm_size

        self.archive = []  # list of [position, fitness]
        self.prev_archive_fitness = None
        self.prev_IA = None
        self.state_history = []  # diagnostic: (state, IA, IB, archive_size) per iteration
        self.peak_archive_size = 0  # diagnostic: largest archive size reached at any point (pre-cap)

    def all_particles(self):
        return [p for sw in self.subswarms for p in sw]

    def update_archive(self, particle):
        dominated = []
        non_dominated = True
        for pos, fit in enumerate(self.archive):
            if dominates(fit[1], particle[1]):
                non_dominated = False
                break
            elif dominates(particle[1], fit[1]):
                dominated.append(pos)
        if non_dominated:
            self.archive = [self.archive[i] for i in range(len(self.archive)) if i not in dominated]
            self.archive.append(particle)
            if len(self.archive) > self.peak_archive_size:
                self.peak_archive_size = len(self.archive)
            self._prune_archive()

    def _normalize(self, F, ref_F):
        lo, hi = ref_F.min(axis=0), ref_F.max(axis=0)
        denom = np.where(hi - lo < EPS, 1.0, hi - lo)
        return (F - lo) / denom

    def _assign_to_subspace(self, fitness_array, ref_fitness):
        norm_F = self._normalize(fitness_array, ref_fitness)
        norms = np.linalg.norm(norm_F, axis=1, keepdims=True)
        norms = np.where(norms < EPS, EPS, norms)
        unit_F = norm_F / norms
        dir_norms = np.linalg.norm(self.directions, axis=1, keepdims=True)
        unit_dirs = self.directions / np.where(dir_norms < EPS, EPS, dir_norms)
        cos_sim = unit_F @ unit_dirs.T
        return np.argmax(cos_sim, axis=1)

    def _prune_archive(self):
        """Algorithm 1 step 5: 'Delete the crowded solutions in the archive
        S(t)'. When |S(t)| exceeds Q, repeatedly find the most crowded
        sub-space (direction-vector bin with the most archive members) and
        remove the member with the smallest nearest-neighbor distance within
        that sub-space, until |S(t)| == Q."""
        while len(self.archive) > self.max_archive_size:
            archive_fitness = np.array([fit for _, fit in self.archive])
            assignment = self._assign_to_subspace(archive_fitness, archive_fitness)
            unique, counts = np.unique(assignment, return_counts=True)
            most_crowded_subspace = unique[np.argmax(counts)]
            members = np.where(assignment == most_crowded_subspace)[0]

            if len(members) > 1:
                norm_F = self._normalize(archive_fitness, archive_fitness)
                sub_F = norm_F[members]
                dmat = np.linalg.norm(sub_F[:, None, :] - sub_F[None, :, :], axis=-1)
                np.fill_diagonal(dmat, np.inf)
                nearest_dist = dmat.min(axis=1)
                to_remove = members[np.argmin(nearest_dist)]
            else:
                to_remove = members[0]

            self.archive = [self.archive[i] for i in range(len(self.archive)) if i != to_remove]

    def _detect_state(self, archive_fitness):
        norm_F = self._normalize(archive_fitness, archive_fitness)
        norms = np.linalg.norm(norm_F, axis=1, keepdims=True)
        norms = np.where(norms < EPS, EPS, norms)
        unit_F = norm_F / norms

        dir_norms = np.linalg.norm(self.directions, axis=1, keepdims=True)
        unit_dirs = self.directions / np.where(dir_norms < EPS, EPS, dir_norms)

        cos_sim = unit_F @ unit_dirs.T  # (n_archive, K)
        assignment = np.argmax(cos_sim, axis=1)

        counts = []
        for k in range(len(self.directions)):
            c = np.sum(assignment == k)
            if c > 0:
                counts.append(c)
        counts = np.array(counts, dtype=float)
        IA = float(np.std(counts)) if len(counts) > 0 else 0.0

        if self.prev_archive_fitness is None:
            IB = 0
            state = "ideal"  # bootstrap: treat first iteration like the ideal-state selection
            return {"IA": IA, "IB": IB, "state": state, "assignment": assignment}

        IB = 0
        for fq in archive_fitness:
            if any(dominates(fq, fp) for fp in self.prev_archive_fitness):
                IB += 1

        prev_IA = self.prev_IA if self.prev_IA is not None else IA
        if IA < prev_IA and IB > 0:
            state = "ideal"
        elif IA >= prev_IA and IB > 0:
            state = "convergence"
        elif IA < prev_IA and IB == 0:
            state = "diversity"
        else:
            state = "stagnation"

        return {"IA": IA, "IB": IB, "state": state, "assignment": assignment}

    def _select_leaders(self, state_info, all_fitness):
        archive_positions = np.array([pos for pos, _ in self.archive])
        archive_fitness = np.array([fit for _, fit in self.archive])
        n_archive = len(self.archive)
        N = self.num_subswarms

        if n_archive == 0:
            # degenerate startup edge case: no non-dominated point yet
            return [p.best_position.copy() for sw in self.subswarms for p in sw[:1]][:N] or \
                   [self.subswarms[0][0].position.copy()] * N

        norm_F = self._normalize(archive_fitness, archive_fitness)

        if n_archive >= N and n_archive >= 2:
            Z = linkage(norm_F, method="ward")
            cluster_ids = fcluster(Z, t=N, criterion="maxclust")
        else:
            cluster_ids = np.arange(1, n_archive + 1)  # each point its own cluster

        unique_clusters = sorted(set(cluster_ids))

        leaders = []
        for n in range(N):
            cid = unique_clusters[n % len(unique_clusters)]
            members = np.where(cluster_ids == cid)[0]

            # sparse degree d(a): distance to nearest OTHER member in this cluster
            if len(members) > 1:
                sub_F = norm_F[members]
                dmat = np.linalg.norm(sub_F[:, None, :] - sub_F[None, :, :], axis=-1)
                np.fill_diagonal(dmat, np.inf)
                d_vals = dmat.min(axis=1)
            else:
                d_vals = np.array([1.0])  # sole member: automatically the sparsest

            # dominant rank R(a): number of ALL current particles dominated by a
            rank_vals = np.array([
                sum(1 for f in all_fitness if dominates(archive_fitness[m], f))
                for m in members
            ], dtype=float)

            d_range = d_vals.max() - d_vals.min()
            d_norm = (d_vals - d_vals.min()) / d_range if d_range > EPS else np.zeros_like(d_vals)
            r_range = rank_vals.max() - rank_vals.min()
            r_norm = (rank_vals - rank_vals.min()) / r_range if r_range > EPS else np.zeros_like(rank_vals)

            d_pick = members[np.argmax(d_vals)]      # diversity candidate d_n(t)
            c_pick = members[np.argmax(rank_vals)]   # convergence candidate c_n(t)
            combined_pick = members[np.argmax(d_norm + r_norm)]  # for ideal state

            state = state_info["state"]
            if state == "ideal":
                pick = combined_pick
            elif state == "diversity":
                pick = c_pick  # good diversity already -> push convergence
            elif state == "convergence":
                pick = d_pick  # good convergence already -> push diversity
            else:  # stagnation
                pick = random.choice([c_pick, d_pick])

            leaders.append(archive_positions[pick].copy())

        return leaders

    def _update_parameters(self, old_pbest_fitness_by_particle):
        for sw in self.subswarms:
            fitness_mat = np.array([p.fitness for p in sw])
            norm_fit = self._normalize(fitness_mat, fitness_mat)
            n = len(sw)
            MD = np.zeros(n)
            for i in range(n):
                diffs = np.sum(np.abs(norm_fit - norm_fit[i]), axis=1)
                diffs[i] = 0.0
                MD[i] = diffs.sum() / max(1, n - 1)
            MD_avg = MD.mean()

            for i, p in enumerate(sw):
                F = 1.0 / (1.0 + np.exp(-(MD[i] - MD_avg)))
                old_pbest = old_pbest_fitness_by_particle[id(p)]

                if dominates(old_pbest, p.fitness):
                    # pbest(t-1) still dominates x(t): poor convergence -> explore more
                    p.w = p.w * (1 + F)
                    p.c1 = p.c1 * (1 + F)
                    p.c2 = p.c2 * F
                elif dominates(p.fitness, old_pbest):
                    # x(t) now dominates pbest(t-1): good progress -> exploit more
                    p.w = p.w * F
                    p.c1 = p.c1 * F
                    p.c2 = p.c2 * (1 + F)
                # otherwise (incomparable): unchanged

                p.w = float(np.clip(p.w, 0.1, 2.0))
                p.c1 = float(np.clip(p.c1, 0.1, 4.0))
                p.c2 = float(np.clip(p.c2, 0.1, 4.0))

    def optimize_variation(self, env, demand_env, eval_fn=None):
        """eval_fn(position) -> (total_cost, stockout_frequency, stockout_amount).
        Defaults to evaluate_solution_original(position[0], position[1], env, demand_env)
        (the stationary/GBM/gs case); pass a custom eval_fn for other decompositions
        (e.g. seasonal's period-0-only evaluation)."""
        if eval_fn is None:
            eval_fn = lambda position: evaluate_solution_original(position[0], position[1], env, demand_env)

        all_particles = self.all_particles()

        for _ in range(self.num_iterations):
            old_pbest_fitness_by_particle = {id(p): p.best_fitness.copy() for p in all_particles}

            for p in all_particles:
                total_cost, stockout_frequency, stockout_amount = eval_fn(p.position)
                p.fitness = np.array([total_cost, stockout_frequency, stockout_amount])
                self.update_archive([p.position.copy(), p.fitness.copy()])
                if dominates(p.fitness, p.best_fitness):
                    p.best_position = p.position.copy()
                    p.best_fitness = p.fitness.copy()

            archive_fitness = np.array([fit for _, fit in self.archive])
            state_info = self._detect_state(archive_fitness)

            all_fitness = [p.fitness for p in all_particles]
            leaders = self._select_leaders(state_info, all_fitness)

            self._update_parameters(old_pbest_fitness_by_particle)

            for n, sw in enumerate(self.subswarms):
                leader = leaders[n]
                for p in sw:
                    r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                    p.velocity = (
                        p.w * p.velocity
                        + p.c1 * r1 * (p.best_position - p.position)
                        + p.c2 * r2 * (leader - p.position)
                    )
                    p.velocity = np.clip(p.velocity, -self.bounds[0][1] / 300, self.bounds[0][1] / 300)
                    p.position = p.position + p.velocity
                    p.position = np.clip(p.position, self.bounds[0][0], self.bounds[0][1])

            self.prev_archive_fitness = archive_fitness
            self.prev_IA = state_info["IA"]
            self.state_history.append((state_info["state"], state_info["IA"], state_info["IB"], len(self.archive)))

        top_n_solutions = get_top_n_solutions(all_particles, n=100)
        return self.archive, top_n_solutions, [[p.position.copy(), p.fitness.copy()] for p in all_particles]


if __name__ == "__main__":
    pass
