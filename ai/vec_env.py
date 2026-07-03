"""VecSnakeEnv — N SnakeEnv instances stepped together (AI_PLAN.md §8; the
"vectorized environments" tier discussed for GPU utilization).

Batching N environments means each policy forward pass during collection is one
(N, C, H, W) tensor instead of N single-sample calls, which is what lets a GPU
actually be busy. Envs auto-reset on termination so collection never stalls.

This is still a single process (not a decoupled actor-learner) — the learner and
the actors share one thread — but the batched-synchronous form is the high-ROI
step before a full distributed split.
"""

from __future__ import annotations

import numpy as np

from .env import SnakeEnv
from .features import grid_encode


class VecSnakeEnv:
    def __init__(self, num_envs, size, seed=0, **env_kwargs):
        self.num_envs = num_envs
        self.size = size
        self.envs = [SnakeEnv(size=size, seed=seed + i, **env_kwargs) for i in range(num_envs)]

    def reset(self):
        return self._stack([e.reset() for e in self.envs])

    def step(self, actions):
        """actions: array of length num_envs. Returns:
        obs (N,C,H,W), rewards (N,), dones (N,), scores (N,) where dones[i] marks
        an episode that ended this step (its score is in scores[i]); the env is
        auto-reset so the returned obs[i] is the fresh episode's first frame."""
        n = self.num_envs
        rewards = np.zeros(n, dtype=np.float32)
        dones = np.zeros(n, dtype=np.float32)
        scores = np.full(n, -1, dtype=np.int64)
        views = []
        for i, (env, a) in enumerate(zip(self.envs, actions)):
            view, r, done, info = env.step(int(a))
            rewards[i] = r
            if done:
                dones[i] = 1.0
                scores[i] = info["score"]
                view = env.reset()  # auto-reset for continuous collection
            views.append(view)
        return self._stack(views), rewards, dones, scores

    def _stack(self, views):
        return np.stack([grid_encode(v) for v in views], axis=0)
