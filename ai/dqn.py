"""A small MLP + DQN, hand-written in numpy (AI_PLAN.md §2, §7).

This is the "simple option" milestone: a feature-vector MLP trained with
Q-learning — literally tabular Q-learning from cliff-walking with a network in
place of the table. No torch dependency; everything is explicit numpy so the
learning is transparent. The CNN/PPO upgrades (AI_PLAN.md §2) would move to
PyTorch.
"""

from __future__ import annotations

import json

import numpy as np


# --------------------------------------------------------------------------
# MLP with explicit forward/backward (ReLU hidden, linear output).
# --------------------------------------------------------------------------
class MLP:
    def __init__(self, sizes, rng):
        self.sizes = list(sizes)
        self.W, self.b = [], []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            # He initialization for ReLU.
            self.W.append(rng.normal(0, np.sqrt(2.0 / fan_in), (fan_in, fan_out)).astype(np.float32))
            self.b.append(np.zeros(fan_out, dtype=np.float32))
        self.n = len(self.W)

    def forward(self, x):
        self.cache = []
        a = x
        for i in range(self.n):
            z = a @ self.W[i] + self.b[i]
            self.cache.append((a, z))
            a = np.maximum(z, 0) if i < self.n - 1 else z
        return a

    def backward(self, dout):
        gW = [None] * self.n
        gb = [None] * self.n
        d = dout
        for i in reversed(range(self.n)):
            a_in, _ = self.cache[i]
            gW[i] = a_in.T @ d
            gb[i] = d.sum(axis=0)
            if i > 0:
                da = d @ self.W[i].T
                z_prev = self.cache[i - 1][1]
                d = da * (z_prev > 0)
        return gW, gb

    def copy_from(self, other):
        self.W = [w.copy() for w in other.W]
        self.b = [b.copy() for b in other.b]

    def params(self):
        return self.W + self.b


# --------------------------------------------------------------------------
# Adam optimizer over the MLP's parameter list.
# --------------------------------------------------------------------------
class Adam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, betas[0], betas[1], eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            mhat = self.m[i] / (1 - self.b1 ** self.t)
            vhat = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


# --------------------------------------------------------------------------
# Experience replay (fixed-size circular buffer).
# --------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity, obs_dim, rng):
        self.capacity = capacity
        self.rng = rng
        self.s = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.ns = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self.ptr = 0

    def add(self, s, a, r, ns, done):
        i = self.ptr
        self.s[i], self.a[i], self.r[i], self.ns[i], self.done[i] = s, a, r, ns, float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch):
        idx = self.rng.integers(0, self.size, size=batch)
        return self.s[idx], self.a[idx], self.r[idx], self.ns[idx], self.done[idx]


# --------------------------------------------------------------------------
# DQN agent: online + target network, epsilon-greedy, Huber-style TD update.
# --------------------------------------------------------------------------
class DQNAgent:
    def __init__(self, obs_dim, n_actions, seed=0, hidden=(128, 128), lr=1e-3, gamma=0.97):
        self.rng = np.random.default_rng(seed)
        self.n_actions = n_actions
        self.gamma = gamma
        sizes = [obs_dim, *hidden, n_actions]
        self.online = MLP(sizes, self.rng)
        self.target = MLP(sizes, self.rng)
        self.target.copy_from(self.online)
        self.opt = Adam(self.online.params(), lr=lr)

    def act(self, obs, epsilon=0.0):
        if self.rng.random() < epsilon:
            return int(self.rng.integers(0, self.n_actions))
        q = self.online.forward(obs[None, :])
        return int(np.argmax(q[0]))

    def learn(self, batch):
        s, a, r, ns, done = batch
        # Target: r + gamma * max_a' Q_target(ns, a') * (1 - done).
        next_q = self.target.forward(ns)
        target = r + self.gamma * next_q.max(axis=1) * (1.0 - done)

        q = self.online.forward(s)
        q_sa = q[np.arange(len(a)), a]
        td = q_sa - target
        # Huber gradient (clip to [-1, 1]) for stability.
        td = np.clip(td, -1.0, 1.0)

        dout = np.zeros_like(q)
        dout[np.arange(len(a)), a] = td / len(a)
        gW, gb = self.online.backward(dout)
        self.opt.step(self.online.params(), gW + gb)
        return float(np.mean(td ** 2))

    def sync_target(self):
        self.target.copy_from(self.online)

    # ---- persistence -----------------------------------------------------
    def save(self, path):
        arrays = {}
        for i, w in enumerate(self.online.W):
            arrays[f"W{i}"] = w
        for i, b in enumerate(self.online.b):
            arrays[f"b{i}"] = b
        np.savez(path, **arrays)

    def export_json(self, path, meta=None):
        """Export weights as JSON for in-browser inference in aiAgent.js.
        A tiny MLP needs no onnxruntime — the forward pass is a few matmuls."""
        model = {
            "sizes": self.online.sizes,
            "W": [w.tolist() for w in self.online.W],
            "b": [b.tolist() for b in self.online.b],
            "meta": meta or {},
        }
        with open(path, "w") as f:
            json.dump(model, f)
