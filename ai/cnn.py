"""CNN-DQN in PyTorch (AI_PLAN.md §2, §3.2, §7) — the milestone after the
feature-MLP. Reads the board as a (4, H, W) grid (`features.grid_encode`) so the
network can reason about spatial structure (enclosed regions, escape routes) the
11-feature vector throws away.

Uses **Double DQN** (online net picks the next action, target net evaluates it)
to reduce value over-estimation, plus a target network and experience replay.
"""

from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

GRID_CHANNELS = 4
N_ACTIONS = 3


class SnakeCNN(nn.Module):
    # Keeps 64 feature channels (the capacity that beat the greedy baseline) but
    # inserts a 2x2 max-pool before the flatten, cutting the spatial dims in half
    # so the first dense matrix is 4x smaller. This recovers big-model performance
    # at a fraction of the size (12x12 ~2.6 MB vs 9.7 MB). The forward is a plain
    # sequence of conv / pool / dense ops, walked generically by export_binary and
    # mirrored in js/agent/aiAgent.js.
    def __init__(self, size, channels=GRID_CHANNELS, n_actions=N_ACTIONS, fc=256):
        super().__init__()
        self.size = size
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                    # size -> size // 2
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )
        pooled = size // 2
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * pooled * pooled, fc), nn.ReLU(),
            nn.Linear(fc, n_actions),
        )

    def forward(self, x):
        return self.head(self.conv(x))


class GridReplay:
    def __init__(self, capacity, size, channels=GRID_CHANNELS):
        self.capacity = capacity
        self.s = np.zeros((capacity, channels, size, size), dtype=np.float32)
        self.ns = np.zeros((capacity, channels, size, size), dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self.ptr = 0

    def add(self, s, a, r, ns, done):
        i = self.ptr
        self.s[i], self.a[i], self.r[i], self.ns[i], self.done[i] = s, a, r, ns, float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_batch(self, s, a, r, ns, done):
        n = len(a)
        idx = (self.ptr + np.arange(n)) % self.capacity
        self.s[idx] = s
        self.ns[idx] = ns
        self.a[idx] = a
        self.r[idx] = r
        self.done[idx] = done
        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch, rng):
        idx = rng.integers(0, self.size, size=batch)
        return (
            torch.from_numpy(self.s[idx]),
            torch.from_numpy(self.a[idx]),
            torch.from_numpy(self.r[idx]),
            torch.from_numpy(self.ns[idx]),
            torch.from_numpy(self.done[idx]),
        )


class CNNAgent:
    def __init__(self, size, seed=0, lr=5e-4, gamma=0.97, device="cpu"):
        torch.manual_seed(seed)
        self.device = torch.device(device)
        self.gamma = gamma
        self.n_actions = N_ACTIONS
        self.online = SnakeCNN(size).to(self.device)
        self.target = SnakeCNN(size).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.opt = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.rng = np.random.default_rng(seed)

    @torch.no_grad()
    def act(self, grid, epsilon=0.0):
        if self.rng.random() < epsilon:
            return int(self.rng.integers(0, self.n_actions))
        x = torch.from_numpy(grid[None]).to(self.device)
        return int(self.online(x).argmax(dim=1).item())

    @torch.no_grad()
    def act_batch(self, grids, epsilon=0.0):
        """Batched epsilon-greedy over a (N,C,H,W) numpy array -> (N,) actions.
        One forward pass for all envs, so the GPU sees a real batch."""
        n = grids.shape[0]
        x = torch.from_numpy(grids).to(self.device)
        greedy = self.online(x).argmax(dim=1).cpu().numpy()
        explore = self.rng.random(n) < epsilon
        rand = self.rng.integers(0, self.n_actions, size=n)
        return np.where(explore, rand, greedy).astype(np.int64)

    def learn(self, batch):
        s, a, r, ns, done = [t.to(self.device) for t in batch]
        q = self.online(s).gather(1, a[:, None]).squeeze(1)
        with torch.no_grad():
            # Double DQN: online selects, target evaluates.
            next_actions = self.online(ns).argmax(dim=1, keepdim=True)
            next_q = self.target(ns).gather(1, next_actions).squeeze(1)
            target = r + self.gamma * next_q * (1.0 - done)
        loss = F.smooth_l1_loss(q, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.opt.step()
        return float(loss.item())

    def sync_target(self):
        self.target.load_state_dict(self.online.state_dict())

    # ---- persistence -----------------------------------------------------
    def save(self, path):
        torch.save(self.online.state_dict(), path + ".pt")

    def load(self, path):
        self.online.load_state_dict(torch.load(path, map_location=self.device))
        self.target.load_state_dict(self.online.state_dict())

    def export_binary(self, path, meta=None):
        """Export weights as a compact binary (.bin, little-endian float32) plus a
        small .json manifest of layer ops, for the plain-JS forward pass in
        aiAgent.js. Walks the module list generically so conv / pool / dense ops
        are recorded in execution order; each weighted layer writes weight then
        bias into the .bin, and JS reads them back in the same order. Lossless and
        no onnxruntime needed for a net this small.
        """
        net = self.online
        layers, buf = [], bytearray()

        def put(arr):
            buf.extend(np.asarray(arr, dtype="<f4").tobytes())

        def walk(seq):
            for m in seq:
                if isinstance(m, nn.Conv2d):
                    layers.append({"type": "conv", "shape": list(m.weight.shape)})
                    put(m.weight.detach().cpu().numpy())
                    put(m.bias.detach().cpu().numpy())
                elif isinstance(m, nn.MaxPool2d):
                    k = m.kernel_size if isinstance(m.kernel_size, int) else m.kernel_size[0]
                    layers.append({"type": "pool", "k": int(k)})
                elif isinstance(m, nn.Linear):
                    layers.append({"type": "dense", "shape": list(m.weight.shape)})
                    put(m.weight.detach().cpu().numpy())
                    put(m.bias.detach().cpu().numpy())
                # ReLU / Flatten are implied by convention in the JS forward.

        walk(net.conv)
        walk(net.head)
        manifest = {"arch": "cnn-v2", "size": net.size, "channels": GRID_CHANNELS,
                    "dtype": "float32", "layers": layers, "meta": meta or {}}
        with open(path + ".json", "w") as f:
            json.dump(manifest, f)
        with open(path + ".bin", "wb") as f:
            f.write(buf)
