"""Train the CNN-DQN on Snake (AI_PLAN.md milestones 3-4).

Usage:
    python -m ai.train_cnn --size 8 --steps 150000 --out ai/models/cnn_8

Produces <out>.pt (torch weights) and <out>.json (weights for aiAgent.js).
Reuses SnakeEnv, grid_encode, and the evaluate harness so results are directly
comparable to the feature-MLP.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from .baselines import GreedyBFSAgent, RandomAgent
from .cnn import CNNAgent, GridReplay
from .env import SnakeEnv
from .evaluate import evaluate, format_metrics
from .features import grid_encode
from .train import linear_epsilon


def train(
    size=8,
    steps=150_000,
    seed=0,
    capacity=50_000,
    batch=64,
    warmup=2_000,
    train_freq=1,
    target_sync=1_000,
    gamma=0.97,
    lr=5e-4,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay_frac=0.5,
    eval_every=25_000,
    out=None,
    resume=None,
    device="cpu",
    verbose=True,
):
    rng = np.random.default_rng(seed)
    env = SnakeEnv(size=size, seed=seed)
    agent = CNNAgent(size, seed=seed, gamma=gamma, lr=lr, device=device)
    if resume:
        agent.load(resume)
        if verbose:
            print(f"resumed from {resume}")
    buffer = GridReplay(capacity, size)

    def greedy_act(view):
        return agent.act(grid_encode(view), epsilon=0.0)

    view = env.reset()
    obs = grid_encode(view)
    recent_scores = []
    decay_steps = int(steps * eps_decay_frac)
    t0 = time.time()

    for step in range(1, steps + 1):
        eps = linear_epsilon(step, eps_start, eps_end, decay_steps)
        action = agent.act(obs, epsilon=eps)
        nview, reward, done, info = env.step(action)
        nobs = grid_encode(nview)
        buffer.add(obs, action, reward, nobs, done)
        obs = nobs

        if done:
            recent_scores.append(info["score"])
            if len(recent_scores) > 100:
                recent_scores.pop(0)
            view = env.reset()
            obs = grid_encode(view)

        if buffer.size >= warmup and step % train_freq == 0:
            buffer_batch = buffer.sample(batch, rng)
            agent.learn(buffer_batch)
        if step % target_sync == 0:
            agent.sync_target()

        if verbose and step % eval_every == 0:
            m = evaluate(greedy_act, size=size, episodes=100, seed=9999)
            avg = np.mean(recent_scores) if recent_scores else 0.0
            print(
                f"step {step:>7d}  eps={eps:4.2f}  train_avg100={avg:5.2f}  "
                f"eval_mean={m['mean_score']:5.2f}  max={m['max_score']:3d}  "
                f"({time.time()-t0:5.1f}s)"
            )

    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        agent.save(out)
        agent.export_binary(out, meta={"size": size, "obs": "grid-v1"})
        if verbose:
            print(f"saved {out}.pt and {out}.json/.bin")

    return agent, greedy_act


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=8)
    p.add_argument("--steps", type=int, default=150_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="ai/models/cnn_8")
    p.add_argument("--resume", type=str, default=None, help="path to a .pt checkpoint to continue from")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--device", type=str, default="auto", help="cpu | cuda | auto")
    p.add_argument("--threads", type=int, default=8, help="torch CPU threads (small nets: fewer is often faster)")
    args = p.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(max(1, args.threads))
    print(f"device: {device}")

    agent, greedy_act = train(
        size=args.size, steps=args.steps, seed=args.seed, out=args.out,
        resume=args.resume, lr=args.lr, eps_start=args.eps_start, eps_end=args.eps_end,
        device=device,
    )

    rng = np.random.default_rng(123)
    print("\n=== Final evaluation (200 games, size %d) ===" % args.size)
    print(format_metrics("CNN-DQN", evaluate(greedy_act, size=args.size, episodes=200)))
    print(format_metrics("GreedyBFS", evaluate(GreedyBFSAgent(rng).act, size=args.size, episodes=200)))
    print(format_metrics("Random", evaluate(RandomAgent(rng).act, size=args.size, episodes=200)))


if __name__ == "__main__":
    main()
