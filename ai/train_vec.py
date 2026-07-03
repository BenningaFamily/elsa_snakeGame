"""Vectorized, GPU-capable CNN-DQN trainer (AI_PLAN.md §8).

Collects from N parallel envs (batched inference) and runs the learner in the
same loop. Each outer step yields N transitions, so wall-clock per transition
drops and the GPU sees real batches during both collection and learning.

Usage (GPU, from the CUDA venv):
    /home/elsa/.venvs/snake-gpu/bin/python -m ai.train_vec \
        --size 10 --envs 64 --steps 12000 --device cuda --out ai/models/cnn_10
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from .baselines import GreedyBFSAgent, RandomAgent
from .cnn import CNNAgent, GridReplay
from .evaluate import evaluate, format_metrics
from .features import grid_encode
from .train import linear_epsilon
from .vec_env import VecSnakeEnv


def train(
    size=10,
    num_envs=64,
    steps=12_000,           # outer steps; total transitions = steps * num_envs
    seed=0,
    capacity=200_000,
    batch=256,
    updates_per_step=1,
    warmup_steps=50,        # outer steps before learning
    target_sync=500,        # in outer steps
    gamma=0.97,
    lr=5e-4,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay_frac=0.4,
    eval_every=2_000,
    device="cpu",
    out=None,
    resume=None,
    verbose=True,
):
    rng = np.random.default_rng(seed)
    vec = VecSnakeEnv(num_envs, size, seed=seed)
    agent = CNNAgent(size, seed=seed, gamma=gamma, lr=lr, device=device)
    if resume:
        agent.load(resume)
        if verbose:
            print(f"resumed from {resume}")
    buffer = GridReplay(capacity, size)

    def greedy_act(view):
        return agent.act(grid_encode(view), epsilon=0.0)

    obs = vec.reset()  # (N,C,H,W)
    recent = []
    decay_steps = int(steps * eps_decay_frac)
    t0 = time.time()

    for step in range(1, steps + 1):
        eps = linear_epsilon(step, eps_start, eps_end, decay_steps)
        actions = agent.act_batch(obs, epsilon=eps)
        nobs, rewards, dones, scores = vec.step(actions)
        buffer.add_batch(obs, actions, rewards, nobs, dones)
        obs = nobs

        finished = scores[scores >= 0]
        if finished.size:
            recent.extend(finished.tolist())
            if len(recent) > 200:
                recent = recent[-200:]

        if buffer.size >= warmup_steps * num_envs:
            for _ in range(updates_per_step):
                agent.learn(buffer.sample(batch, rng))
        if step % target_sync == 0:
            agent.sync_target()

        if verbose and step % eval_every == 0:
            m = evaluate(greedy_act, size=size, episodes=100, seed=9999)
            avg = np.mean(recent) if recent else 0.0
            trans = step * num_envs
            print(
                f"step {step:>6d} ({trans/1000:5.0f}k trans)  eps={eps:4.2f}  "
                f"train_avg={avg:5.2f}  eval_mean={m['mean_score']:5.2f}  "
                f"max={m['max_score']:3d}  ({time.time()-t0:5.1f}s)"
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
    p.add_argument("--size", type=int, default=10)
    p.add_argument("--envs", type=int, default=64)
    p.add_argument("--steps", type=int, default=12_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="ai/models/cnn_vec")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--updates-per-step", type=int, default=1)
    p.add_argument("--device", type=str, default="auto", help="cpu | cuda | auto")
    p.add_argument("--threads", type=int, default=8)
    args = p.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(max(1, args.threads))
    print(f"device: {device} | envs: {args.envs} | size: {args.size}")

    agent, greedy_act = train(
        size=args.size, num_envs=args.envs, steps=args.steps, seed=args.seed,
        out=args.out, resume=args.resume, lr=args.lr, batch=args.batch,
        updates_per_step=args.updates_per_step, device=device,
    )

    rng = np.random.default_rng(123)
    print("\n=== Final evaluation (200 games, size %d) ===" % args.size)
    print(format_metrics("CNN-DQN", evaluate(greedy_act, size=args.size, episodes=200)))
    print(format_metrics("GreedyBFS", evaluate(GreedyBFSAgent(rng).act, size=args.size, episodes=200)))
    print(format_metrics("Random", evaluate(RandomAgent(rng).act, size=args.size, episodes=200)))


if __name__ == "__main__":
    main()
