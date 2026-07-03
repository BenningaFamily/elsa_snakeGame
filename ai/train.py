"""Train the MLP-DQN on Snake (AI_PLAN.md milestones 2-3).

Usage:
    python -m ai.train --size 8 --steps 100000 --out ai/models/dqn_8

Produces <out>.npz (numpy weights) and <out>.json (weights for aiAgent.js).
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from .baselines import GreedyBFSAgent, RandomAgent
from .dqn import DQNAgent, ReplayBuffer
from .env import SnakeEnv
from .evaluate import evaluate, format_metrics
from .features import FEATURE_SIZE, featurize


def linear_epsilon(step, start, end, decay_steps):
    if step >= decay_steps:
        return end
    return start + (end - start) * (step / decay_steps)


def train(
    size=8,
    steps=100_000,
    seed=0,
    capacity=50_000,
    batch=64,
    warmup=1_000,
    train_freq=1,
    target_sync=1_000,
    gamma=0.97,
    lr=1e-3,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay_frac=0.4,
    eval_every=20_000,
    out=None,
    verbose=True,
):
    rng = np.random.default_rng(seed)
    env = SnakeEnv(size=size, seed=seed)
    agent = DQNAgent(FEATURE_SIZE, 3, seed=seed, gamma=gamma, lr=lr)
    buffer = ReplayBuffer(capacity, FEATURE_SIZE, rng)

    def greedy_act(view):
        return agent.act(featurize(view), epsilon=0.0)

    view = env.reset()
    obs = featurize(view)
    ep_score = 0
    recent_scores = []
    decay_steps = int(steps * eps_decay_frac)
    t0 = time.time()

    for step in range(1, steps + 1):
        eps = linear_epsilon(step, eps_start, eps_end, decay_steps)
        action = agent.act(obs, epsilon=eps)
        nview, reward, done, info = env.step(action)
        nobs = featurize(nview)
        buffer.add(obs, action, reward, nobs, done)

        obs = nobs
        ep_score = info["score"]

        if done:
            recent_scores.append(ep_score)
            if len(recent_scores) > 100:
                recent_scores.pop(0)
            view = env.reset()
            obs = featurize(view)
            ep_score = 0

        if buffer.size >= warmup and step % train_freq == 0:
            agent.learn(buffer.sample(batch))
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
        agent.save(out + ".npz")
        agent.export_json(out + ".json", meta={"size": size, "obs": "features-v1", "features": FEATURE_SIZE})
        if verbose:
            print(f"saved {out}.npz and {out}.json")

    return agent, greedy_act


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=8)
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="ai/models/dqn_8")
    args = p.parse_args()

    agent, greedy_act = train(size=args.size, steps=args.steps, seed=args.seed, out=args.out)

    # Final head-to-head vs baselines (AI_PLAN.md §9).
    rng = np.random.default_rng(123)
    print("\n=== Final evaluation (200 games, size %d) ===" % args.size)
    print(format_metrics("DQN", evaluate(greedy_act, size=args.size, episodes=200)))
    print(format_metrics("GreedyBFS", evaluate(GreedyBFSAgent(rng).act, size=args.size, episodes=200)))
    print(format_metrics("Random", evaluate(RandomAgent(rng).act, size=args.size, episodes=200)))


if __name__ == "__main__":
    main()
