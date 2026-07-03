"""Evaluation harness (AI_PLAN.md §9). Runs greedy episodes and reports the
primary metrics: apples eaten, steps survived, best game."""

from __future__ import annotations

import numpy as np

from .env import SnakeEnv


def run_episode(env, act_fn):
    view = env.reset()
    steps = 0
    while True:
        action = act_fn(view)
        view, _, done, info = env.step(action)
        steps += 1
        if done:
            return info["score"], steps


def evaluate(act_fn, size=8, episodes=200, seed=1234):
    """act_fn: view -> relative action. Returns a metrics dict."""
    env = SnakeEnv(size=size, seed=seed)
    scores, lengths = [], []
    for _ in range(episodes):
        score, steps = run_episode(env, act_fn)
        scores.append(score)
        lengths.append(steps)
    scores = np.array(scores)
    return {
        "mean_score": float(scores.mean()),
        "median_score": float(np.median(scores)),
        "max_score": int(scores.max()),
        "mean_steps": float(np.mean(lengths)),
        "episodes": episodes,
    }


def format_metrics(name, m):
    return (
        f"{name:<16} mean_score={m['mean_score']:5.2f}  "
        f"median={m['median_score']:4.1f}  max={m['max_score']:3d}  "
        f"mean_steps={m['mean_steps']:6.1f}"
    )
