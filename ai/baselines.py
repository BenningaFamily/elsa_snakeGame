"""Baseline (non-learning) agents for benchmarking (AI_PLAN.md §9).

Each agent exposes ``act(view) -> relative_action`` in {0,1,2}, matching the
env's action space, so they can be evaluated with the same harness as the DQN.
"""

from __future__ import annotations

from collections import deque

from .env import relative_to_absolute, turn_left, turn_right


def _safe(view, cell):
    size = view["size"]
    x, y = cell
    if not (0 <= x < size and 0 <= y < size):
        return False
    tail = view["snake"][-1]
    if cell == tail:
        return True
    return cell not in view["snake_cells"]


def _safe_actions(view):
    head = view["snake"][0]
    d = view["direction"]
    out = []
    for a in (0, 1, 2):
        nd = relative_to_absolute(d, a)
        if _safe(view, (head[0] + nd[0], head[1] + nd[1])):
            out.append(a)
    return out


def _delta_to_action(heading, delta):
    if delta == heading:
        return 0
    if delta == turn_left(heading):
        return 1
    if delta == turn_right(heading):
        return 2
    return None  # a reverse — not reachable via relative actions


class RandomAgent:
    """Picks a uniformly random *safe* action; matches js RandomAgent (§5.1)."""

    def __init__(self, rng):
        self.rng = rng

    def act(self, view):
        safe = _safe_actions(view)
        if not safe:
            return 0
        return int(safe[self.rng.integers(0, len(safe))])


class GreedyBFSAgent:
    """Breadth-first search from the head to the nearest reachable apple; steps
    one cell along that path. Falls back to any safe move. A strong, simple
    benchmark for the learned agent to beat (AI_PLAN.md §9)."""

    def __init__(self, rng=None):
        self.rng = rng

    def act(self, view):
        step = self._bfs_first_step(view)
        if step is not None:
            action = _delta_to_action(view["direction"], step)
            if action is not None:
                return action
        # No path to food (or path requires a reverse): stay safe.
        safe = _safe_actions(view)
        if not safe:
            return 0
        if self.rng is not None:
            return int(safe[self.rng.integers(0, len(safe))])
        return safe[0]

    def _bfs_first_step(self, view):
        size = view["size"]
        head = view["snake"][0]
        tail = view["snake"][-1]
        blocked = set(view["snake_cells"])
        blocked.discard(tail)  # the tail cell will vacate
        apples = view["apples"]
        if not apples:
            return None

        # BFS over free cells; record the first move that leaves the head.
        # queue holds (cell, first_step_delta)
        q = deque()
        seen = {head}
        for delta in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nc = (head[0] + delta[0], head[1] + delta[1])
            if 0 <= nc[0] < size and 0 <= nc[1] < size and nc not in blocked:
                # Don't allow the immediate reverse of the current heading.
                d = view["direction"]
                if delta == (-d[0], -d[1]):
                    continue
                q.append((nc, delta))
                seen.add(nc)

        while q:
            cell, first = q.popleft()
            if cell in apples:
                return first
            for delta in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                nc = (cell[0] + delta[0], cell[1] + delta[1])
                if (
                    0 <= nc[0] < size
                    and 0 <= nc[1] < size
                    and nc not in blocked
                    and nc not in seen
                ):
                    seen.add(nc)
                    q.append((nc, first))
        return None
