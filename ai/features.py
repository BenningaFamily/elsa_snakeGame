"""Observation encoders (AI_PLAN.md §3).

- ``featurize`` — the simple 11-feature vector for the MLP-DQN milestone (§3.1).
- ``grid_encode`` — the multi-channel board tensor for the later CNN (§3.2),
  included now so both milestones share one spec.

PARITY: whatever encoding is used to train must be reproduced byte-for-byte at
inference time in ``js/agent/aiAgent.js`` (AI_PLAN.md §3.2, "Parity rule").
"""

from __future__ import annotations

import numpy as np

from .env import turn_left, turn_right

FEATURE_SIZE = 11


def _unsafe(view, cell):
    """Would occupying ``cell`` next tick be fatal? Mirrors env collision rules
    (tail cell is vacating, so it counts as safe)."""
    size = view["size"]
    x, y = cell
    if not (0 <= x < size and 0 <= y < size):
        return True
    tail = view["snake"][-1]
    if cell == tail:
        return False
    return cell in view["snake_cells"]


def featurize(view):
    """11-D float32 vector:

        [danger_straight, danger_left, danger_right,
         dir_up, dir_down, dir_left, dir_right,
         food_up, food_down, food_left, food_right]
    """
    head = view["snake"][0]
    d = view["direction"]
    left = turn_left(d)
    right = turn_right(d)

    def cell(delta):
        return (head[0] + delta[0], head[1] + delta[1])

    danger_straight = _unsafe(view, cell(d))
    danger_left = _unsafe(view, cell(left))
    danger_right = _unsafe(view, cell(right))

    dir_up = d == (0, -1)
    dir_down = d == (0, 1)
    dir_left = d == (-1, 0)
    dir_right = d == (1, 0)

    # Nearest apple by Manhattan distance; encode its direction from the head.
    food_up = food_down = food_left = food_right = 0
    if view["apples"]:
        hx, hy = head
        ax, ay = min(view["apples"], key=lambda a: abs(a[0] - hx) + abs(a[1] - hy))
        food_up = ay < hy
        food_down = ay > hy
        food_left = ax < hx
        food_right = ax > hx

    return np.array(
        [
            danger_straight, danger_left, danger_right,
            dir_up, dir_down, dir_left, dir_right,
            food_up, food_down, food_left, food_right,
        ],
        dtype=np.float32,
    )


def grid_encode(view):
    """(4, H, W) float32 tensor: [body, head, apples, walls]. For the CNN
    milestone (AI_PLAN.md §3.2). Walls channel is all zeros here since the board
    has no interior walls; kept for a stable channel layout."""
    size = view["size"]
    body = np.zeros((size, size), dtype=np.float32)
    head = np.zeros((size, size), dtype=np.float32)
    apples = np.zeros((size, size), dtype=np.float32)
    walls = np.zeros((size, size), dtype=np.float32)

    hx, hy = view["snake"][0]
    head[hy, hx] = 1.0
    for (x, y) in view["snake"][1:]:
        body[y, x] = 1.0
    for (x, y) in view["apples"]:
        apples[y, x] = 1.0
    return np.stack([body, head, apples, walls], axis=0)
