"""SnakeEnv — a Gym-style Snake environment that faithfully mirrors the game
logic in ``js/state.js`` (see DESIGN.md §6.3 and AI_PLAN.md §8).

Kept deliberately dependency-light (only numpy, for the RNG) so it can serve as
the single source of truth for training. The tick / collision / spawn rules here
must stay behaviorally identical to the JavaScript engine — the parity tests in
``ai/tests/`` guard that.

Action space: 3 *relative* actions (AI_PLAN.md §4):
    0 = go straight, 1 = turn left, 2 = turn right
decoded against the current heading into an absolute direction. The 180° reverse
is unreachable, exactly as in the game.
"""

from __future__ import annotations

import numpy as np

# Directions in screen coordinates (y grows downward), matching js/state.js.
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

APPLE_MIN = 5
APPLE_MAX = 7
START_LENGTH = 3


def turn_left(d):
    # RIGHT(1,0) -> UP(0,-1); counter-clockwise in y-down coords.
    return (d[1], -d[0])


def turn_right(d):
    # RIGHT(1,0) -> DOWN(0,1); clockwise in y-down coords.
    return (-d[1], d[0])


def relative_to_absolute(heading, action):
    if action == 0:
        return heading
    if action == 1:
        return turn_left(heading)
    if action == 2:
        return turn_right(heading)
    raise ValueError(f"invalid action {action}")


class SnakeEnv:
    def __init__(
        self,
        size=8,
        seed=None,
        reward_eat=10.0,
        reward_die=-10.0,
        reward_step=-0.01,
        max_no_food=None,
    ):
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.reward_eat = reward_eat
        self.reward_die = reward_die
        self.reward_step = reward_step
        # Truncate an episode if the snake goes too long without eating (it is
        # stuck / circling). With 5-7 apples always present, a long drought means
        # the agent is trapped.
        self.max_no_food = max_no_food if max_no_food is not None else size * size
        self.reset()

    # ---- lifecycle -------------------------------------------------------
    def reset(self):
        mid = self.size // 2
        # Snake horizontal, head furthest right, oriented RIGHT (as in state.js).
        self.snake = [(mid - i, mid) for i in range(START_LENGTH)]
        self.snake_cells = set(self.snake)
        self.direction = RIGHT
        self.status_over = False
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0

        self.apple_count = APPLE_MIN + int(self.rng.integers(0, APPLE_MAX - APPLE_MIN + 1))
        self.apples = set()
        for _ in range(self.apple_count):
            self._spawn_apple()
        return self.view()

    @property
    def head(self):
        return self.snake[0]

    def view(self):
        """Read-only snapshot, mirroring the JS GameView (AI_PLAN.md §3)."""
        return {
            "size": self.size,
            "snake": list(self.snake),
            "snake_cells": set(self.snake_cells),
            "apples": set(self.apples),
            "direction": self.direction,
        }

    def in_bounds(self, cell):
        x, y = cell
        return 0 <= x < self.size and 0 <= y < self.size

    def _spawn_apple(self):
        free = [
            (x, y)
            for x in range(self.size)
            for y in range(self.size)
            if (x, y) not in self.snake_cells and (x, y) not in self.apples
        ]
        if not free:
            return  # board full — skip rather than loop forever
        idx = int(self.rng.integers(0, len(free)))
        self.apples.add(free[idx])

    # ---- stepping --------------------------------------------------------
    def step(self, action):
        """Apply a relative action. Returns (view, reward, done, info)."""
        if self.status_over:
            raise RuntimeError("step() called on a finished episode; call reset()")

        # Decode the relative action into an absolute heading. A choice that
        # reverses into the neck is ignored (keep going straight), matching the
        # engine's reverse guard — but relative actions never produce a reverse.
        new_dir = relative_to_absolute(self.direction, action)
        if not (new_dir[0] == -self.direction[0] and new_dir[1] == -self.direction[1]):
            self.direction = new_dir

        self.steps += 1
        self.steps_since_food += 1

        next_head = (self.head[0] + self.direction[0], self.head[1] + self.direction[1])

        # Wall collision.
        if not self.in_bounds(next_head):
            self.status_over = True
            return self.view(), self.reward_die, True, {"reason": "wall", "score": self.score}

        eating = next_head in self.apples
        tail = self.snake[-1]

        # Self collision, with the tail-vacate exception (DESIGN.md §6.3).
        if next_head in self.snake_cells and not (next_head == tail and not eating):
            self.status_over = True
            return self.view(), self.reward_die, True, {"reason": "self", "score": self.score}

        # Move the head.
        self.snake.insert(0, next_head)
        self.snake_cells.add(next_head)

        if eating:
            self.apples.discard(next_head)
            self.score += 1
            self.steps_since_food = 0
            self._spawn_apple()
            reward = self.reward_eat
        else:
            self.snake.pop()
            if tail not in self.snake:  # only free the cell if unoccupied now
                self.snake_cells.discard(tail)
            reward = self.reward_step

        # Truncation: stuck without food for too long.
        truncated = self.steps_since_food >= self.max_no_food
        done = truncated
        info = {"score": self.score}
        if truncated:
            info["reason"] = "truncated"
        return self.view(), reward, done, info
