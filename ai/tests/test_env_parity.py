"""Parity tests: the Python SnakeEnv must behave identically to js/state.js.

These mirror the exact scenarios in tests/state.test.js (the JS suite) so the
training environment and the game engine can't silently diverge (AI_PLAN.md §8,
§11). Run with:  python -m unittest discover -s ai/tests
"""

import unittest

from ai.env import DOWN, LEFT, RIGHT, UP, SnakeEnv, relative_to_absolute


def make(size=10):
    env = SnakeEnv(size=size, seed=0)
    env.apples = set()  # clear apples for deterministic movement scenarios
    return env


def set_snake(env, cells, direction):
    env.snake = list(cells)
    env.snake_cells = set(cells)
    env.direction = direction
    env.status_over = False


def absolute_action(heading, target_dir):
    """Return the relative action that yields target_dir from heading."""
    for a in (0, 1, 2):
        if relative_to_absolute(heading, a) == target_dir:
            return a
    raise AssertionError("target_dir is a reverse — unreachable")


class TestParity(unittest.TestCase):
    def test_head_advances_and_tail_follows(self):
        env = make()
        set_snake(env, [(5, 5), (4, 5), (3, 5)], RIGHT)
        n0 = len(env.snake)
        env.step(0)  # straight
        self.assertEqual(env.head, (6, 5))
        self.assertEqual(len(env.snake), n0)  # no growth without eating

    def test_eating_grows_and_scores(self):
        env = make()
        set_snake(env, [(5, 5), (4, 5), (3, 5)], RIGHT)
        env.apples = {(6, 5)}
        env.apple_count = 1
        n0 = len(env.snake)
        _, reward, done, info = env.step(0)
        self.assertEqual(info["score"], 1)
        self.assertEqual(len(env.snake), n0 + 1)
        self.assertEqual(env.reward_eat, reward)
        self.assertFalse(done)
        self.assertEqual(len(env.apples), 1)  # respawned to keep count

    def test_wall_death_all_edges(self):
        for direction, head in [
            (RIGHT, (9, 5)), (LEFT, (0, 5)), (UP, (5, 0)), (DOWN, (5, 9)),
        ]:
            env = make()
            set_snake(env, [head], direction)
            _, reward, done, info = env.step(0)
            self.assertTrue(done)
            self.assertEqual(info["reason"], "wall")
            self.assertEqual(env.reward_die, reward)

    def test_self_collision(self):
        env = make()
        # Spiral where turning onto a NON-tail body segment is fatal.
        set_snake(
            env,
            [(5, 5), (4, 5), (4, 6), (5, 6), (6, 6), (6, 5), (6, 4)],
            RIGHT,
        )
        # head (5,5) -> (6,5) is a body cell (not the tail (6,4)).
        _, _, done, info = env.step(0)  # straight = RIGHT
        self.assertTrue(done)
        self.assertEqual(info["reason"], "self")

    def test_tail_follow_is_safe(self):
        env = make()
        # 2x2 loop: head (5,5), tail (5,6). Turning DOWN onto the vacating tail
        # cell is legal, not a crash.
        set_snake(env, [(5, 5), (6, 5), (6, 6), (5, 6)], RIGHT)
        action = absolute_action(RIGHT, DOWN)
        _, _, done, _ = env.step(action)
        self.assertFalse(done)
        self.assertEqual(env.head, (5, 6))

    def test_reverse_is_unreachable_via_relative_actions(self):
        # No relative action produces the reverse of the heading.
        for heading in (UP, DOWN, LEFT, RIGHT):
            reverse = (-heading[0], -heading[1])
            produced = {relative_to_absolute(heading, a) for a in (0, 1, 2)}
            self.assertNotIn(reverse, produced)

    def test_initial_apple_count_and_placement(self):
        for seed in range(20):
            env = SnakeEnv(size=10, seed=seed)
            self.assertTrue(5 <= env.apple_count <= 7)
            self.assertEqual(len(env.apples), env.apple_count)
            self.assertTrue(env.apples.isdisjoint(env.snake_cells))


if __name__ == "__main__":
    unittest.main()
