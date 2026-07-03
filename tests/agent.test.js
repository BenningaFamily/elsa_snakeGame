// RandomAgent tests. See DESIGN.md §13.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { RandomAgent } from '../js/agent/randomAgent.js';
import { Direction, key } from '../js/state.js';

function viewFrom(size, snake, direction) {
  return {
    size,
    snake,
    snakeCells: new Set(snake.map(key)),
    apples: new Set(),
    direction,
  };
}

test('never returns a move into a wall when a safe move exists', () => {
  // Head in the top-left corner moving RIGHT: UP and LEFT are walls.
  const view = viewFrom(8, [{ x: 0, y: 0 }, { x: 0, y: 1 }], Direction.RIGHT);
  const agent = new RandomAgent(() => 0);
  for (let i = 0; i < 50; i++) {
    const d = agent.nextDirection(view);
    const next = { x: view.snake[0].x + d.x, y: view.snake[0].y + d.y };
    assert.ok(next.x >= 0 && next.x < 8 && next.y >= 0 && next.y < 8, 'stays in bounds');
  }
});

test('never returns a move into the body when a safe move exists', () => {
  // Head at (5,5) with body cells blocking UP; DOWN/LEFT/RIGHT stay open.
  const snake = [{ x: 5, y: 5 }, { x: 5, y: 4 }, { x: 6, y: 4 }];
  const view = viewFrom(10, snake, Direction.RIGHT);
  const agent = new RandomAgent(() => 0);
  for (let i = 0; i < 50; i++) {
    const d = agent.nextDirection(view);
    const next = key({ x: 5 + d.x, y: 5 + d.y });
    // The only body cell adjacent is (5,4) = UP; agent must never pick it.
    assert.notEqual(next, key({ x: 5, y: 4 }));
  }
});

test('returns null when boxed in on all sides', () => {
  // Head at (1,1) walled/bodied on all four sides.
  const snake = [
    { x: 1, y: 1 },
    { x: 1, y: 0 }, // UP
    { x: 2, y: 1 }, // RIGHT
    { x: 1, y: 2 }, // DOWN
    { x: 0, y: 1 }, // LEFT
  ];
  const view = viewFrom(10, snake, Direction.RIGHT);
  const agent = new RandomAgent(() => 0);
  assert.equal(agent.nextDirection(view), null);
});

test('never reverses directly into its own neck', () => {
  const snake = [{ x: 5, y: 5 }, { x: 4, y: 5 }, { x: 3, y: 5 }];
  const view = viewFrom(10, snake, Direction.RIGHT);
  const agent = new RandomAgent(() => 0.999);
  for (let i = 0; i < 50; i++) {
    const d = agent.nextDirection(view);
    assert.ok(!(d.x === -Direction.RIGHT.x && d.y === -Direction.RIGHT.y));
  }
});
