// Core logic tests. See DESIGN.md §13. Run with: npm test
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { GameState, Direction, Status, key } from '../js/state.js';

// Build a game with a controlled layout: no apples on the snake's path, so the
// snake can move deterministically. We clear apples after construction for the
// movement/collision tests that shouldn't involve eating.
function freshGame(size = 10) {
  const g = new GameState(size, () => 0); // rng->0 is fine; we override apples
  g.apples = new Set();
  return g;
}

test('head advances by the direction delta each tick; tail follows', () => {
  const g = freshGame();
  const head0 = { ...g.head };
  const len0 = g.snake.length;
  g.tick();
  assert.deepEqual(g.head, { x: head0.x + 1, y: head0.y }); // moving RIGHT
  assert.equal(g.snake.length, len0); // no growth without eating
});

test('eating grows the snake, bumps score, and keeps apple count', () => {
  const g = freshGame();
  const target = { x: g.head.x + 1, y: g.head.y };
  g.apples = new Set([key(target)]);
  g.appleCount = 1;
  const len0 = g.snake.length;
  g.tick();
  assert.equal(g.score, 1);
  assert.equal(g.snake.length, len0 + 1); // grew
  assert.equal(g.apples.size, 1);         // respawned to keep count
  assert.ok(!g.apples.has(key(target)));  // the eaten apple is gone
});

test('wall collision ends the game on each of the four edges', () => {
  for (const [dir, place] of [
    [Direction.RIGHT, (g) => ({ x: g.size - 1, y: 5 })],
    [Direction.LEFT, (g) => ({ x: 0, y: 5 })],
    [Direction.UP, (g) => ({ x: 5, y: 0 })],
    [Direction.DOWN, (g) => ({ x: 5, y: g.size - 1 })],
  ]) {
    const g = freshGame();
    const h = place(g);
    // Rebuild a length-1 snake at the edge facing the wall.
    g.snake = [h];
    g.snakeCells = new Set([key(h)]);
    g.direction = dir;
    g.tick();
    assert.equal(g.status, Status.GAME_OVER);
  }
});

test('self collision ends the game', () => {
  const g = freshGame();
  // Spiral where the head, turning RIGHT, steps onto a NON-tail body segment.
  // head (5,5); moving RIGHT lands on (6,5), a body cell (index 5). The tail is
  // (6,4), so (6,5) does NOT vacate — a genuine self-collision.
  g.snake = [
    { x: 5, y: 5 }, { x: 4, y: 5 }, { x: 4, y: 6 }, { x: 5, y: 6 },
    { x: 6, y: 6 }, { x: 6, y: 5 }, { x: 6, y: 4 },
  ];
  g.snakeCells = new Set(g.snake.map(key));
  g.direction = Direction.RIGHT;
  g.tick();
  assert.equal(g.status, Status.GAME_OVER);
});

test('stepping into the vacating tail cell is NOT a collision', () => {
  const g = freshGame();
  // 2x2 loop: head at (5,5), tail at (5,6). Turning DOWN sends the head onto
  // (5,6) — but the tail vacates it the same tick, so the move is legal.
  g.snake = [{ x: 5, y: 5 }, { x: 6, y: 5 }, { x: 6, y: 6 }, { x: 5, y: 6 }];
  g.snakeCells = new Set(g.snake.map(key));
  g.direction = Direction.DOWN; // head (5,5) -> (5,6), the current tail cell
  g.tick();
  assert.equal(g.status, Status.PLAYING);
  assert.deepEqual(g.head, { x: 5, y: 6 });
});

test('reverse-into-neck requests are ignored', () => {
  const g = freshGame();
  g.direction = Direction.RIGHT;
  g.setDirection(Direction.LEFT); // 180° reverse
  assert.equal(g.pending, null);
  g.tick();
  assert.equal(g.direction, Direction.RIGHT); // unchanged
});

test('initial apple count is within [5, 7] and apples avoid the snake', () => {
  for (let seed = 0; seed < 20; seed++) {
    const g = new GameState(10, mulberry32(seed));
    assert.ok(g.appleCount >= 5 && g.appleCount <= 7);
    assert.equal(g.apples.size, g.appleCount);
    for (const a of g.apples) assert.ok(!g.snakeCells.has(a));
  }
});

// Small seeded PRNG for reproducible tests.
function mulberry32(seed) {
  let a = seed + 0x6d2b79f5;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
