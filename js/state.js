// GameState: pure game logic, no DOM references. See DESIGN.md §5 and §6.3.
// Fully unit-testable in isolation (Node/Vitest).

import { APPLE_MIN, APPLE_MAX, START_LENGTH } from './config.js';

export const Direction = {
  UP:    { x: 0,  y: -1 },
  DOWN:  { x: 0,  y: 1  },
  LEFT:  { x: -1, y: 0  },
  RIGHT: { x: 1,  y: 0  },
};

export const Status = { MENU: 'menu', PLAYING: 'playing', GAME_OVER: 'game_over' };

// Cell <-> string helpers. Cells are keyed as "x,y" so they live in a Set for
// O(1) membership tests (JS Set can't dedupe object references by value).
export const key = ({ x, y }) => `${x},${y}`;
export const parse = (s) => {
  const [x, y] = s.split(',').map(Number);
  return { x, y };
};

const isOpposite = (a, b) => a && b && a.x === -b.x && a.y === -b.y;

export class GameState {
  // `rng` is injectable so tests can be deterministic; defaults to Math.random.
  constructor(size, rng = Math.random) {
    this.size = size;
    this.rng = rng;
    this.reset();
  }

  reset() {
    const mid = Math.floor(this.size / 2);
    // Snake horizontal, head furthest right, oriented RIGHT.
    this.snake = [];
    for (let i = 0; i < START_LENGTH; i++) {
      this.snake.push({ x: mid - i, y: mid });
    }
    this.snakeCells = new Set(this.snake.map(key));
    this.direction = Direction.RIGHT;
    this.pending = null;
    this.status = Status.PLAYING;
    this.score = 0;

    // Roll apple count once for the round, then place them.
    this.appleCount = APPLE_MIN + this.randInt(APPLE_MAX - APPLE_MIN + 1);
    this.apples = new Set();
    for (let i = 0; i < this.appleCount; i++) this.spawnApple();
  }

  // Uniform integer in [0, n).
  randInt(n) {
    return Math.floor(this.rng() * n);
  }

  get head() {
    return this.snake[0];
  }

  inBounds({ x, y }) {
    return x >= 0 && x < this.size && y >= 0 && y < this.size;
  }

  // Buffer a direction request; applied at the start of the next tick.
  // A reverse-into-neck request is rejected (§6.2).
  setDirection(dir) {
    if (!dir || isOpposite(dir, this.direction)) return;
    this.pending = dir;
  }

  // Read-only snapshot handed to agents each tick (§5.1).
  view() {
    return {
      size: this.size,
      snake: this.snake,
      snakeCells: this.snakeCells,
      apples: this.apples,
      direction: this.direction,
    };
  }

  spawnApple() {
    // free = all interior cells − snakeCells − apples
    const free = [];
    for (let x = 0; x < this.size; x++) {
      for (let y = 0; y < this.size; y++) {
        const k = `${x},${y}`;
        if (!this.snakeCells.has(k) && !this.apples.has(k)) free.push(k);
      }
    }
    if (free.length === 0) return; // board full — skip rather than loop forever
    this.apples.add(free[this.randInt(free.length)]);
  }

  // Advance one cell. See DESIGN.md §6.3.
  tick() {
    if (this.status !== Status.PLAYING) return;

    // 1. Apply buffered direction (already validated as non-reverse).
    if (this.pending && !isOpposite(this.pending, this.direction)) {
      this.direction = this.pending;
    }
    this.pending = null;

    // 2. Compute next head.
    const nextHead = {
      x: this.head.x + this.direction.x,
      y: this.head.y + this.direction.y,
    };

    // 3. Wall collision.
    if (!this.inBounds(nextHead)) {
      this.status = Status.GAME_OVER;
      return;
    }

    const nextKey = key(nextHead);
    const eating = this.apples.has(nextKey);

    // 5. Self collision. The tail cell is vacated this tick, so moving into it
    //    is NOT a collision — unless we're eating (tail won't move).
    const tail = this.snake[this.snake.length - 1];
    const tailKey = key(tail);
    if (this.snakeCells.has(nextKey) && !(nextKey === tailKey && !eating)) {
      this.status = Status.GAME_OVER;
      return;
    }

    // 6. Push new head.
    this.snake.unshift(nextHead);
    this.snakeCells.add(nextKey);

    // 7. Eat (grow + respawn) or move (drop tail).
    if (eating) {
      this.apples.delete(nextKey);
      this.score += 1;
      this.spawnApple();
    } else {
      this.snake.pop();
      // Only forget the tail cell if no other segment still occupies it.
      if (!this.snake.some((c) => c.x === tail.x && c.y === tail.y)) {
        this.snakeCells.delete(tailKey);
      }
    }
  }
}
