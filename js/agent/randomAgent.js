// RandomAgent: placeholder heuristic. See DESIGN.md §5.1.
//
// Deliberately simple — it exists so computer mode is playable before any AI is
// trained. From the three non-reverse directions it keeps only the SAFE ones
// (in-bounds and not a body cell, applying the same tail-follow rule as the
// engine) and picks one uniformly at random. It is intentionally NOT
// apple-seeking — that is the job of the trained AI, not the placeholder.

import { Direction, key } from '../state.js';

const ALL = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT];
const isOpposite = (a, b) => a.x === -b.x && a.y === -b.y;

export class RandomAgent {
  constructor(rng = Math.random) {
    this.rng = rng;
  }

  isSafe(view, cell) {
    // In bounds?
    if (cell.x < 0 || cell.x >= view.size || cell.y < 0 || cell.y >= view.size) {
      return false;
    }
    // The tail cell is vacated this tick, so stepping into it is safe (the snake
    // is not eating on a random-heuristic move check — it only would if an apple
    // sat there, in which case it's still a legal, non-fatal move).
    const tail = view.snake[view.snake.length - 1];
    if (cell.x === tail.x && cell.y === tail.y) return true;
    return !view.snakeCells.has(key(cell));
  }

  nextDirection(view) {
    const head = view.snake[0];
    const candidates = ALL.filter((d) => !isOpposite(d, view.direction));
    const safe = candidates.filter((d) =>
      this.isSafe(view, { x: head.x + d.x, y: head.y + d.y })
    );
    if (safe.length === 0) return null; // boxed in — unavoidable crash
    return safe[Math.floor(this.rng() * safe.length)];
  }

  reset() {}
}
