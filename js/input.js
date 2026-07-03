// InputHandler: keyboard handling. See DESIGN.md §10.
//
// Steering keys are buffered and consumed once per tick by HumanAgent. Menu /
// game-over actions (Enter, Esc) are dispatched through callbacks so main.js
// owns the screen state machine.

import { Direction } from './state.js';

const KEY_TO_DIR = {
  ArrowUp: Direction.UP,
  ArrowDown: Direction.DOWN,
  ArrowLeft: Direction.LEFT,
  ArrowRight: Direction.RIGHT,
  w: Direction.UP,
  s: Direction.DOWN,
  a: Direction.LEFT,
  d: Direction.RIGHT,
};

export class InputHandler {
  constructor() {
    this.buffered = null; // last steering direction since the previous tick
    this.handlers = { enter: null, escape: null };
    window.addEventListener('keydown', (e) => this.onKeyDown(e));
  }

  on(action, fn) {
    this.handlers[action] = fn;
  }

  onKeyDown(e) {
    if (e.key === 'Enter') {
      this.handlers.enter?.();
      return;
    }
    if (e.key === 'Escape') {
      this.handlers.escape?.();
      return;
    }
    const dir = KEY_TO_DIR[e.key];
    if (dir) {
      e.preventDefault(); // stop arrow keys from scrolling the page
      this.buffered = dir;
    }
  }

  // Consume the buffered direction (returns null if nothing pressed since last call).
  consumeDirection() {
    const d = this.buffered;
    this.buffered = null;
    return d;
  }
}
