# elsa_snakeGame

A classic Snake game (the one Google surfaces in search) built as a
zero-dependency web app. A snake moves continuously around a walled grid; steer
it to eat apples, and it grows with each one. Crash into a wall or yourself and
the game ends.

Before each game you pick:

- **Board size** — 8×8, 10×10, or 12×12.
- **Control** — **You** (arrow keys), **Random** (a placeholder heuristic), or
  **AI** (a trained model; coming soon). All three sit behind one controller
  interface, so the trained AI drops in without touching the game engine.

See [`DESIGN.md`](DESIGN.md) for the full design.

## Play

The app is static files with no build step, but ES module imports need to be
served over HTTP (not opened as `file://`):

```
python3 -m http.server 8000
# then open http://localhost:8000
```

**Controls:** arrow keys (or WASD) to steer · `Enter` to start / return to the
menu · `Esc` to quit a game back to the menu.

## Tests

The game logic in `js/state.js` is pure (no DOM) and unit-tested with Node's
built-in test runner:

```
npm test
```

## Project layout

```
index.html        canvas + menu/game-over overlays
styles.css        layout and overlay styling
js/
  main.js         game loop (fixed timestep) + screen state machine
  state.js        GameState — pure game logic
  render.js       canvas rendering (HiDPI-aware)
  input.js        keyboard handling
  ui.js           menu + HUD + game-over overlays
  config.js       tunable constants
  agent/          controllers: human, random (placeholder), ai (stub)
tests/            state and agent unit tests
```

## AI controller

The **AI** control option is **live**: `js/agent/aiAgent.js` runs a convolutional
network trained with reinforcement learning (Double DQN) — one model per board
size. Training code and docs are in [`ai/`](ai/README.md); the approach is in
[`AI_PLAN.md`](AI_PLAN.md).

- Strength (mean apples, **beats the greedy-BFS baseline on every board**):

  | Board | CNN | GreedyBFS |
  |-------|-----|-----------|
  | 8×8   | 31.4 | 24.9 |
  | 10×10 | 36.5 | 29.8 |
  | 12×12 | 38.7 | 37.8 |

- Trained on GPU with vectorized environments (`ai/train_vec.py`), exported as a
  compact binary (`models/snake-cnn-<size>.{json,bin}`, ~5.5 MB total), and run
  in-browser as a plain conv/pool/dense forward — no ML runtime dependency. A
  CPU-only feature-MLP baseline (`ai/train.py`) is also included.
- Next optimizations (model shrink, PPO, actor–learner) are in `AI_PLAN.md` §2
  and [`DESIGN.md`](DESIGN.md) §15.1.
