# Snake — Design Document (Web Application)

## 1. Overview

A classic Snake game (the same one Google surfaces in search results), built as
a **web application** that runs entirely in the browser. A snake moves
continuously around a walled grid, steering to eat apples. Each apple eaten
grows the snake and spawns a new apple. The game ends if the snake's head hits a
wall or its own body.

Before each game the player picks who controls the snake from three options:
**You** (arrow keys), **Random** (a placeholder heuristic), or **AI** (a trained
model). All three are defined behind one stable controller interface, so the AI
slot can be filled later without touching the game engine, rendering, or loop.
Until the trained model ships, the **AI** option is present but disabled in the
menu (see §5.1, §8, and §15.1).

**Target platform:** Modern desktop browsers (Chrome, Firefox, Safari, Edge).
**Stack:** Vanilla **HTML + CSS + JavaScript (ES modules)** with a `<canvas>`
for rendering.
**Why this stack:** Snake is small and has no server-side needs, so a
zero-dependency, no-build client keeps it simple, instantly loadable, and easy
to host as static files. `<canvas>` gives efficient full-grid redraws each tick;
`requestAnimationFrame` drives a smooth loop; `keydown` handles arrow input.

> **Client-only.** There is no backend. The entire game — logic, rendering,
> input — runs in the browser. It can be served as static files from any host
> (GitHub Pages, Netlify, an S3 bucket, or `python -m http.server` locally).

## 2. Goals & Non-Goals

### Goals
- Faithful recreation of the classic Snake mechanics.
- Three player-selectable board sizes chosen before the game starts.
- Player chooses **who controls the snake** — You / Random / AI — before starting.
- A pluggable controller/agent interface so the trained AI slots in beside the
  placeholder heuristic with no engine changes.
- 5–7 apples on the board at once, positions randomized.
- Continuous snake movement steered by the arrow keys (human) or the agent.
- Grow-on-eat, respawn-on-eat, and death on wall or self collision.
- Runs as a static site with no build step and no dependencies.

### Non-Goals (for this version)
- Server-side logic, accounts, or networked/multiplayer play.
- Persistent high-score storage (a `localStorage` best score is a stretch goal, §13).
- Configurable speed, obstacles, or power-ups.
- Sound and music (may be added later; see §13).
- Mobile touch / swipe controls (stretch goal, §13).

## 3. Game Rules

| Rule | Behavior |
|------|----------|
| **Board sizes** | Player chooses **8×8**, **10×10**, or **12×12** on a start screen before play begins. |
| **Walls** | The board is bounded by walls on all four edges. Playable cells are the interior grid. |
| **Apples** | A randomized count of **5–7** apples is present at all times, on empty cells. |
| **Movement** | The snake advances one cell per tick in its current direction, continuously. |
| **Steering** | Arrow keys set the direction. A 180° reversal into the neck is ignored. |
| **Eating** | When the head enters an apple's cell, that apple is consumed: the snake grows by one segment and a new apple spawns on a random empty cell (keeping the total at its current count). |
| **Wall collision** | If the head would move into a wall (off the interior grid), the game ends. |
| **Self collision** | If the head enters a cell occupied by the snake's body, the game ends. |
| **Win/loss** | There is no win state; the goal is the longest snake / highest score before a crash. |

### 3.1 Coordinate system
- The board is a grid of `N × N` cells where `N ∈ {8, 10, 12}`.
- Cell `(0, 0)` is the top-left interior cell; `(N-1, N-1)` is bottom-right.
- Walls are conceptually the border just outside `[0, N-1]` on each axis. A move
  is a wall collision when the head's next cell falls outside that range.
- Directions: `UP = {x:0, y:-1}`, `DOWN = {x:0, y:1}`, `LEFT = {x:-1, y:0}`,
  `RIGHT = {x:1, y:0}`.

### 3.2 Starting state
- Snake starts at length **3**, centered on the board, oriented `RIGHT`.
- Apple count for the round is rolled once at start: `5 + randInt(0, 2)` → 5–7.
- Initial apples are placed on random empty cells (not on the snake).

## 4. Architecture

A clean split between **pure game logic** and **browser concerns** (rendering,
input, DOM). The logic module imports nothing from the DOM, so it is
unit-testable in isolation and could be reused (e.g. Node test runner).

```
  index.html
      │ loads
      ▼
  main.js  ── controller / game loop (requestAnimationFrame) ──┐
      │                                                        │
      ├── input.js    keydown → direction / menu actions       │
      ├── agent/      controllers that pick a direction  ◄──────┤
      │   ├── index.js       Agent interface + factory          │
      │   ├── human.js       reads buffered keyboard input       │
      │   ├── randomAgent.js placeholder heuristic               │
      │   └── aiAgent.js     trained model (added later)          │
      ├── state.js    GameState: pure logic (no DOM)  ◄─────────┤
      ├── render.js   draws GameState onto <canvas>            │
      └── ui.js       menu & game-over overlays (DOM/HTML)      │
                                                                │
             screen state machine: MENU → PLAYING → GAME_OVER ──┘
```

- **`state.js` (model)** — pure logic. Holds the snake, apples, board size,
  direction, and status. Exposes `setDirection()`, `tick()`, and read-only
  accessors. No `window`/`document`/`canvas` references — fully testable.
- **`agent/` (controller strategy)** — every controller, human or computer,
  implements the same `Agent` interface (§5.1). `main.js` asks the active agent
  for a direction each tick and never branches on "is this human or AI." This is
  the seam a trained AI plugs into later.
- **`render.js` (view)** — draws a `GameState` onto the canvas 2D context.
  Stateless beyond cached colors/metrics.
- **`input.js` (view)** — attaches `keydown` listeners; maps arrow keys to
  direction requests (consumed by the human agent) and Enter/Esc to menu actions.
- **`ui.js` (view)** — the menu and game-over screens, rendered as HTML overlays
  on top of the canvas (easier to style/select than drawing menu text on canvas).
- **`main.js` (controller)** — owns the `requestAnimationFrame` loop with a
  fixed logic timestep, wires the modules together, constructs the chosen agent,
  and drives the screen state machine (menu → play → game over).

### 4.1 Proposed file layout
```
snake/
├── index.html         # canvas, overlay containers, <script type="module">
├── styles.css         # layout, menu/game-over overlays, HUD
└── js/
    ├── main.js        # controller + loop + screen state machine
    ├── state.js       # GameState, Direction/Status constants (pure logic)
    ├── render.js      # canvas rendering
    ├── input.js       # keyboard handling
    ├── ui.js          # menu & game-over overlay wiring
    ├── config.js      # constants: board sizes, tick rate, colors, apple range
    └── agent/
        ├── index.js       # Agent interface docs + createAgent() factory
        ├── human.js       # HumanAgent: returns buffered keyboard direction
        ├── randomAgent.js # RandomAgent: placeholder heuristic
        └── aiAgent.js     # AiAgent: trained model (added later; §15.1)
tests/
    ├── state.test.js  # movement, growth, collisions, apple spawn (Node/Vitest)
    └── agent.test.js  # RandomAgent never suicides when a safe move exists
```

No bundler is required: `index.html` loads `main.js` as an ES module and the
browser resolves the imports. A test runner (Vitest or `node --test`) is the
only dev-time dependency, and only for `state.js`.

## 5. Core Data Model

```js
const Direction = {
  UP:    { x: 0,  y: -1 },
  DOWN:  { x: 0,  y: 1  },
  LEFT:  { x: -1, y: 0  },
  RIGHT: { x: 1,  y: 0  },
};

const Status = { MENU: 'menu', PLAYING: 'playing', GAME_OVER: 'game_over' };

class GameState {
  size;          // 8, 10, or 12
  snake;         // Array<{x,y}>, head at snake[0], tail at snake[at end]
  snakeCells;    // Set<string> "x,y" — O(1) collision checks
  apples;        // Set<string> "x,y" apple positions
  appleCount;    // rolled 5..7 at start; held constant
  direction;     // current heading (one of Direction.*)
  pending;       // buffered next direction or null (see §6.2)
  status;        // one of Status.*
  score;         // apples eaten
}
```

Cells are keyed as `"x,y"` strings so they can live in a `Set` for O(1)
membership tests (JS `Set` can't dedupe object references by value). This keeps
self-collision and apple-placement checks constant-time instead of scanning the
snake each tick. Small helpers `key({x,y})` / `parse("x,y")` convert between
forms.

### 5.1 Agent interface (who controls the snake)

Every controller — human or computer — implements one small interface. Each
tick, the loop asks the active agent which direction to head next, given a
**read-only** view of the current game state. The agent returns a `Direction`
(or `null` to keep the current heading). The engine still validates the choice
(the reverse-into-neck guard and collision rules in §6.3 apply to agent moves
exactly as they do to human moves), so an agent can never corrupt game state —
at worst it returns a losing move.

```js
/**
 * @typedef {Object} Agent
 * @property {(view: GameView) => (Direction | null)} nextDirection
 *     Called once per tick BEFORE state.tick(). Returns the desired heading,
 *     or null to continue straight. Must not mutate `view`.
 * @property {() => void} [reset]   Optional: called when a new game starts.
 */

/**
 * GameView — the read-only snapshot an agent may inspect.
 * @typedef {Object} GameView
 * @property {number} size
 * @property {ReadonlyArray<{x,y}>} snake      // head first
 * @property {ReadonlySet<string>} snakeCells  // "x,y"
 * @property {ReadonlySet<string>} apples      // "x,y"
 * @property {Direction} direction             // current heading
 */
```

- **`HumanAgent`** — returns the buffered keyboard direction from `input.js`
  (or `null` if nothing pressed since the last tick). This routes human control
  through the *same* per-tick call, so the loop has one code path.
- **`RandomAgent`** (placeholder) — the heuristic below. Deliberately simple; it
  exists so the game is fully playable in computer mode before any AI is trained.
- **`AiAgent`** (added later) — wraps the trained model; see §15.1. Until it
  exists, `createAgent('ai')` throws and the menu keeps the **AI** option
  disabled, so the mode can't be selected.
- **`createAgent(mode)`** — factory in `agent/index.js` mapping the menu choice
  (`'human'` / `'random'` / `'ai'`) to an instance. Filling in the AI is a
  one-line change here plus the new module — no other file changes.

**RandomAgent heuristic (placeholder).** Pure random would suicide almost
immediately, making computer mode useless to watch, so the placeholder does the
minimum to survive: from the three non-reverse directions, keep only those whose
next cell is *safe* (in-bounds and not a body cell, applying the same
tail-follow rule as §6.3), and pick one uniformly at random. If no move is safe,
return `null` and let the snake crash (unavoidable). This is intentionally
**not** apple-seeking — that behavior is the job of the trained AI, not the
placeholder.

```
RandomAgent.nextDirection(view):
    candidates = the 3 directions that aren't the reverse of view.direction
    safe = [d for d in candidates if isSafe(view, head + d)]
    return safe ? randomChoice(safe) : null
```

> **Determinism note for the future AI.** `nextDirection` is a pure function of
> `GameView` (plus, later, model weights). That keeps agents easy to unit-test
> and lets a trained policy be dropped in as just another `Agent` — whether it
> runs in-browser (e.g. ONNX Runtime Web / TensorFlow.js) or is precomputed. The
> `GameView` fields are exactly the observation an RL policy needs; if the model
> wants a tensor encoding, that conversion lives inside the AI agent, not in the
> engine. See §15.1.

## 6. Game Loop & Movement

### 6.1 Fixed timestep over `requestAnimationFrame`
`requestAnimationFrame` fires ~60×/sec and gives a high-resolution timestamp.
The snake must move on a *fixed cadence* independent of the display refresh
rate, so the loop accumulates elapsed time and runs one logic `tick()` per
`stepInterval` — the ms-per-tick value chosen on the menu from `SPEEDS`
(Slow 200 / Medium 130 / Fast 80; ~5 / 7.7 / 12.5 cells/sec), set per game in
`config.js`. Rendering happens once per animation frame.

```js
function frame(now) {
  const dt = now - last; last = now;
  accumulator += dt;
  while (accumulator >= stepInterval) {
    const dir = agent.nextDirection(state.view());  // human OR computer
    if (dir) state.setDirection(dir);               // engine still validates it
    state.tick();               // one cell of movement + collision + eating
    accumulator -= stepInterval;
  }
  render(ctx, state);
  requestAnimationFrame(frame);
}
```

The agent is consulted **inside** the fixed-timestep loop (once per logic tick),
not once per animation frame — so a fast machine doesn't poll the agent more
often than the snake actually moves. The single `agent.nextDirection(...)` call
is the only place control differs between human and computer, and even that
difference is hidden behind the interface.

- Using a fixed timestep keeps movement speed consistent regardless of a 60 Hz
  vs 144 Hz monitor.
- When the tab is backgrounded, browsers throttle `requestAnimationFrame`; on
  return, `dt` could be huge. Clamp `accumulator` to a small maximum (e.g. a few
  steps) so the snake doesn't teleport across many cells at once.

### 6.2 Direction buffering
Key presses set a `pending` direction rather than mutating heading immediately.
`tick()` applies `pending` at the start of the step. This prevents a fast
double-tap within one tick from letting the snake reverse into itself (e.g.
`RIGHT → UP → LEFT` before a single move). A pending direction that is the exact
reverse of the current heading is rejected.

### 6.3 `tick()` algorithm
```
1. If pending is set and not the reverse of `direction`: direction = pending.
2. Compute nextHead = head + direction
3. If nextHead is outside [0, size-1] on either axis:
       status = GAME_OVER; return          # wall collision
4. eating = apples has key(nextHead)
5. If not eating and snakeCells has key(nextHead), excluding the tail cell that
   is about to move away:
       status = GAME_OVER; return          # self collision
6. Unshift nextHead onto snake; add to snakeCells.
7. If eating:
       delete nextHead from apples
       score += 1
       spawnApple()                         # keep total at appleCount
   Else:
       pop tail from snake; delete from snakeCells   # move without growing
```

> **Tail-follow subtlety (step 5):** the cell the tail currently occupies is
> vacated this same tick, so moving the head into it is *not* a self-collision
> (unless the snake just ate and the tail didn't move). The check excludes the
> tail cell when not eating.

## 7. Apple Spawning

```
spawnApple():
    free = all interior cells − snakeCells − apples
    if free is empty: return          # board full (extreme edge case)
    choose a uniformly random cell from free
    add its key to apples
```
- Computed against `snakeCells` and existing `apples` so an apple never lands on
  the snake or another apple.
- On the rare full-board case, spawning is skipped rather than looping forever.
- Initial placement calls `spawnApple()` `appleCount` times.

## 8. Screens & Flow

```
        ┌──────────────────────────┐  size+speed+ctrl  ┌────────────┐
        │          MENU            │  chosen, + Enter   │  PLAYING   │
        │  Board:   8 / 10 / 12     │ ─────────────────> │            │
        │  Speed:   Slow/Med/Fast   │                   └─────┬──────┘
        │  Control: You/Random/AI   │                         │ crash
        └──────────────────────────┘                         v
              ^                                        ┌────────────┐
              │            press Enter                 │ GAME_OVER  │
              └──────────────────────────────         │ score shown│
                                                       └────────────┘
```

- **Menu:** An HTML overlay with the title and three choices:
  - **Board size** — three buttons (8×8 / 10×10 / 12×12), also number keys 1/2/3.
  - **Speed** — three buttons (Slow / Medium / Fast) mapping to the logic-tick
    interval (§11); lower interval = faster snake. Defaults to Medium.
  - **Control** — three buttons picking who steers the snake:
    - **You** — arrow-key control via `HumanAgent`. Default selection.
    - **Random** — the placeholder `RandomAgent` heuristic (§5.1).
    - **AI** — the trained model (feature-MLP DQN), now **live** and selectable
      (`AI_ENABLED = true`). Runs `aiAgent.js`; see §15.1. (It renders disabled
      only if `AI_ENABLED` is turned back off.)

  Selections are made by click or keyboard, then **Enter** starts. The chosen
  size drives grid dimensions and canvas size; the chosen speed sets the loop's
  per-tick interval for the round; the chosen control mode string is passed to
  `createAgent(mode)` (§5.1) to build the agent for the round.
- **Playing:** Canvas shows the live grid; an HTML HUD strip shows the current
  score and the active control mode (e.g. "Random"). Overlays are hidden. In any
  computer mode, arrow keys are ignored for steering (Esc still returns to menu).
- **Game Over:** An HTML overlay with "Game Over", the final score, and buttons
  to play again (returns to menu, Enter) or the menu directly. Restarting
  re-rolls the apple count and rebuilds the board.

Screens are managed as a small state machine in `main.js`; overlays are shown /
hidden by toggling a CSS class rather than tearing down the canvas.

## 9. Rendering

- The canvas backing size is `size * CELL_PX` square, plus an HTML HUD strip
  above it for the score. `CELL_PX` is fixed (e.g. **40 px**), so 8×8 → 320 px
  board, 12×12 → 480 px. The canvas is resized when a board size is chosen.
- **HiDPI:** set the canvas `width/height` attributes to `cssPixels * devicePixelRatio`
  and scale the context so grid lines stay crisp on retina displays.
- Draw order per tick: clear → background → grid lines → apples → snake body →
  snake head (distinct shade so heading is readable).
- The whole board is small (≤144 cells), so a full redraw each tick is trivially
  cheap; no dirty-rect optimization is needed.
- Palette (in `config.js`): dark background, green snake (brighter head), red
  apples, muted grid lines. HUD/overlay colors live in `styles.css`.

## 10. Input Handling

- A single `keydown` listener on `window`.
- **Playing (human control):** Arrow keys (and optionally WASD) set the buffered
  direction that `HumanAgent` returns on the next tick. `Esc` returns to the menu.
- **Playing (computer control):** Steering keys are ignored — the active agent
  supplies directions. `Esc` still returns to the menu. Optionally, keys can
  reserve a future "take over" hook, but that is out of scope for this version.
- **Menu / Game Over:** 1–3 (or click) select a size; a key/click toggles the
  You/Computer control mode; Enter confirms/starts.
- Call `event.preventDefault()` for arrow keys so the page doesn't scroll.
- Ignore auto-repeat spam gracefully — buffering (§6.2) already collapses
  multiple presses per tick to the last valid one.

## 11. Configuration Constants (`config.js`)

| Constant | Value | Notes |
|----------|-------|-------|
| `BOARD_SIZES` | `[8, 10, 12]` | Selectable on the menu. |
| `APPLE_MIN`, `APPLE_MAX` | `5`, `7` | Rolled once per round. |
| `START_LENGTH` | `3` | Initial snake length. |
| `SPEEDS` | `{slow:200, medium:130, fast:80}` | ms per logic tick; menu-selectable. Lower = faster. |
| `DEFAULT_SPEED` | `'medium'` | Pre-selected speed on the menu. |
| `CELL_PX` | `40` | Pixel size of a grid cell. |
| `MAX_CATCHUP_STEPS` | `5` | Clamp on `accumulator` after tab throttling. |
| `CONTROL_MODES` | `['human', 'random', 'ai']` | Menu options; maps to `createAgent`. |
| `DEFAULT_CONTROL` | `'human'` | Pre-selected control mode on the menu. |
| `AI_ENABLED` | `true` | Gates the AI menu button; now on — the trained model ships (§15.1). |

## 12. Edge Cases

- **Reverse-into-neck:** blocked by the pending-direction reverse check (§6.2).
- **Simultaneous eat + tail-follow:** eating suppresses the tail pop, so the
  snake grows; self-collision check accounts for this (§6.3).
- **Apple on tail cell being vacated:** allowed — the head can eat an apple that
  spawned where the tail is leaving; treated as a normal eat.
- **Board nearly full at high length:** `spawnApple` skips gracefully when no
  free cell exists; the round continues with fewer apples.
- **Rapid key mashing:** only the last valid pending direction per tick applies.
- **Backgrounded tab / rAF throttling:** `accumulator` is clamped
  (`MAX_CATCHUP_STEPS`) so returning to the tab doesn't teleport the snake.
- **Page scroll on arrow keys:** suppressed via `preventDefault()`.
- **Window resize:** the board is a fixed pixel size; the layout centers it and
  does not rescale mid-game.

## 13. Testing Strategy

Because `state.js` has no DOM dependency, the core is unit-tested directly with
Vitest or `node --test`:

- **Movement:** head advances by the direction delta each tick; tail follows.
- **Growth:** eating increments length and score and suppresses tail pop.
- **Wall death:** a tick that steps off each of the four edges ends the game.
- **Self death:** a coiled snake stepping into its body ends the game; stepping
  into the vacating tail cell does **not**.
- **Reverse guard:** a reverse-direction request never changes heading — whether
  it originates from a human key or an agent's returned direction.
- **Apple invariants:** spawned apples are always on empty interior cells; total
  count is preserved after an eat; initial count is within 5–7.
- **RandomAgent safety:** given a `GameView` with at least one safe move,
  `nextDirection` never returns a direction that steps into a wall or body;
  given a fully boxed-in view it returns `null`. Testable because the agent is a
  pure function of `GameView` (§5.1).

Rendering, input, and overlays are covered by lightweight manual/smoke testing
in the browser.

## 14. Deployment

- Static hosting: commit `index.html`, `styles.css`, and `js/`; deploy to GitHub
  Pages, Netlify, Vercel, or any static host — no build step.
- Local dev: `python -m http.server` (or any static server) from the project
  root, then open `http://localhost:8000`. A static server is needed rather than
  `file://` because ES module imports require HTTP.

## 15. Future Extensions

### 15.1 Trained AI controller — **implemented (simple milestone)**
The control-mode seam (§5.1) exists specifically so a trained model can replace
`RandomAgent` with **no changes to the engine, loop, rendering, or menu wiring**.
This is now live: `AI_ENABLED = true`, the **AI** menu button is selectable, and
`agent/aiAgent.js` runs a trained network.

**What ships today** — a **grid + CNN trained with Double DQN** (PyTorch, in
`ai/`), which **beats the greedy-BFS baseline on every board**:

| Board | CNN | GreedyBFS | Random | Model size |
|-------|-----|-----------|--------|------------|
| 8×8   | 31.4 | 24.9 | 8.5 | 1.3 MB |
| 10×10 | 36.5 | 29.8 | 8.5 | 1.9 MB |
| 12×12 | 38.7 | 37.8 | 7.3 | 2.6 MB |

- Architecture (`ai/cnn.py SnakeCNN`): three 3×3 conv layers (→64 channels) with
  a 2×2 max-pool before the flatten — the pool keeps full channel capacity while
  cutting the dense matrix 4×, so each model is ~1–2.6 MB (5.5 MB total) instead
  of ~21 MB.
- Trained in Python on the GPU with **vectorized environments** (`ai/train_vec.py`,
  ~37× the single-env throughput), one model **per board size** (the flatten head
  is size-specific), exported as a compact binary — a tiny `.json` op manifest +
  a float32 `.bin` (`models/snake-cnn-<size>.{json,bin}`).
- `aiAgent.js` picks the model for the active board size, encodes `GameView` into
  the **same 4-channel grid** as `ai/features.py grid_encode` (parity is critical
  — AI_PLAN.md §3.2), and runs a plain-JS conv/pool/dense forward — no runtime
  dependency needed. Verified numerically against torch (diff ~2e-5, argmax
  identical). The earlier feature-MLP remains in `ai/` as the simple baseline.

**Further upgrades** (AI_PLAN.md §2): a size-agnostic head (global-average-pool +
coordinate channels → one model for all boards); Dueling / prioritized replay;
PPO; and a decoupled actor–learner for full GPU saturation. All stay contained in
`aiAgent.js` + the `ai/` package.

Because `GameView` already carries exactly the observation a policy needs (board
size, snake, apples, heading) and `nextDirection` is a pure function, the same
agent code can be exercised headlessly in training/eval and in the browser. The
reverse-guard and collision validation in the engine remain the safety net, so
a mis-trained model can lose but never break the game.

### 15.2 Other ideas
- Touch / swipe controls for mobile, plus responsive canvas sizing.
- Selectable speed / difficulty levels (also useful for watching the AI).
- Persistent best score per board size and control mode via `localStorage`.
- Sound effects (eat, crash) via the Web Audio API and a pause key.
- Wrap-around ("no walls") mode as an alternate ruleset.
- "Take over" hotkey to switch from AI to human mid-game (the loop already
  supports swapping the active agent between ticks).
