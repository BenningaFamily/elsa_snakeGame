// AiAgent: trained-model controller. See DESIGN.md §15.1 and AI_PLAN.md.
//
// Runs a small MLP (trained in Python, ai/train.py) directly in the browser —
// the network is tiny, so a few matmuls beat pulling in a runtime like
// onnxruntime-web. Weights are fetched as JSON (exported by DQNAgent.export_json).
//
// PARITY (AI_PLAN.md §3.2): the feature encoding below MUST match ai/features.py
// `featurize` byte-for-byte — same 11 features, same order — or the agent sees a
// different world than it trained on. Keep the two in lockstep.

const MODEL_URL = 'models/snake-dqn.json';

// Relative-action turns, matching ai/env.py (y grows downward).
const turnLeft = (d) => ({ x: d.y, y: -d.x });
const turnRight = (d) => ({ x: -d.y, y: d.x });
const relToAbs = (heading, action) =>
  action === 0 ? heading : action === 1 ? turnLeft(heading) : turnRight(heading);

export class AiAgent {
  constructor() {
    this.model = null;
    // Fetch weights asynchronously; until they arrive, nextDirection() falls
    // back to a safe move so the snake doesn't die during the brief load.
    fetch(MODEL_URL)
      .then((r) => r.json())
      .then((m) => { this.model = m; })
      .catch((e) => console.error('AiAgent: failed to load model', e));
  }

  nextDirection(view) {
    if (!this.model) return this.safeFallback(view);
    const q = this.forward(this.featurize(view));
    let best = 0;
    for (let i = 1; i < q.length; i++) if (q[i] > q[best]) best = i;
    return relToAbs(view.direction, best);
  }

  // ---- feature encoding — mirror of ai/features.py featurize() -----------
  unsafe(view, cell) {
    if (cell.x < 0 || cell.x >= view.size || cell.y < 0 || cell.y >= view.size) return true;
    const tail = view.snake[view.snake.length - 1];
    if (cell.x === tail.x && cell.y === tail.y) return false;
    return view.snakeCells.has(`${cell.x},${cell.y}`);
  }

  featurize(view) {
    const head = view.snake[0];
    const d = view.direction;
    const left = turnLeft(d);
    const right = turnRight(d);
    const cell = (delta) => ({ x: head.x + delta.x, y: head.y + delta.y });

    const dangerStraight = this.unsafe(view, cell(d)) ? 1 : 0;
    const dangerLeft = this.unsafe(view, cell(left)) ? 1 : 0;
    const dangerRight = this.unsafe(view, cell(right)) ? 1 : 0;

    const dirUp = d.x === 0 && d.y === -1 ? 1 : 0;
    const dirDown = d.x === 0 && d.y === 1 ? 1 : 0;
    const dirLeft = d.x === -1 && d.y === 0 ? 1 : 0;
    const dirRight = d.x === 1 && d.y === 0 ? 1 : 0;

    let foodUp = 0, foodDown = 0, foodLeft = 0, foodRight = 0;
    if (view.apples.size > 0) {
      let bx = 0, by = 0, bestDist = Infinity;
      for (const k of view.apples) {
        const [ax, ay] = k.split(',').map(Number);
        const dist = Math.abs(ax - head.x) + Math.abs(ay - head.y);
        if (dist < bestDist) { bestDist = dist; bx = ax; by = ay; }
      }
      foodUp = by < head.y ? 1 : 0;
      foodDown = by > head.y ? 1 : 0;
      foodLeft = bx < head.x ? 1 : 0;
      foodRight = bx > head.x ? 1 : 0;
    }

    return [
      dangerStraight, dangerLeft, dangerRight,
      dirUp, dirDown, dirLeft, dirRight,
      foodUp, foodDown, foodLeft, foodRight,
    ];
  }

  // ---- MLP forward pass (ReLU hidden, linear output) ---------------------
  forward(x) {
    const { W, b } = this.model;
    let a = x;
    for (let layer = 0; layer < W.length; layer++) {
      const Wl = W[layer];      // shape (fan_in, fan_out)
      const bl = b[layer];      // shape (fan_out)
      const out = new Array(bl.length).fill(0);
      for (let k = 0; k < bl.length; k++) {
        let s = bl[k];
        for (let j = 0; j < a.length; j++) s += a[j] * Wl[j][k];
        out[k] = layer < W.length - 1 ? Math.max(0, s) : s; // ReLU except last
      }
      a = out;
    }
    return a;
  }

  // Safe move used only while the model is still loading.
  safeFallback(view) {
    const head = view.snake[0];
    for (const action of [0, 1, 2]) {
      const d = relToAbs(view.direction, action);
      if (!this.unsafe(view, { x: head.x + d.x, y: head.y + d.y })) return d;
    }
    return view.direction;
  }

  reset() {}
}
