// AiAgent: trained-model controller. See DESIGN.md §15.1 and AI_PLAN.md.
//
// Runs a trained CNN (PyTorch, ai/train_vec.py) directly in the browser via a
// plain-JS conv/dense forward — the net is small enough that no onnxruntime is
// needed. Weights load from a compact binary (models/snake-cnn-<size>.bin) plus
// a tiny JSON manifest of layer shapes (ai/cnn.py export_binary); a separate
// model is trained per board size because the CNN's dense head is size-specific.
//
// PARITY (AI_PLAN.md §3.2): gridEncode() mirrors ai/features.py grid_encode(),
// and the forward pass mirrors ai/cnn.py SnakeCNN — both verified numerically
// against torch before shipping. All tensors are kept as flat Float32Array with
// (C,H,W) / (out,in) indexing, matching torch's memory order.

const base = (size) => `models/snake-cnn-${size}`;

// Relative-action turns, matching ai/env.py (y grows downward).
const turnLeft = (d) => ({ x: d.y, y: -d.x });
const turnRight = (d) => ({ x: -d.y, y: d.x });
const relToAbs = (heading, action) =>
  action === 0 ? heading : action === 1 ? turnLeft(heading) : turnRight(heading);

export class AiAgent {
  constructor(deps = {}) {
    this.size = deps.size || 8;
    this.net = null;
    this.load(this.size).catch((e) => console.error('AiAgent: failed to load model', e));
  }

  async load(size) {
    const [manifest, buf] = await Promise.all([
      fetch(`${base(size)}.json`).then((r) => r.json()),
      fetch(`${base(size)}.bin`).then((r) => r.arrayBuffer()),
    ]);
    const f32 = new Float32Array(buf); // browsers are little-endian
    let off = 0;
    const layers = manifest.layers.map((l) => {
      const wlen = l.shape.reduce((a, b) => a * b, 1);
      const w = f32.subarray(off, off + wlen); off += wlen;
      const blen = l.shape[0];
      const b = f32.subarray(off, off + blen); off += blen;
      return { type: l.type, shape: l.shape, w, b };
    });
    this.net = { layers };
  }

  nextDirection(view) {
    if (!this.net) return this.safeFallback(view);
    const q = this.forward(this.gridEncode(view), view.size);
    let best = 0;
    for (let i = 1; i < q.length; i++) if (q[i] > q[best]) best = i;
    return relToAbs(view.direction, best);
  }

  // ---- grid encoding — mirror of ai/features.py grid_encode() ------------
  // Flat Float32Array of shape (C,H,W): channels [body, head, apples, walls].
  gridEncode(view) {
    const n = view.size;
    const HW = n * n;
    const g = new Float32Array(4 * HW); // walls channel (index 3) stays zero
    const bodyOff = 0, headOff = HW, appleOff = 2 * HW;
    const h = view.snake[0];
    g[headOff + h.y * n + h.x] = 1;
    for (let i = 1; i < view.snake.length; i++) {
      const c = view.snake[i];
      g[bodyOff + c.y * n + c.x] = 1;
    }
    for (const k of view.apples) {
      const [ax, ay] = k.split(',').map(Number);
      g[appleOff + ay * n + ax] = 1;
    }
    return g;
  }

  // ---- forward pass — mirror of ai/cnn.py SnakeCNN -----------------------
  // Layers run in order: conv (3x3, pad 1, ReLU) / pool (2x2 max) then dense
  // (ReLU on all but the last). Tracks the spatial size `n`, which shrinks at
  // each pool.
  forward(grid, n) {
    const layers = this.net.layers;
    let a = grid;                 // (cin, n, n) flat
    let cin = 4;
    let i = 0;
    for (; i < layers.length && layers[i].type !== 'dense'; i++) {
      const L = layers[i];
      if (L.type === 'conv') {
        const cout = L.shape[0];
        a = this.conv3x3ReLU(a, cin, n, L.w, L.b, cout);
        cin = cout;
      } else if (L.type === 'pool') {
        a = this.maxPool(a, cin, n, L.k);
        n = Math.floor(n / L.k);
      }
    }
    // `a` is (cin, n, n) row-major == torch flatten order.
    let vec = a;
    const denses = layers.slice(i);
    for (let li = 0; li < denses.length; li++) {
      const { w, b, shape } = denses[li]; // shape [out, in]
      const outN = shape[0], inN = shape[1];
      const out = new Float32Array(outN);
      for (let o = 0; o < outN; o++) {
        let s = b[o];
        const wb = o * inN;
        for (let j = 0; j < inN; j++) s += vec[j] * w[wb + j];
        out[o] = li < denses.length - 1 ? (s > 0 ? s : 0) : s;
      }
      vec = out;
    }
    return vec;
  }

  // k x k max-pool, stride k (torch default), floor mode. (c,n,n) -> (c, n/k, n/k).
  maxPool(inp, c, n, k) {
    const no = Math.floor(n / k);
    const out = new Float32Array(c * no * no);
    for (let ch = 0; ch < c; ch++) {
      const inBase = ch * n * n, outBase = ch * no * no;
      for (let y = 0; y < no; y++) {
        for (let x = 0; x < no; x++) {
          let m = -Infinity;
          for (let dy = 0; dy < k; dy++) {
            const row = inBase + (y * k + dy) * n;
            for (let dx = 0; dx < k; dx++) {
              const v = inp[row + x * k + dx];
              if (v > m) m = v;
            }
          }
          out[outBase + y * no + x] = m;
        }
      }
    }
    return out;
  }

  // 3x3 conv, stride 1, zero-pad 1, ReLU. w flat (Cout,Cin,3,3), b (Cout).
  conv3x3ReLU(inp, cin, n, w, b, cout) {
    const HW = n * n;
    const out = new Float32Array(cout * HW);
    for (let o = 0; o < cout; o++) {
      const oBase = o * HW;
      const woBase = o * cin * 9;
      for (let y = 0; y < n; y++) {
        for (let x = 0; x < n; x++) {
          let s = b[o];
          for (let i = 0; i < cin; i++) {
            const inBase = i * HW;
            const wiBase = woBase + i * 9;
            for (let ky = 0; ky < 3; ky++) {
              const iy = y + ky - 1;
              if (iy < 0 || iy >= n) continue;
              const rowBase = inBase + iy * n;
              const wkBase = wiBase + ky * 3;
              for (let kx = 0; kx < 3; kx++) {
                const ix = x + kx - 1;
                if (ix < 0 || ix >= n) continue;
                s += inp[rowBase + ix] * w[wkBase + kx];
              }
            }
          }
          out[oBase + y * n + x] = s > 0 ? s : 0;
        }
      }
    }
    return out;
  }

  // Safe move used only while the model is still loading.
  safeFallback(view) {
    const head = view.snake[0];
    for (const action of [0, 1, 2]) {
      const d = relToAbs(view.direction, action);
      const cell = { x: head.x + d.x, y: head.y + d.y };
      const inB = cell.x >= 0 && cell.x < view.size && cell.y >= 0 && cell.y < view.size;
      const tail = view.snake[view.snake.length - 1];
      const onTail = cell.x === tail.x && cell.y === tail.y;
      if (inB && (onTail || !view.snakeCells.has(`${cell.x},${cell.y}`))) return d;
    }
    return view.direction;
  }

  reset() {}
}
