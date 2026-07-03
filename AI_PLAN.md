# Building the Snake AI — Approach & Design

This document proposes how to build the trained AI controller that plugs into
the existing `AiAgent` slot (`js/agent/aiAgent.js`, see `DESIGN.md` §15.1). It
covers the choice of algorithm, network, state/action representation, reward
design, training infrastructure, and a phased plan.

It assumes the reinforcement-learning background this repo started from
(cliff-walking / tabular Q-learning) and builds directly on it.

---

## 1. Framing: Snake as a reinforcement-learning problem

Snake is a **Markov Decision Process (MDP)** — the same framework as
cliff-walking, just much larger:

| MDP piece | Cliff-walking | Snake (this game) |
|-----------|---------------|-------------------|
| **State** | which of ~48 cells the agent is on | the full board: snake body + head + 5–7 apples + walls |
| **Action** | up/down/left/right | turn left / straight / right (see §4) |
| **Reward** | −1 per step, −100 for the cliff | + for eating, − for dying, small step cost (§5) |
| **Transition** | deterministic grid move | deterministic: move, maybe eat & grow, maybe die |
| **Goal** | reach the goal cheaply | eat as many apples as possible without crashing |

The key difference is **state-space size**. Cliff-walking has ~48 states, so a
lookup **table** `Q[state][action]` works. In Snake, the state is the whole
board configuration. On an 8×8 board the number of reachable configurations is
astronomically large (every arrangement of a growing snake plus apple
positions), and it's even bigger on 10×10 and 12×12. **A Q-table cannot
represent this.**

> **The one-sentence bridge from cliff-walking:** we keep Q-learning's idea —
> learn the value of taking an action in a state — but replace the *table* with
> a *neural network* that takes the board as input and outputs action values.
> That is exactly what a **Deep Q-Network (DQN)** is.

### 1.1 This game's specific quirks (they matter for design)

- **Multiple apples (5–7 at once), not one.** Most Snake tutorials assume a
  single apple. Here, food is never far away, so **foraging is easy and
  *survival* is the hard part** — the agent must avoid boxing itself in as it
  grows. This shifts emphasis toward collision-avoidance and space management,
  and changes reward shaping (§5).
- **Three board sizes (8/10/12).** Start with one fixed size; generalize later
  (§6).
- **A clean seam already exists.** `AiAgent.nextDirection(view)` is a **pure
  function** of a read-only `GameView` (`{size, snake, snakeCells, apples,
  direction}`). That is precisely a policy `π(action | state)`. The AI is a drop
  in replacement — no engine changes (DESIGN §15.1).

---

## 2. Recommended approach

**Primary recommendation: Deep Q-Network (DQN) over a small convolutional
network (CNN) that reads the board as a multi-channel grid.**

Why this first:
- It's the **smallest conceptual step from the tabular Q-learning** already
  familiar here — same update rule, function approximator instead of a table.
- **Discrete, tiny action set** (3 actions) — DQN's sweet spot.
- The environment is **fully observable and deterministic** — no need for
  recurrence or belief states.
- The board is small, so a CNN is **cheap to train** (minutes-to-hours on a CPU
  or a modest GPU) and cheap to run in the browser.

**Upgrade path once DQN works: PPO (Proximal Policy Optimization).** PPO is a
policy-gradient / actor-critic method that is the current default for this kind
of problem — often more stable and sample-efficient than DQN, and it directly
learns a policy (useful if we later want stochastic play). Treat DQN as the
learning-friendly first milestone and PPO as the "make it strong" follow-up.

**Standard DQN improvements** to layer in incrementally (each is a small change):
Double DQN (reduces value over-estimation), Dueling DQN (separates state-value
from action-advantage), and Prioritized Experience Replay (learn more from rare,
important transitions). Add them one at a time and measure.

**Non-goals / what to skip:** recurrent nets (game is fully observable),
model-based RL, and anything exotic. Not needed for a board this size.

---

## 3. State representation (the most important design choice)

The network needs the board as numbers. Two viable encodings — I recommend
building the simple one first, then moving to the grid one.

### 3.1 Option A — hand-crafted feature vector → MLP (fast first milestone)

A short vector of features fed to a small multilayer perceptron (MLP). Classic
"DQN Snake" tutorials use ~11 features; for multi-apple we extend slightly:

- Danger straight / left / right (would the next cell in each relative direction
  kill us?) — 3 values.
- Current heading as one-hot — 4 values.
- Direction to the **nearest** apple (sign of dx, dy, or a small bearing) — 2–4
  values.
- Optionally: normalized distance to nearest apple, snake length, free space
  around the head.

**Pros:** tiny, trains in minutes, easy to debug, great for a first learning
signal. **Cons:** hand-crafted, throws away most of the board, and struggles
with long-snake "don't trap yourself" reasoning. It will plateau — that's
expected; use it to validate the whole pipeline.

### 3.2 Option B — multi-channel grid → CNN (recommended target)

Encode the board as a small image with one binary channel per entity, shape
`(C, H, W)`:

| Channel | Contents |
|---------|----------|
| 0 | snake **body** cells (1/0) |
| 1 | snake **head** cell (1/0) |
| 2 | **apples** (1/0) |
| 3 | **walls / out-of-bounds** (1 on the border, or a constant frame) |

(Optionally add a "tail" channel, or a channel encoding heading, but start with
these four.) A small CNN then reads this stack. Because the input **is** the
board, the network can learn spatial reasoning (enclosed regions, escape routes)
that the feature vector can't.

**This is the recommended representation for the real model.** It also
generalizes naturally across board sizes if the CNN is fully convolutional (§6).

> **Parity rule (critical):** the encoding in the Python training environment
> and the encoding in `aiAgent.js` at inference time **must be byte-for-byte
> identical** — same channel order, same orientation, same normalization.
> Any mismatch means the browser agent sees a different world than the one it
> trained on and plays badly. Keep one written spec and test both against it.

---

## 4. Action space

Use **3 relative actions**: `{ turn-left, go-straight, turn-right }`, decoded
against the current heading into an absolute `Direction`.

Why relative rather than 4 absolute directions (up/down/left/right):
- The 180° reverse is **impossible** in this game (the engine ignores it), so an
  absolute "reverse" action is wasted and confusing to learn. Relative actions
  are all always legal.
- Smaller action space = easier learning.
- It's rotation-invariant, which helps the network generalize.

`aiAgent.js` maps the chosen relative action back to a `Direction` and returns
it; the engine still validates the move, so a bad choice can lose but never
corrupt state (DESIGN §5.1).

---

## 5. Reward design

Rewards define what "good play" means. Start **minimal**, add shaping only if
learning stalls — over-shaping is the #1 cause of weird behavior (e.g. endless
safe circling).

**Baseline reward:**

| Event | Reward |
|-------|--------|
| Eat an apple | **+10** |
| Die (wall or self) | **−10** |
| Each step survived | **−0.01** (tiny) — discourages aimless wandering |

The small per-step penalty pushes the agent toward *efficient* apple-getting
rather than stalling. Keep it small so survival still dominates.

**Optional potential-based shaping** (only if the sparse reward learns too
slowly): add `γ·Φ(s') − Φ(s)` where `Φ` rewards being closer to the nearest
apple (e.g. `Φ = −distance_to_nearest_apple`). Potential-based shaping is
**provably safe** — it speeds learning without changing the optimal policy.
With multiple apples, "nearest apple" is the natural potential.

**Watch out for (reward-hacking symptoms):**
- Agent circles forever without eating → step penalty too small, or apple reward
  too low relative to survival.
- Agent rushes food and dies young → death penalty too small, or shaping too
  aggressive toward apples.
Tune by watching *behavior*, not just the reward curve.

---

## 6. Handling the three board sizes

Don't solve this on day one. Recommended order:

1. **Fix one size first (8×8).** Fastest iteration; get the whole pipeline
   working end-to-end before adding variation.
2. Then pick one of:
   - **Separate model per size** — simplest, three checkpoints, `aiAgent.js`
     loads the one matching `view.size`. Totally fine and easy to reason about.
   - **One fully-convolutional model** — a CNN with only conv/pooling layers
     (no fixed-size dense layer tied to H×W) can accept 8×8, 10×10, or 12×12 and
     share learning across them. More elegant, slightly more work. Train by
     sampling board sizes during training.
   - **Pad to 12×12** — always encode into a 12×12 grid with a wall frame for
     smaller boards; one model, fixed input. Simple, wastes a little capacity.

Start with **separate-model-per-size** (or just 8×8) and only invest in the
fully-conv path if you want the single-model elegance.

---

## 7. Network architecture (concrete starting points)

Small networks are enough — this is not ImageNet.

**MLP (for the feature-vector option, §3.1):**
```
input (≈12 features)
  → Dense(128) + ReLU
  → Dense(128) + ReLU
  → Dense(3)              # Q-value per relative action
```

**CNN (for the grid option, §3.2) — recommended:**
```
input (4, H, W)
  → Conv 3×3, 32 filters, pad=1 + ReLU
  → Conv 3×3, 64 filters, pad=1 + ReLU
  → Conv 3×3, 64 filters, pad=1 + ReLU
  → Flatten (or Global Average Pool for size-agnostic)
  → Dense(256) + ReLU
  → Dense(3)              # Q-values (DQN) or logits+value (PPO)
```
Use `padding=1` so the board resolution is preserved through conv layers. For
the fully-convolutional / size-agnostic variant, replace `Flatten → Dense` with
**global average pooling** so the head doesn't depend on H×W.

For **PPO**, the same trunk feeds two heads: a **policy head** (3 action logits)
and a **value head** (1 scalar).

---

## 8. Training infrastructure — how this connects to the game

The game engine is JavaScript, but **training should happen in Python** (mature
RL tooling, GPUs, speed). The path:

```
  Python: reimplement the game as a Gym env  ──►  train (PyTorch)  ──►  export weights
                                                                          (ONNX file)
                                                                             │
  Browser: aiAgent.js loads the ONNX model  ◄──────────────────────────────┘
           encodes GameView → tensor, runs inference, decodes action
```

### 8.1 Reuse the logic you already have
`js/state.js` is **pure game logic with no DOM** — deliberately. A faithful
**Python port already exists** (it was written to validate the collision rules).
Turn that into a **Gymnasium** environment:
- `reset()` → new board, returns the initial observation (§3).
- `step(action)` → apply the relative action via the same `tick()` logic,
  return `(observation, reward, terminated, info)`.
- Keep it **behaviorally identical** to `state.js`. Ideally share a small
  cross-language test suite (the existing scenarios are a good start) so the two
  implementations can never silently diverge.

### 8.2 Libraries
- **Environment:** [Gymnasium](https://gymnasium.farama.org/) API.
- **Training:** [PyTorch](https://pytorch.org/). For learning-by-doing, the
  single-file implementations in **[CleanRL](https://github.com/vwxyzjn/cleanrl)**
  (`dqn.py`, `ppo.py`) are excellent and easy to adapt. For batteries-included,
  **[Stable-Baselines3](https://stable-baselines3.readthedocs.io/)** gives you
  `DQN` and `PPO` out of the box.
- **Export / inference:** train → export to **ONNX** → run in-browser with
  **[onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/)**. Alternative:
  **TensorFlow.js**. Inference is cheap (one small forward pass per tick).

### 8.3 Why not train in the browser?
You *can* (TensorFlow.js), but training is far slower and the tooling/debugging
is weaker. **Train in Python, infer in the browser.** The `aiAgent.js` stub is
already the right place for the inference-only code.

### 8.4 DQN hyperparameters — reasonable starting points
| Hyperparameter | Starting value |
|----------------|----------------|
| Discount γ | 0.95–0.99 |
| Learning rate | 1e-3 → 1e-4 |
| Replay buffer size | 100k transitions |
| Batch size | 64 |
| Target-network update | every ~1000 steps (or soft τ=0.005) |
| Exploration ε | 1.0 → 0.05, annealed over the first ~10–20% of training |
| Training length | ~0.5–2M environment steps (8×8) |

These are ballparks — expect to tune. PPO has its own set (rollout length, GAE
λ, clip ε≈0.2, entropy bonus); the CleanRL/SB3 defaults are a fine starting
point.

---

## 9. Baselines & evaluation

**Build a non-ML baseline first** — it validates the environment and gives a
number to beat:
- **Greedy BFS:** each step, breadth-first search from the head to the nearest
  apple, move one step along the path; if no safe path, take any safe move.
  Simple and surprisingly strong on multi-apple boards.
- (Aside: a **Hamiltonian cycle** that visits every cell will play Snake
  *perfectly* with no learning at all. Worth knowing that "optimal Snake" is a
  solved non-ML problem — the point of the RL model here is to *learn* good play,
  and ideally to beat the greedy baseline while playing more naturally.)

**Metrics** (average over many fixed-seed games):
- **Apples eaten per game** (primary score).
- **Steps survived** (survival).
- **Max snake length / % of board filled** (mastery).
- Compare against `RandomAgent` (floor) and the greedy BFS agent (target).

Evaluate with exploration **off** (greedy action / argmax), on held-out seeds.

---

## 10. Suggested milestones

1. **Baseline + env.** Greedy-BFS agent; Python Gym env ported from `state.js`
   with parity tests. *(No learning yet — infrastructure.)*
2. **MLP-DQN on 8×8** with the feature vector (§3.1). Goal: beat `RandomAgent`,
   prove the training loop works end-to-end.
3. **CNN-DQN on 8×8** with the grid encoding (§3.2). Goal: approach/beat the
   greedy baseline.
4. **Reward & algorithm tuning.** Add step penalty / optional shaping; add Double
   + Dueling DQN. Goal: stable, efficient play.
5. **Scale to 10×10 and 12×12** (separate models or fully-conv, §6).
6. **Export & integrate.** ONNX → `aiAgent.js` (encode `GameView`, run
   onnxruntime-web, decode action) → flip `AI_ENABLED = true` in `js/config.js`.
   The menu's **AI** button lights up (DESIGN §15.1). *(No other code changes.)*
7. **(Optional) PPO upgrade** for stronger, smoother play.

---

## 11. Key risks & how to manage them

- **Train/inference mismatch.** The Python encoding vs `aiAgent.js` encoding
  drifting apart is the most likely bug. One written spec + tests on both sides
  (§3.2).
- **Self-trapping as the snake grows.** Long-horizon credit assignment is hard;
  this is where the grid+CNN and survival-oriented reward matter most. The BFS
  baseline helps you see how much room there is to improve.
- **Reward hacking / circling.** Tune by watching behavior (§5).
- **Sparse early reward.** The many apples actually help here (frequent +10s);
  if still slow, add potential-based shaping.
- **Sim-to-real drift is not a concern** — the "real" environment *is* the same
  deterministic logic, so a well-trained policy transfers exactly, provided the
  encoding matches.

---

## 12. TL;DR

- Model **DQN with a small CNN** over a **multi-channel grid** of the board;
  **3 relative actions**; reward **+10 eat / −10 die / −0.01 step**.
- Train in **Python** (PyTorch + Gymnasium, CleanRL/SB3) using a **Gym env
  ported from `state.js`**; **export ONNX**; run inference in **`aiAgent.js`**
  via **onnxruntime-web**; flip `AI_ENABLED`.
- Start on **8×8**, beat a **greedy-BFS baseline**, then scale to 10×10/12×12
  and optionally upgrade to **PPO**.
- It's tabular Q-learning from cliff-walking, but with a **network in place of
  the Q-table** — everything else is representation, reward, and plumbing.
