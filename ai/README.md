# Snake AI (`ai/`)

Reinforcement-learning training code for the Snake AI controller. Implements the
**simple milestone** from [`../AI_PLAN.md`](../AI_PLAN.md): a feature-vector
MLP trained with DQN, hand-written in numpy. Trains in Python, then runs in the
browser via `js/agent/aiAgent.js` (no torch/onnxruntime needed at inference —
the network is tiny).

## Layout

| File | Purpose |
|------|---------|
| `env.py` | `SnakeEnv` — Gym-style env, a faithful port of `js/state.js` (3 relative actions). |
| `features.py` | Observation encoders: 11-feature vector (`featurize`) + grid tensor (`grid_encode`, for the future CNN). |
| `baselines.py` | `RandomAgent`, `GreedyBFSAgent` — non-learning benchmarks. |
| `dqn.py` | numpy MLP + Adam + replay buffer + target network; `DQNAgent`. |
| `train.py` | Training loop + final head-to-head vs baselines. |
| `evaluate.py` | Metrics harness (apples eaten, steps survived). |
| `tests/` | `unittest` parity tests vs the JS engine. |

## Requirements

Only **numpy**. (PyTorch is the documented upgrade path for the CNN/PPO
milestones — see `AI_PLAN.md` §2, §8 — but is not used here.)

```
pip install numpy
```

## GPU (optional, for CNN / larger runs)

CPU is fine for the feature-MLP and small CNN runs. A CUDA build of PyTorch is
installed in an **isolated venv** so it never clashes with the CPU `torch` in
`~/.local`:

```
/home/elsa/.venvs/snake-gpu/bin/python -m ai.train_cnn --device cuda ...
```

`--device auto` picks CUDA when available. Note: for a *tiny* 8×8 CNN with
single-env stepping, the GPU is often no faster than CPU (per-step launch/sync
overhead dominates) — the real GPU win comes with vectorized environments and
bigger nets. Verified working on an RTX 5080 (sm_120, CUDA 13). For small CPU
runs, fewer threads is often faster: `--threads 8` (the default).

## Train

```
python -m ai.train --size 8 --steps 100000 --out ai/models/dqn_8
```

Writes `ai/models/dqn_8.npz` (numpy weights) and `ai/models/dqn_8.json`
(weights for the browser). To use a trained model in the game, copy the JSON to
the served path the frontend loads:

```
cp ai/models/dqn_8.json models/snake-dqn.json
```

## Evaluate / test

```
python -m unittest discover -s ai/tests      # env parity vs js/state.js
```

Each trainer prints a final comparison against the `GreedyBFS` and `Random`
baselines.

## Results

**Feature-MLP** (`train.py`, 8×8, ~35s CPU): mean ≈ 22 apples (≈ greedy-BFS 25,
random 9). The 11 features are board-size agnostic, so one MLP plays all sizes.

**Grid + CNN** (`train_vec.py`, GPU, vectorized) — **beats the greedy baseline on
every board**. Architecture keeps 64 conv channels with a 2×2 max-pool before the
flatten (`ai/cnn.py`), which recovers full performance at ~1/4 the model size:

| Board | CNN-DQN | GreedyBFS | Random | Model | Time (GPU) |
|-------|---------|-----------|--------|-------|------------|
| 8×8   | **31.4**| 24.9      | 8.5    | 1.3 MB | ~1.5 min  |
| 10×10 | **36.5**| 29.8      | 8.5    | 1.9 MB | ~2 min    |
| 12×12 | **38.7**| 37.8      | 7.3    | 2.6 MB | ~3 min    |

Vectorized GPU training runs at ~5,200 transitions/s — ~37× the original
single-env CPU loop.

## Deployment to the browser

The CNN is the in-game **AI** controller. A model is trained per board size
(the dense head is size-specific) and exported as a compact binary:

```
python -m ai.train_vec --size 10 --envs 64 --steps 30000 --device cuda --out ai/models/cnn_10
cp ai/models/cnn_10.json models/snake-cnn-10.json   # manifest
cp ai/models/cnn_10.bin  models/snake-cnn-10.bin    # weights
```

`export_binary` walks the module list (conv / pool / dense) and writes a small
`.json` op manifest + a little-endian float32 `.bin`. `js/agent/aiAgent.js` loads
`models/snake-cnn-<size>.{json,bin}` and runs a plain-JS conv/pool/dense forward —
verified numerically against torch (max abs diff ~2e-5, argmax identical).

## Next steps (see `AI_PLAN.md`)

- **Size-agnostic model**: a global-average-pool head plus coordinate channels
  would let one model serve all board sizes (and shrink further).
- Double DQN is already in (`cnn.py`); add Dueling / prioritized replay; then
  optionally **PPO** (actor-critic).
- For much larger scale, a decoupled actor–learner architecture would fully
  saturate the GPU (currently single-process batched-synchronous).
