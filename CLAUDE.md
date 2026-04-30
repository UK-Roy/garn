# GARN — Claude Code Project Guide

## What this project is
Reproduction of the paper **"Group-Aware Robot Navigation in Crowds Using Spatio-Temporal Graph Attention Network With Deep Reinforcement Learning"** (IEEE RA-L 2025, Lu et al.).

Goal: reproduce Table I results (S1 and S2 scenarios) in a modular, GPU/CPU-agnostic codebase, then reuse the model in other environments.

---

## Environment setup

```bash
conda activate garn          # Python 3.10, PyTorch 2.5.1+cpu
# On GPU machine, reinstall PyTorch with CUDA before training:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

All packages: `torch`, `gymnasium`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `pyyaml`, `tqdm`, `pandas`, `tensorboard`.

---

## How to run

```bash
# Train (CPU or GPU — auto-detected)
python train.py

# Resume from checkpoint
python train.py --resume checkpoints/garn_ep5000.pt

# Override episode count (e.g. quick smoke run)
python train.py --episodes 100

# Evaluate a trained checkpoint
python test.py --checkpoint checkpoints/garn_ep20000.pt

# Evaluate specific scenario only
python test.py --checkpoint checkpoints/garn_ep20000.pt --scenario s1
```

All configs live in `config/`. Edit them instead of touching code.

---

## Project layout

```
garn/
├── CLAUDE.md                ← this file
├── PROGRESS.md              ← session-by-session progress tracker
├── GARN.pdf                 ← source paper
├── requirements.txt
├── train.py                 ← training entry point
├── test.py                  ← evaluation entry point
├── config/
│   ├── env_config.yaml      ← world size, robot params, scenarios S1/S2
│   ├── model_config.yaml    ← STGAN hyperparameters, action space
│   └── train_config.yaml    ← lr, gamma, epsilon, buffer, reward weights
├── environment/
│   ├── crowd_env.py         ← Gymnasium env (CrowdNavEnv)
│   ├── agents/
│   │   ├── robot.py         ← Robot state + kinematics
│   │   ├── pedestrian.py    ← Pedestrian state + personal space
│   │   └── obstacle.py      ← CircularObstacle, RectangularObstacle
│   ├── groups/
│   │   ├── group_detector.py  ← F-formation (static) + DBSCAN (dynamic)
│   │   └── group_space.py     ← Convex hull group space representation
│   └── sfm/
│       └── social_force.py  ← Extended SFM for pedestrian simulation
├── models/
│   ├── stgan.py             ← Full STGAN (assembles all modules)
│   └── modules/
│       ├── attention.py     ← Attention extraction (4 MLPs + Gaussian sim)
│       ├── relation.py      ← GCN (2 layers) + per-type LSTMs
│       └── value_estimator.py ← Value MLP → Q-values
├── gam/
│   ├── reward.py            ← R_gl + R_obs + R_prox + R_grp (Eq. 3-8)
│   └── group_awareness.py   ← GAM interface used by env and trainer
├── training/
│   ├── buffer.py            ← Replay buffer + batch collation
│   ├── policy.py            ← Double DQN (online + target nets)
│   └── trainer.py           ← Main training loop, checkpointing, TB logging
├── evaluation/
│   ├── evaluator.py         ← All 9 metrics from paper Table I
│   └── scenarios.py         ← S1 and S2 scenario definitions
└── utils/
    ├── transforms.py        ← Coord transforms, Gaussian space, convex hull
    └── visualization.py     ← Episode trajectory rendering
```

---

## Key paper parameters (hardcoded in configs)

| What | Value | Where |
|------|-------|--------|
| World size | 10 m × 10 m | `env_config.yaml` |
| Time step Δt | 0.25 s | `env_config.yaml` |
| Time limit | 25 s | `env_config.yaml` |
| Robot radius / v_pref | 0.25 m / 1 m/s | `env_config.yaml` |
| Optimizer | Adam, lr=0.0005 | `train_config.yaml` |
| Discount γ | 0.9 | `train_config.yaml` |
| Training episodes | 20,000 | `train_config.yaml` |
| c1 (group OT/FLW) | 0.05 | `train_config.yaml` |
| c2 (coop pass) | 0.025 | `train_config.yaml` |
| GCN dims | [256, 126] | `model_config.yaml` |
| LSTM hidden | 126 | `model_config.yaml` |
| Action space | 15 (3 speeds × 5 rotations) | `model_config.yaml` |

---

## Device handling
The code uses `torch.device('cuda' if torch.cuda.is_available() else 'cpu')` everywhere. No code changes needed when switching between CPU and GPU machines — only reinstall PyTorch with the matching build.

---

## Target results (paper Table I, GARN*)

| Scenario | r_s↑ | r_c↓ | t_s↓ | d_o↑ | n_i↓ | n_cp↑ | n_flw↑ |
|----------|------|------|------|------|------|-------|--------|
| S1 | 1.00 | 0.00 | 12.87 s | 0.22 m | 29/500 | 8/500 | 25/500 |
| S2 | 0.97 | 0.00 | 20.83 s | 0.21 m | 72/500 | 23/500 | 19/500 |

---

## Progress tracking
See `PROGRESS.md` for session-by-session log. Update it after every session.

## Extending to a new environment
The model (`models/stgan.py`) is decoupled from the environment. To use STGAN in a new env:
1. Produce the same `net_obs` dict (keys: `robot_partial`, `obs_partial`, `ped_partial`, `grp_partial`, `robot_full`, `obs_full`, `ped_full`, `grp_full`)
2. Call `model.get_action(net_obs, hx, device)` → action
3. Load a trained checkpoint with `model.load_state_dict(ckpt['online_net'])`

---

## Custom environment transfer workflow

Goal: train **once** here and drop the checkpoint into a user's custom env
with zero dimension surgery.

### Step 1 — Extract spec from the custom env
User pastes a prepared prompt into Claude Code running inside their custom
env's repo. That session inspects the env code (and runs `env.reset()` /
`env.step()` if possible) and produces a single `env_spec.yaml` describing:
world frame, robot model, action space, observation fields, pedestrian/
obstacle/group schema, episode termination, typical/max counts, plus a real
sample observation and short episode trace.

The exact prompt lives in chat history; if it needs to be re-issued, the key
fields the spec must cover are:
`metadata, world, robot, action_space, observation, pedestrians, obstacles,
groups, goal, episode, counts, sample_observation, sample_episode`.

### Step 2 — Match GARN's schema to the spec
User pastes the YAML back here. Then:
1. Save it as `config/custom_env_spec.yaml`.
2. Update `config/model_config.yaml`:
   - `action.speeds` / `action.rotations` → match the custom env's action space
   - `input_dims.*` → match the custom env's observation fields
3. Update `environment/crowd_env.py` padding constants
   (`max_obstacles`, `max_pedestrians`, `max_groups`) → match the custom env's
   `counts.max_*`.
4. (Optional) Tweak `environment/sfm/social_force.py` parameters so the
   training-time pedestrian dynamics resemble the custom env's.

### Step 3 — Train ONCE here
```bash
python train.py     # 20k episodes, schema now matches custom env
```
Produces `checkpoints/garn_ep20000.pt` that is born compatible with the
custom env.

### Step 4 — Drop into custom env
Copy `models/`, `utils/`, the checkpoint, and a thin
`environment/custom_adapter.py` (which translates the user's raw observation
into GARN's 8-key dict). Then:
```python
model = STGAN(model_cfg, n_actions)
model.load_state_dict(torch.load('garn_ep20000.pt')['online_net'])
adapter = CustomAdapter(your_env)
while not done:
    net_obs = adapter(your_env.observe())
    action, hx = model.get_action(net_obs, hx, device)
    your_env.step(action)
```

### Optional Step 5 — Fine-tune in custom env
If the trained policy underperforms (different dynamics, harder group behavior),
run a fine-tune pass inside the custom env starting from the checkpoint:

```bash
# Copy into your custom env's repo:
#   models/  utils/  environment/custom_adapter.py
#   config/model_config.yaml  finetune_custom_env.py

# Edit make_custom_env() in finetune_custom_env.py to return your env instance,
# then:
python finetune_custom_env.py --checkpoint checkpoints/garn_ep20000.pt

# Key differences from full training:
#   --lr 1e-4          (5× lower than training — small adjustments only)
#   --epsilon-start 0.1  (model already knows how to navigate)
#   --episodes 2000    (recalibration, not relearning from scratch)
```

Fine-tune parameters and their defaults:

| Param | Default | Notes |
|-------|---------|-------|
| `--episodes` | 2000 | Increase to 3000 if reward still rising |
| `--lr` | 1e-4 | Lower (5e-5) if performance degrades |
| `--epsilon-start` | 0.1 | Start exploring a bit; model already has policy |
| `--target-update` | 100 | More frequent than training (100 vs 200) |
| `--min-buffer` | 500 | Start learning sooner than training (500 vs 1000) |

Fine-tuned checkpoint saved to `checkpoints/garn_finetuned_final.pt`.

---

## Quick run reference

```bash
conda activate garn
cd /home/lenovo/garn

# Smoke test (~7 min on CPU, verified working 2026-04-30)
python train.py --episodes 100

# Sanity check (~70 min on CPU)
python train.py --episodes 1000

# Full training (~24 hours on CPU, ~2 hours on GPU)
python train.py

# Monitor in a second terminal
tensorboard --logdir /home/lenovo/garn/logs
# → open http://localhost:6006
```
