# GARN Reproduction — Progress Tracker

**Paper:** Group-Aware Robot Navigation in Crowds Using Spatio-Temporal Graph Attention Network With Deep Reinforcement Learning (IEEE RA-L 2025)  
**Authors:** Xiaojun Lu, Angela Faragasso, Yongdong Wang, Atsushi Yamashita, Hajime Asama  
**Goal:** Fully reproduce GARN's simulation results (Table I) in a modular, GPU/CPU-agnostic codebase.

---

## Session Log

### Session 1 — 2026-04-30
**Status:** In progress

**Completed:**
- [x] Read and understood the full paper
- [x] Created conda environment `garn` (Python 3.10)
- [x] Installed all packages: PyTorch 2.11.0+cpu, Gymnasium 1.3.0, scikit-learn, scipy, matplotlib, tensorboard, tqdm, pyyaml, pandas
- [x] Created project directory structure
- [x] Written `requirements.txt`
- [x] Written `config/env_config.yaml`
- [x] Written `config/model_config.yaml`
- [x] Written `config/train_config.yaml`
- [x] Written `utils/transforms.py`
- [x] Written `environment/agents/robot.py`
- [x] Written `environment/agents/pedestrian.py`
- [x] Written `environment/agents/obstacle.py`
- [x] Written `environment/sfm/social_force.py`
- [x] Written `environment/groups/group_detector.py`
- [x] Written `environment/groups/group_space.py`
- [x] Written `environment/crowd_env.py`
- [x] Written `models/modules/attention.py`
- [x] Written `models/modules/relation.py`
- [x] Written `models/modules/value_estimator.py`
- [x] Written `models/stgan.py`
- [x] Written `gam/reward.py`
- [x] Written `gam/group_awareness.py`
- [x] Written `training/buffer.py`
- [x] Written `training/policy.py`
- [x] Written `training/trainer.py`
- [x] Written `evaluation/scenarios.py`
- [x] Written `evaluation/evaluator.py`
- [x] Written `utils/visualization.py`
- [x] Written `train.py`
- [x] Written `test.py`
- [x] Written all `__init__.py` files

**Smoke tests passed (2026-04-30):**
- [x] Smoke-test environment (single episode rollout) ✓
- [x] Smoke-test model forward pass: q shape=(1,15), A shape=(1,32,32) ✓
- [x] 20-step rollout with epsilon-greedy action selection ✓
- [x] Bug fix: obstacle partial state unified to 5D (circle padded, rect native)

**Pending (next session):**
- [ ] Full training run (20k episodes) — push to GPU machine and run
- [ ] Evaluation on S1 and S2 scenarios (500 episodes each)
- [ ] Compare results to Table I of the paper
- [ ] Visualize agent trajectories

### Session 2 — 2026-04-30

**Completed:**
- [x] Written `environment/custom_adapter.py` — translates CrowdSimVarNum obs → GARN net_obs
  - Handles sentinel 15.0 (invisible agents masked to zero)
  - Builds robot_partial(5D), robot_full(9D), ped_partial(I,5), ped_full(I,9), grp_partial(M,4), grp_full(M,7)
  - action_to_vel(): maps GARN discrete action index → (vx, vy) for custom env
- [x] Written `deploy_in_custom_env.py` — deployment script for custom env inference

**Key decision: Training env unchanged**
  Model weights depend only on per-agent feature sizes (5D partial, 9D full) — same in both envs.
  The adapter handles all format translation at inference time. No architecture surgery needed.

**Pending:**
- [ ] Full training run (20k episodes) — push to GPU machine and run
- [ ] Wire in actual custom env import in deploy_in_custom_env.py and finetune_custom_env.py
- [ ] Optional: fine-tune inside custom env if zero-shot transfer underperforms

**Smoke test result (2026-04-30, CPU):**
- 100 episodes in 7m 12s (~4.3 s/ep)
- Projected: 1000 ep ≈ 70 min, 20k ep ≈ 24 hours on CPU; push to GPU for full run
- No crashes, training loop confirmed working end-to-end

---

## Key Paper Parameters

### Environment
| Parameter | Value |
|-----------|-------|
| Space | 10 m × 10 m square |
| Robot radius | 0.25 m |
| Preferred speed | 1 m/s |
| Time step Δt | 0.25 s |
| Time limit | 25 s |
| Start position | (0, 0.5) |
| Goal position | (2, 9.5) |

### Scenario S1
- 6 individuals, 8 obstacles
- 1 static group (2 members), 1 dynamic group (3 members)

### Scenario S2
- 6 individuals, 8 obstacles
- 2 static groups (2+3 members), 2 dynamic groups (3+4 members)

### Model Hyperparameters
| Component | Hidden sizes |
|-----------|-------------|
| f_r (robot attention MLP) | (256, 128) |
| f_o (obstacle attention MLP) | (128, 128) |
| f_p (pedestrian attention MLP) | (256, 128) |
| f_gp (group attention MLP) | (128, 128) |
| f_gs (group state MLP) | (512, 256) |
| f_cr (robot relation MLP) | (128, 128) |
| f_cs (agent relation MLP) | (512, 256) |
| f_egp (group relation MLP) | (256, 128) |
| f_v (value MLP) | (256, 128, 128) |
| W_θ, W_φ dimensions | 128 |
| GCN L1 | 256 |
| GCN L2 | 126 |

### Training Hyperparameters
| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning rate | 0.0005 |
| Discount factor γ | 0.9 |
| Training episodes | 20,000 |
| Test episodes | 500 |
| c1 (group intrusion weight) | 0.05 |
| c2 (cooperative passing weight) | 0.025 |

### Reward Components
- R_gl: goal reward
- R_obs: obstacle reward
- R_prox: proximity reward
- R_grp = R_intru + R_ot/flw + R_cops
  - R_intru: -0.25 per step inside group convex hull
  - R_ot/flw: overtaking/following reward (c1=0.05)
  - R_cops: cooperative passing reward (c2=0.025)

### Target Metrics (Table I, GARN*)
| Scenario | r_s↑ | r_c↓ | t_s(s)↓ | d_o(m)↑ | n_dis↓ | n_i↓ | n_cp↑ | n_flw↑ | n_ot↑ |
|----------|------|------|---------|---------|--------|------|-------|--------|-------|
| S1 | 1.00 | 0.00 | 12.87 | 0.22 | 0/500 | 29/500 | 8/500 | 25/500 | — |
| S2 | 0.97 | 0.00 | 20.83 | 0.21 | 7/500 | 72/500 | 23/500 | 19/500 | — |

---

## Architecture Summary

```
Joint state J_t = [S_t, O^k_t, P^i_t, GP^m_t]
                          ↓
              [STGAN]
              ┌─────────────────────────────────────┐
              │  Attention Extraction               │
              │  f_r, f_o, f_p, f_gp → x_t        │
              │  similarity(x_t) → A_t (softmax)   │
              │                                     │
              │  Relation Modeling                  │
              │  f_cs(states) → E_t                │
              │  E_t × A_t → GCN(f_g) → C_t       │
              │  LSTM(f_L*)(C_t) → H_t             │
              │                                     │
              │  Value Estimation                   │
              │  [H_t[0,:] ∥ S_t] → f_v → v, a_t │
              └─────────────────────────────────────┘
                          ↓
              action a_t = v_t (velocity command)
```

---

## File Structure

```
garn/
├── PROGRESS.md              ← this file (updated each session)
├── GARN.pdf                 ← paper
├── requirements.txt
├── train.py                 ← main training entry point
├── test.py                  ← main testing entry point
├── config/
│   ├── env_config.yaml
│   ├── model_config.yaml
│   └── train_config.yaml
├── environment/
│   ├── crowd_env.py         ← gym-compatible crowd navigation env
│   ├── agents/
│   │   ├── robot.py
│   │   ├── pedestrian.py
│   │   └── obstacle.py
│   ├── groups/
│   │   ├── group_detector.py
│   │   └── group_space.py
│   └── sfm/
│       └── social_force.py
├── models/
│   ├── stgan.py             ← full STGAN network
│   └── modules/
│       ├── attention.py
│       ├── relation.py
│       └── value_estimator.py
├── gam/
│   ├── group_awareness.py
│   └── reward.py
├── training/
│   ├── trainer.py
│   ├── buffer.py
│   └── policy.py
├── evaluation/
│   ├── evaluator.py
│   └── scenarios.py
└── utils/
    ├── transforms.py
    └── visualization.py
```
