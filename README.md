# GARN — Group-Aware Robot Navigation

Reproduction of **"Group-Aware Robot Navigation in Crowds Using Spatio-Temporal Graph Attention Network With Deep Reinforcement Learning"** (IEEE RA-L 2025, Lu et al.).

The model (STGAN) combines attention extraction, graph convolution, and per-type LSTMs, trained with Double DQN, to navigate a robot through crowds while respecting pedestrian group boundaries.

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/UK-Roy/garn.git
cd garn
```

### 2. Create and activate the conda environment
```bash
conda create -n garn python=3.10 -y
conda activate garn
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> **GPU machine:** Replace the PyTorch install with the CUDA build before training:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```
> No code changes needed — the code auto-detects GPU/CPU.

---

## Training

```bash
conda activate garn

# Quick smoke test (~7 min on CPU)
python train.py --episodes 100

# Full training — 20k episodes (~24 hr CPU / ~2 hr GPU)
python train.py

# Resume from a checkpoint
python train.py --resume checkpoints/garn_ep10000.pt

# Monitor training in a second terminal
tensorboard --logdir logs
# → open http://localhost:6006
```

Checkpoints are saved every 1000 episodes to `checkpoints/`.

---

## Evaluation

```bash
# Evaluate on both scenarios (S1 and S2)
python test.py --checkpoint checkpoints/garn_ep20000.pt

# Evaluate on one scenario only
python test.py --checkpoint checkpoints/garn_ep20000.pt --scenario s1
```

### Target results (paper Table I, GARN*)

| Scenario | Success↑ | Collision↓ | Time↓ | Min dist↑ | Intrusions↓ |
|----------|----------|------------|-------|-----------|-------------|
| S1 | 1.00 | 0.00 | 12.87 s | 0.22 m | 29/500 |
| S2 | 0.97 | 0.00 | 20.83 s | 0.21 m | 72/500 |

---

## Transfer to a Custom Environment

The model is fully decoupled from the training environment. To deploy a trained checkpoint in a new environment:

### Step 1 — Copy the required files
```
models/
utils/
environment/custom_adapter.py
config/model_config.yaml
deploy_in_custom_env.py
finetune_custom_env.py   ← optional, for fine-tuning
```

### Step 2 — Translate observations with the adapter
The adapter converts your environment's observation dict into the 8-key format expected by STGAN:
```python
from environment.custom_adapter import CustomEnvAdapter

adapter = CustomEnvAdapter(speeds=[0.0, 0.5, 1.0],
                           rotations=[-0.5236, -0.2618, 0.0, 0.2618, 0.5236])

obs, _ = env.reset()
net_obs = adapter.obs_to_net(obs)
action_idx, hx = model.get_action(net_obs, hx, device, epsilon=0.0)
vx, vy = adapter.action_to_vel(action_idx, robot_theta)
env.step([vx, vy])
```

### Step 3 — Deploy
Edit `make_custom_env()` in `deploy_in_custom_env.py` to return your env instance, then:
```bash
python deploy_in_custom_env.py --checkpoint checkpoints/garn_ep20000.pt
```

### Step 4 — Fine-tune (optional)
If zero-shot performance is weak, fine-tune for 1–2k episodes inside your custom env:
```bash
python finetune_custom_env.py --checkpoint checkpoints/garn_ep20000.pt \
    --episodes 2000 --lr 1e-4 --epsilon-start 0.1
```

---

## Project Structure

```
garn/
├── train.py                    ← training entry point
├── test.py                     ← evaluation entry point
├── deploy_in_custom_env.py     ← zero-shot deployment in custom env
├── finetune_custom_env.py      ← fine-tuning in custom env
├── config/
│   ├── env_config.yaml         ← world, robot, scenario S1/S2 params
│   ├── model_config.yaml       ← STGAN architecture hyperparameters
│   └── train_config.yaml       ← optimizer, DQN, reward weights
├── environment/
│   ├── crowd_env.py            ← Gymnasium training environment
│   ├── custom_adapter.py       ← obs translator for custom envs
│   ├── agents/                 ← Robot, Pedestrian, Obstacle classes
│   ├── groups/                 ← group detection + convex hull space
│   └── sfm/                    ← Extended Social Force Model
├── models/
│   ├── stgan.py                ← full STGAN network
│   └── modules/
│       ├── attention.py        ← attention extraction (4 MLPs + Gaussian sim)
│       ├── relation.py         ← GCN (2 layers) + per-type LSTMs
│       └── value_estimator.py  ← value MLP → Q-values
├── gam/
│   ├── reward.py               ← R_gl + R_obs + R_prox + R_grp
│   └── group_awareness.py      ← group awareness module
├── training/
│   ├── trainer.py              ← main training loop + TensorBoard logging
│   ├── policy.py               ← Double DQN (online + target networks)
│   └── buffer.py               ← experience replay buffer
├── evaluation/
│   ├── evaluator.py            ← 9 metrics from paper Table I
│   └── scenarios.py            ← S1 and S2 scenario definitions
└── utils/
    ├── transforms.py           ← coordinate transforms, action mapping
    └── visualization.py        ← trajectory rendering
```

---

## Citation

```bibtex
@article{lu2025garn,
  title={Group-Aware Robot Navigation in Crowds Using Spatio-Temporal Graph Attention Network With Deep Reinforcement Learning},
  author={Lu, Xiaojun and Faragasso, Angela and Wang, Yongdong and Yamashita, Atsushi and Asama, Hajime},
  journal={IEEE Robotics and Automation Letters},
  year={2025}
}
```
