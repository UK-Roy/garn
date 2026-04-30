"""
Drop GARN into the custom CrowdSimVarNum environment.

Copy this file + models/ + utils/ + environment/custom_adapter.py
into your custom env's repo, then run:

    python deploy_in_custom_env.py --checkpoint checkpoints/garn_ep20000.pt

Requirements:
    pip install torch pyyaml numpy
"""

import argparse
import yaml
import torch
import numpy as np

# --- These imports assume this file lives in the garn/ root ---
from models.stgan import STGAN
from environment.custom_adapter import CustomEnvAdapter


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def run_episode(env, model, adapter, device):
    obs, _ = env.reset()
    hx = None
    done = False
    total_reward = 0.0
    steps = 0
    info = {}

    while not done:
        net_obs = adapter.obs_to_net(obs)
        action_idx, hx = model.get_action(net_obs, hx, device, epsilon=0.0)
        robot_theta = float(np.asarray(obs['robot_node']).flatten()[6])
        vx, vy = adapter.action_to_vel(action_idx, robot_theta)
        obs, reward, terminated, truncated, info = env.step([vx, vy])
        done = terminated or truncated
        total_reward += reward
        steps += 1

    return total_reward, steps, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--model-cfg', default='config/model_config.yaml')
    parser.add_argument('--episodes', type=int, default=10)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_cfg = load_config(args.model_cfg)

    # Adapter — matches the custom env's max agent counts
    adapter = CustomEnvAdapter(
        speeds=model_cfg['action']['speeds'],
        rotations=model_cfg['action']['rotations'],
        max_obstacles=12,
        max_pedestrians=20,
        max_groups=3,
    )

    n_actions = len(model_cfg['action']['speeds']) * len(model_cfg['action']['rotations'])
    model = STGAN(model_cfg, n_actions).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['online_net'])
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}  (episode {ckpt.get('episode', '?')})")

    # --- Plug in your custom env here ---
    # Example (adjust imports to your project):
    #
    #   import sys; sys.path.insert(0, '/path/to/crowd_nav_with_group')
    #   from crowd_sim.envs.crowd_sim_var_num import CrowdSimVarNum
    #   from crowd_nav.configs.config import Config
    #   config = Config()
    #   env = CrowdSimVarNum()
    #   env.configure(config)
    #   # set robot, phase, seed, etc. as needed
    #
    # For demonstration we use a placeholder that raises NotImplementedError:
    raise NotImplementedError(
        "Replace this line with your custom env instantiation.\n"
        "See the comment block above for the import pattern."
    )

    rewards, lengths = [], []
    for ep in range(args.episodes):
        r, s, info = run_episode(env, model, adapter, device)
        rewards.append(r)
        lengths.append(s)
        print(f"  ep {ep+1:3d}  reward={r:7.3f}  steps={s:3d}  {info}")

    print(f"\nMean reward: {np.mean(rewards):.3f}  Mean steps: {np.mean(lengths):.1f}")


if __name__ == '__main__':
    main()
