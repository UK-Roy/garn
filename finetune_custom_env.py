"""
Fine-tune a trained GARN checkpoint inside the custom CrowdSimVarNum environment.

Copy this file + models/ + utils/ + environment/custom_adapter.py
into your custom env's repo, then run:

    python finetune_custom_env.py --checkpoint checkpoints/garn_ep20000.pt

Fine-tuning differences from full training:
  - Starts from trained weights (not random)
  - Low initial epsilon (0.1 → 0.01): model already knows how to navigate
  - Lower learning rate (1e-4 vs 5e-4): small adjustments, not full relearning
  - Fewer episodes (2000 default vs 20000): recalibration, not from scratch
  - Fresh replay buffer: old GARN-env transitions are wrong distribution

Usage:
    python finetune_custom_env.py \\
        --checkpoint checkpoints/garn_ep20000.pt \\
        --episodes 2000 \\
        --lr 1e-4 \\
        --epsilon-start 0.1
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.optim as optim
import yaml
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.stgan import STGAN
from training.buffer import ReplayBuffer, collate_batch
from environment.custom_adapter import CustomEnvAdapter


# ── helpers ───────────────────────────────────────────────────────────────────

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def storable(net_obs):
    """Strip int-count fields before buffer storage."""
    return {k: v for k, v in net_obs.items() if isinstance(v, np.ndarray)}


def make_custom_env():
    """
    Instantiate and configure your custom environment.

    Replace the body of this function with your actual env setup.
    The env must follow the gym interface: reset() → obs, {}
                                           step(action) → obs, r, terminated, truncated, info

    Example (adjust imports/paths for your project):
        sys.path.insert(0, '/path/to/crowd_nav_with_group')
        from crowd_sim.envs.crowd_sim_var_num import CrowdSimVarNum
        from crowd_nav.configs.config import Config
        config = Config()
        env = CrowdSimVarNum()
        env.configure(config)
        env.thisSeed   = 0
        env.nenv       = 1
        env.phase      = 'train'
        from crowd_sim.envs.utils.robot import Robot
        robot = Robot(config, 'robot')
        env.set_robot(robot)
        return env
    """
    raise NotImplementedError(
        "Edit make_custom_env() in this file to return your custom env instance."
    )


# ── fine-tuning loop ──────────────────────────────────────────────────────────

def run_episode(env, online_net, adapter, device, epsilon):
    """One episode: collect transitions, return (list_of_transitions, total_reward, info)."""
    obs, _ = env.reset()
    hx = None
    transitions = []
    total_reward = 0.0
    info = {}
    done = False

    while not done:
        net_obs = adapter.obs_to_net(obs)

        # Epsilon-greedy action from the online network
        action_idx, hx = online_net.get_action(net_obs, hx, device, epsilon=epsilon)

        # Map discrete action → (vx, vy) for custom env
        robot_theta = float(np.asarray(obs['robot_node']).flatten()[6])
        vx, vy = adapter.action_to_vel(action_idx, robot_theta)

        next_obs, reward, terminated, truncated, info = env.step([vx, vy])
        done = terminated or truncated

        next_net_obs = adapter.obs_to_net(next_obs)

        transitions.append((
            storable(net_obs),
            action_idx,
            float(reward),
            storable(next_net_obs),
            float(done),
        ))

        obs = next_obs
        total_reward += reward

    return transitions, total_reward, info


def finetune(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Finetune] Device: {device}")

    model_cfg = load_config(args.model_cfg)
    n_actions = len(model_cfg['action']['speeds']) * len(model_cfg['action']['rotations'])

    # Load online and target networks from the trained checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    online_net = STGAN(model_cfg, n_actions).to(device)
    target_net = STGAN(model_cfg, n_actions).to(device)
    online_net.load_state_dict(ckpt['online_net'])
    target_net.load_state_dict(ckpt['online_net'])  # hard-reset target from online
    target_net.eval()
    print(f"[Finetune] Loaded checkpoint: {args.checkpoint}")

    # Fine-tune optimizer: lower lr than original training
    optimizer = optim.Adam(online_net.parameters(), lr=args.lr)
    loss_fn = torch.nn.SmoothL1Loss()

    # Fresh replay buffer (custom env has different obs distribution)
    buffer = ReplayBuffer(capacity=args.buffer_size)

    adapter = CustomEnvAdapter(
        speeds=model_cfg['action']['speeds'],
        rotations=model_cfg['action']['rotations'],
        max_obstacles=12,
        max_pedestrians=20,
        max_groups=3,
    )

    env = make_custom_env()

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    epsilon = args.epsilon_start
    epsilon_end = args.epsilon_end
    # Linear decay: reach epsilon_end by 80% of fine-tune episodes
    epsilon_step = (epsilon - epsilon_end) / max(1, int(0.8 * args.episodes))

    update_count = 0
    recent_rewards = deque(maxlen=50)

    for ep in range(1, args.episodes + 1):
        online_net.train()
        transitions, ep_reward, ep_info = run_episode(
            env, online_net, adapter, device, epsilon)

        # Push all transitions from this episode into the buffer
        for (obs, action, reward, next_obs, done) in transitions:
            buffer.push(obs, action, reward, next_obs, done)

        # Gradient updates: one per step collected (same as original trainer)
        loss_vals = []
        for _ in range(len(transitions)):
            if not buffer.is_ready(args.min_buffer):
                break

            batch = collate_batch(buffer.sample(args.batch_size), device)
            obs_b      = batch['obs']
            next_obs_b = batch['next_obs']
            actions_b  = batch['actions']
            rewards_b  = batch['rewards']
            dones_b    = batch['dones']

            def fwd(net, o):
                q, _, _ = net(
                    o['robot_partial'], o['obs_partial'],
                    o['ped_partial'],   o['grp_partial'],
                    o['robot_full'],    o['obs_full'],
                    o['ped_full'],      o['grp_full'],
                    hx=None)
                return q

            q_current = fwd(online_net, obs_b).gather(
                1, actions_b.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                best_a = fwd(online_net, next_obs_b).argmax(dim=1)
                q_next = fwd(target_net, next_obs_b).gather(
                    1, best_a.unsqueeze(1)).squeeze(1)
                target = rewards_b + args.gamma * q_next * (1.0 - dones_b)

            loss = loss_fn(q_current, target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(online_net.parameters(), 10.0)
            optimizer.step()
            loss_vals.append(loss.item())
            update_count += 1

            # Hard target update
            if update_count % args.target_update == 0:
                target_net.load_state_dict(online_net.state_dict())

        # Epsilon decay
        epsilon = max(epsilon_end, epsilon - epsilon_step)
        recent_rewards.append(ep_reward)

        if ep % 50 == 0:
            mean_r = np.mean(recent_rewards)
            mean_l = np.mean(loss_vals) if loss_vals else float('nan')
            print(f"  ep {ep:5d}/{args.episodes}  "
                  f"reward={mean_r:7.3f}  loss={mean_l:.4f}  eps={epsilon:.3f}  "
                  f"buf={len(buffer)}")

        if ep % args.save_interval == 0:
            path = f'checkpoints/garn_finetuned_ep{ep}.pt'
            torch.save({
                'online_net': online_net.state_dict(),
                'target_net': target_net.state_dict(),
                'episode': ep,
            }, path)
            print(f"  [Saved] {path}")

    # Final checkpoint
    final_path = 'checkpoints/garn_finetuned_final.pt'
    torch.save({'online_net': online_net.state_dict(),
                'target_net': target_net.state_dict(),
                'episode': args.episodes}, final_path)
    print(f"[Finetune] Done. Final checkpoint: {final_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Fine-tune GARN in the custom env')
    parser.add_argument('--checkpoint',    required=True,
                        help='Path to trained GARN checkpoint (e.g. checkpoints/garn_ep20000.pt)')
    parser.add_argument('--model-cfg',     default='config/model_config.yaml')
    parser.add_argument('--episodes',      type=int,   default=2000)
    parser.add_argument('--lr',            type=float, default=1e-4,
                        help='Fine-tune learning rate (default: 1e-4, lower than training 5e-4)')
    parser.add_argument('--gamma',         type=float, default=0.9)
    parser.add_argument('--epsilon-start', type=float, default=0.1,
                        help='Start epsilon (low — model already knows how to navigate)')
    parser.add_argument('--epsilon-end',   type=float, default=0.01)
    parser.add_argument('--batch-size',    type=int,   default=64)
    parser.add_argument('--buffer-size',   type=int,   default=20000)
    parser.add_argument('--min-buffer',    type=int,   default=500,
                        help='Min transitions before first gradient update')
    parser.add_argument('--target-update', type=int,   default=100,
                        help='Target network hard update interval (steps)')
    parser.add_argument('--save-interval', type=int,   default=500)
    args = parser.parse_args()
    finetune(args)


if __name__ == '__main__':
    main()
