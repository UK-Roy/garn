"""
Main training loop for GARN.
Trains on scenario S1, evaluates on S1 and S2 every test_interval episodes.
Matches paper: 20k training episodes, 500 test episodes, Adam lr=0.0005, γ=0.9.
"""

import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from environment.crowd_env import CrowdNavEnv
from training.policy import DoubleDQN
from evaluation.evaluator import Evaluator


class Trainer:
    def __init__(self, env_cfg, model_cfg, train_cfg):
        self.env_cfg = env_cfg
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg

        # Device: GPU if available, else CPU (auto-detect)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Trainer] Using device: {self.device}")

        # Action space size
        n_speeds = len(model_cfg['action']['speeds'])
        n_rots = len(model_cfg['action']['rotations'])
        self.n_actions = n_speeds * n_rots

        # Training environment (scenario S1)
        self.env = CrowdNavEnv(env_cfg, model_cfg, train_cfg, scenario='s1')

        # Policy (Double DQN)
        self.policy = DoubleDQN(model_cfg, train_cfg, self.n_actions, self.device)

        # Evaluator
        self.evaluator = Evaluator(env_cfg, model_cfg, train_cfg, self.device)

        tr = train_cfg['training']
        self.n_episodes = tr['n_episodes']
        self.test_interval = tr['test_interval']
        self.save_interval = tr['save_interval']
        self.log_interval = tr['log_interval']

        ckpt_dir = tr['checkpoint_dir']
        log_dir = tr['log_dir']
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        self.ckpt_dir = ckpt_dir
        self.writer = SummaryWriter(log_dir)

        self.start_episode = 0

    def train(self, resume_path=None):
        if resume_path and os.path.exists(resume_path):
            self.policy.load(resume_path)
            print(f"[Trainer] Resumed from {resume_path}")

        ep_rewards = []
        ep_losses = []
        outcomes = {'goal': 0, 'collision': 0, 'timeout': 0}

        for episode in tqdm(range(self.start_episode, self.n_episodes),
                            desc='Training', unit='ep'):

            ep_reward, ep_loss, ep_info = self._run_episode(train=True)
            ep_rewards.append(ep_reward)
            if ep_loss is not None:
                ep_losses.append(ep_loss)

            # Track episode outcomes
            if ep_info.get('reached_goal'):
                outcomes['goal'] += 1
            elif ep_info.get('collision'):
                outcomes['collision'] += 1
            else:
                outcomes['timeout'] += 1

            self.policy.decay_epsilon(episode)

            # Console + TensorBoard logging
            if (episode + 1) % self.log_interval == 0:
                mean_r    = np.mean(ep_rewards[-self.log_interval:])
                mean_loss = np.mean(ep_losses[-self.log_interval:]) if ep_losses else float('nan')
                total     = sum(outcomes.values())
                tqdm.write(
                    f"  ep {episode+1:6d} | "
                    f"reward {mean_r:+7.3f} | "
                    f"loss {mean_loss:.4f} | "
                    f"eps {self.policy.epsilon:.3f} | "
                    f"buf {len(self.policy.buffer):6d} | "
                    f"goal {outcomes['goal']}/{total} "
                    f"col {outcomes['collision']}/{total} "
                    f"to {outcomes['timeout']}/{total}"
                )
                outcomes = {'goal': 0, 'collision': 0, 'timeout': 0}
                self.writer.add_scalar('train/reward', mean_r, episode)
                self.writer.add_scalar('train/loss', mean_loss, episode)
                self.writer.add_scalar('train/epsilon', self.policy.epsilon, episode)

            # Evaluation
            if (episode + 1) % self.test_interval == 0:
                tqdm.write(f"\n[Eval] Episode {episode + 1}")
                for scen in ['s1', 's2']:
                    metrics = self.evaluator.evaluate(
                        self.policy.online_net, scen,
                        n_episodes=self.train_cfg['training']['test_episodes'])
                    self._log_metrics(metrics, scen, episode)
                    self._print_metrics(metrics, scen)
                tqdm.write("")

            # Checkpoint
            if (episode + 1) % self.save_interval == 0:
                path = os.path.join(self.ckpt_dir, f'garn_ep{episode + 1}.pt')
                self.policy.save(path)
                tqdm.write(f"[Saved] {path}")

        self.writer.close()
        print("[Trainer] Training complete.")

    def _run_episode(self, train=True):
        obs, _ = self.env.reset()
        net_obs = self.env.get_state_for_network()
        hx = self.policy.online_net.reset_hidden(batch_size=1, device=self.device)

        total_reward = 0.0
        losses = []
        info = {}

        while True:
            action, new_hx = self.policy.select_action(net_obs, hx)
            next_obs_gym, reward, terminated, truncated, info = self.env.step(action)
            next_net_obs = self.env.get_state_for_network()

            done = terminated or truncated

            if train:
                store_obs = self._net_obs_to_storable(net_obs)
                store_next = self._net_obs_to_storable(next_net_obs)
                self.policy.store(store_obs, action, reward, store_next, float(done))
                loss = self.policy.update()
                if loss is not None:
                    losses.append(loss)

            hx = new_hx
            net_obs = next_net_obs
            total_reward += reward

            if done:
                break

        mean_loss = np.mean(losses) if losses else None
        return total_reward, mean_loss, info

    def _net_obs_to_storable(self, net_obs):
        """Keep only numpy array fields for buffer storage."""
        return {k: v for k, v in net_obs.items()
                if isinstance(v, np.ndarray)}

    def _log_metrics(self, metrics, scenario, episode):
        prefix = f'eval_{scenario}'
        for k, v in metrics.items():
            self.writer.add_scalar(f'{prefix}/{k}', v, episode)

    def _print_metrics(self, metrics, scenario):
        print(f"  Scenario {scenario.upper()}: " +
              " | ".join(f"{k}={v:.3f}" for k, v in metrics.items()))
