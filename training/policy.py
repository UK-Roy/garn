"""
Double DQN policy for GARN training (paper uses Double Deep Q-learning).
Reference: Van Hasselt et al., AAAI 2016.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from models.stgan import STGAN
from training.buffer import ReplayBuffer, collate_batch


class DoubleDQN:
    """
    Double DQN with:
    - Online network  (updated every step)
    - Target network  (hard update every target_update steps)
    - Epsilon-greedy exploration with linear decay
    """

    def __init__(self, model_cfg, train_cfg, n_actions, device):
        self.device = device
        self.n_actions = n_actions
        dqn_cfg = train_cfg['dqn']

        self.gamma = dqn_cfg['gamma']
        self.batch_size = dqn_cfg['batch_size']
        self.target_update = dqn_cfg['target_update']
        self.min_buffer = dqn_cfg['min_buffer']

        self.epsilon = dqn_cfg['epsilon_start']
        self.epsilon_end = dqn_cfg['epsilon_end']
        self.epsilon_decay = dqn_cfg['epsilon_decay']

        # Online and target networks
        self.online_net = STGAN(model_cfg, n_actions).to(device)
        self.target_net = STGAN(model_cfg, n_actions).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        opt_cfg = train_cfg['optimizer']
        self.optimizer = optim.Adam(
            self.online_net.parameters(),
            lr=opt_cfg['lr'],
            weight_decay=opt_cfg.get('weight_decay', 0.0))

        self.buffer = ReplayBuffer(dqn_cfg['buffer_size'])
        self.update_count = 0
        self.loss_fn = nn.SmoothL1Loss()

    def select_action(self, obs_tensors, hx):
        """Epsilon-greedy action selection."""
        action, new_hx = self.online_net.get_action(
            obs_tensors, hx, self.device, epsilon=self.epsilon)
        return action, new_hx

    def store(self, obs, action, reward, next_obs, done):
        """Add transition to replay buffer."""
        self.buffer.push(obs, action, reward, next_obs, done)

    def update(self):
        """One gradient step on the online network. Returns loss or None."""
        if not self.buffer.is_ready(self.min_buffer):
            return None

        transitions = self.buffer.sample(self.batch_size)
        batch = collate_batch(transitions, self.device)

        loss = self._compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.target_update == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.item()

    def _compute_loss(self, batch):
        obs = batch['obs']
        next_obs = batch['next_obs']
        actions = batch['actions']
        rewards = batch['rewards']
        dones = batch['dones']

        def forward_net(net, o):
            q, _, _ = net(
                o['robot_partial'], o['obs_partial'],
                o['ped_partial'], o['grp_partial'],
                o['robot_full'], o['obs_full'],
                o['ped_full'], o['grp_full'],
                hx=None)
            return q

        # Current Q-values
        q_online = forward_net(self.online_net, obs)
        q_current = q_online.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN target: action selected by online, evaluated by target
        with torch.no_grad():
            q_next_online = forward_net(self.online_net, next_obs)
            best_actions = q_next_online.argmax(dim=1)
            q_next_target = forward_net(self.target_net, next_obs)
            q_next = q_next_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)
            target = rewards + self.gamma * q_next * (1.0 - dones)

        return self.loss_fn(q_current, target)

    def decay_epsilon(self, episode):
        """Linear epsilon decay."""
        frac = min(1.0, episode / self.epsilon_decay)
        self.epsilon = self.epsilon_end + (1.0 - self.epsilon_end) * (1.0 - frac)

    def save(self, path):
        torch.save({
            'online_net': self.online_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'update_count': self.update_count,
            'epsilon': self.epsilon,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt['online_net'])
        self.target_net.load_state_dict(ckpt['target_net'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.update_count = ckpt.get('update_count', 0)
        self.epsilon = ckpt.get('epsilon', self.epsilon_end)
