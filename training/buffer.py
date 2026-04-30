"""
Prioritized / uniform replay buffer for Double DQN training.
Stores transitions as dicts of numpy arrays for variable-size observations.
"""

import numpy as np
from collections import deque
import random


class Transition:
    """Single (s, a, r, s', done) transition with variable-size obs."""
    __slots__ = ('obs', 'action', 'reward', 'next_obs', 'done')

    def __init__(self, obs, action, reward, next_obs, done):
        self.obs = obs
        self.action = action
        self.reward = reward
        self.next_obs = next_obs
        self.done = done


class ReplayBuffer:
    """Uniform experience replay buffer."""

    def __init__(self, capacity):
        self.capacity = int(capacity)
        self._buf = deque(maxlen=self.capacity)

    def push(self, obs, action, reward, next_obs, done):
        self._buf.append(Transition(obs, action, reward, next_obs, done))

    def sample(self, batch_size):
        transitions = random.sample(self._buf, min(batch_size, len(self._buf)))
        return transitions

    def __len__(self):
        return len(self._buf)

    def is_ready(self, min_size):
        return len(self._buf) >= min_size


def collate_batch(transitions, device):
    """
    Convert a list of Transition objects into batched tensors.
    Handles variable-size padded observations by stacking numpy arrays.
    Returns dict of tensors.
    """
    import torch

    def stack_obs(obs_list, key):
        return np.stack([o[key] for o in obs_list], axis=0)

    obs_keys = transitions[0].obs.keys()
    obs_batch = {k: stack_obs([t.obs for t in transitions], k) for k in obs_keys}
    next_obs_batch = {k: stack_obs([t.next_obs for t in transitions], k)
                      for k in obs_keys}

    actions = np.array([t.action for t in transitions], dtype=np.int64)
    rewards = np.array([t.reward for t in transitions], dtype=np.float32)
    dones = np.array([t.done for t in transitions], dtype=np.float32)

    def to_tensor(arr):
        return torch.from_numpy(arr).to(device)

    def obs_to_tensors(obs_dict):
        out = {}
        for k, v in obs_dict.items():
            if isinstance(v[0], (int, np.integer)):
                out[k] = torch.from_numpy(v).long().to(device)
            else:
                out[k] = torch.from_numpy(v.astype(np.float32)).to(device)
        return out

    return {
        'obs': obs_to_tensors(obs_batch),
        'next_obs': obs_to_tensors(next_obs_batch),
        'actions': to_tensor(actions),
        'rewards': to_tensor(rewards),
        'dones': to_tensor(dones),
    }
