"""
STGAN: Spatio-Temporal Graph Attention Network (paper Section III-C, Fig. 3 & 4).

Full pipeline:
  [J_t] → AttentionExtraction → A_t
        → RelationModeling(A_t) → H_t
        → ValueEstimator(H_t[0,:], S_t) → Q-values → action
"""

import torch
import torch.nn as nn

from models.modules.attention import AttentionExtraction
from models.modules.relation import RelationModeling
from models.modules.value_estimator import ValueEstimator


class STGAN(nn.Module):
    """
    Full Spatio-Temporal Graph Attention Network.

    Args:
        model_cfg: dict from model_config.yaml
        n_actions:  number of discrete actions

    Forward input tensors (all batched, B = batch size):
        robot_partial:  (B, 5)
        obs_partial:    (B, K, 3)
        ped_partial:    (B, I, 5)
        grp_partial:    (B, M, 4)
        robot_full:     (B, 9)
        obs_full:       (B, K, 5)
        ped_full:       (B, I, 9)
        grp_full:       (B, M, 7)
        hx:             dict of LSTM hidden states (None → zeros)

    Returns:
        q_values: (B, n_actions)
        new_hx:   updated LSTM hidden states
        A_t:      (B, N, N) attention matrix (for inspection / pre-training)
    """

    def __init__(self, model_cfg, n_actions):
        super().__init__()
        self.attention = AttentionExtraction(model_cfg)
        self.relation = RelationModeling(model_cfg)
        self.value = ValueEstimator(model_cfg, n_actions)
        self.n_actions = n_actions

    def forward(self, robot_partial, obs_partial, ped_partial, grp_partial,
                robot_full, obs_full, ped_full, grp_full, hx=None):

        n_obs = obs_partial.size(1)
        n_ped = ped_partial.size(1)
        n_grp = grp_partial.size(1)

        # 1. Attention extraction
        A_t, x_t = self.attention(
            robot_partial, obs_partial, ped_partial, grp_partial,
            n_obs, n_ped, n_grp)

        # 2. Relation modeling
        H_t, new_hx = self.relation(
            robot_full, obs_full, ped_full, grp_full, A_t, hx)

        # 3. Value estimation — use robot's row (index 0) of H_t
        h_robot = H_t[:, 0, :]           # (B, lstm_h)
        q_values = self.value(h_robot, robot_full)

        return q_values, new_hx, A_t

    def reset_hidden(self, batch_size=1, device='cpu'):
        """Return fresh zero LSTM hidden states for a new episode."""
        return self.relation._init_hx(batch_size, device)

    def get_action(self, obs_tensors, hx, device, epsilon=0.0):
        """
        Epsilon-greedy action selection for a single observation.
        obs_tensors: dict from env.get_state_for_network(), numpy arrays
        Returns: action (int), new_hx
        """
        import numpy as np

        if torch.rand(1).item() < epsilon:
            return torch.randint(0, self.n_actions, (1,)).item(), hx

        self.eval()
        with torch.no_grad():
            t = {k: torch.FloatTensor(v).unsqueeze(0).to(device)
                 for k, v in obs_tensors.items()
                 if isinstance(v, (list, __import__('numpy').ndarray))}

            q_values, new_hx, _ = self.forward(
                t['robot_partial'], t['obs_partial'],
                t['ped_partial'], t['grp_partial'],
                t['robot_full'], t['obs_full'],
                t['ped_full'], t['grp_full'],
                hx)

        action = q_values.argmax(dim=-1).item()
        return action, new_hx
