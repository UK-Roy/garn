"""
Value Estimation Module (paper Section III-C-4, Fig. 3).

Input:  concatenation of H_t[0,:] (robot's LSTM hidden state) + S_t (full robot state)
Output: Q-values for all discrete actions (Double DQN style)
"""

import torch
import torch.nn as nn
from models.modules.attention import build_mlp


class ValueEstimator(nn.Module):
    """
    f_v: MLP that maps [H_t[0,:] ∥ S_t] → Q-values for each action.
    hidden dims: (256, 128, 128) per paper.
    """

    def __init__(self, cfg, n_actions):
        super().__init__()
        val = cfg['value']
        lstm_h = cfg['relation']['lstm_hidden']   # 126
        robot_full_dim = 9                        # full robot state dimension

        input_dim = lstm_h + robot_full_dim       # 126 + 9 = 135

        self.f_v = build_mlp(input_dim, val['f_v_dims'], n_actions)

    def forward(self, h_robot, robot_full):
        """
        h_robot:    (B, lstm_h)  — H_t[0,:], robot's aggregated relation feature
        robot_full: (B, 9)       — S_t, robot's full state
        Returns:
            q_values: (B, n_actions)
        """
        x = torch.cat([h_robot, robot_full], dim=-1)
        return self.f_v(x)
