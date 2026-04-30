"""
Attention Extraction Module (paper Section III-C-2, Fig. 4b).

Produces attention matrix A_t ∈ R^{N×N} using embedded Gaussian
similarity between all agent feature vectors.
N = 1 (robot) + K (obstacles) + I (pedestrians) + M (groups).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_mlp(input_dim, hidden_dims, output_dim=None, activation=nn.ReLU):
    """Build a simple MLP with ReLU activations."""
    layers = []
    dims = [input_dim] + list(hidden_dims)
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(activation())
    if output_dim is not None:
        layers.append(nn.Linear(dims[-1], output_dim))
    return nn.Sequential(*layers)


class AttentionExtraction(nn.Module):
    """
    Computes N×N attention matrix A_t over all agents.

    Input partial states:
        robot:     (B, 5)   [px, py, theta, vx, vy]
        obstacles: (B, K, 3) [px, py, r]
        peds:      (B, I, 5) [px, py, theta, vx, vy]
        groups:    (B, M, 4) [gx, gy, gvx, gvy]

    Output:
        A_t: (B, N, N)  softmax attention weights
        x_t: (B, N, embed_dim)  per-agent feature vectors
    """

    def __init__(self, cfg):
        super().__init__()
        attn = cfg['attention']
        embed_dim = attn['embed_dim']

        robot_in = 5    # partial state dims from model_config
        obs_in = 5      # unified 5D for both circular and rectangular
        ped_in = 5
        grp_in = 4

        # Four separate MLPs — one per agent type
        self.f_r = build_mlp(robot_in, attn['f_r_dims'], embed_dim)
        self.f_o = build_mlp(obs_in, attn['f_o_dims'], embed_dim)
        self.f_p = build_mlp(ped_in, attn['f_p_dims'], embed_dim)
        self.f_gp = build_mlp(grp_in, attn['f_gp_dims'], embed_dim)

        # Trainable weight matrices W_θ and W_φ (128×128 each)
        self.W_theta = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_phi = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, robot_partial, obs_partial, ped_partial, grp_partial,
                n_obs, n_ped, n_grp):
        """
        robot_partial: (B, 5)
        obs_partial:   (B, K_max, 3)
        ped_partial:   (B, I_max, 5)
        grp_partial:   (B, M_max, 4)
        n_obs, n_ped, n_grp: scalars or (B,) — actual counts (rest are padding)

        Returns: A_t (B, N, N), x_t (B, N, embed_dim)
        """
        B = robot_partial.size(0)

        # Embed each agent type
        x_r = self.f_r(robot_partial).unsqueeze(1)          # (B, 1, embed)

        B_, K, _ = obs_partial.shape
        x_o = self.f_o(obs_partial.view(B_ * K, -1)).view(B_, K, -1)   # (B, K, embed)

        B_, I, _ = ped_partial.shape
        x_p = self.f_p(ped_partial.view(B_ * I, -1)).view(B_, I, -1)   # (B, I, embed)

        B_, M, _ = grp_partial.shape
        x_g = self.f_gp(grp_partial.view(B_ * M, -1)).view(B_, M, -1) # (B, M, embed)

        # Concatenate all agent features: N = 1+K+I+M
        x_t = torch.cat([x_r, x_o, x_p, x_g], dim=1)   # (B, N, embed)

        # Embedded Gaussian similarity: A_{ij} = exp(f_θ(x_i)^T W f_φ(x_j))
        # W = W_θ^T W_φ is implicit through two linear layers
        q = self.W_theta(x_t)   # (B, N, embed)
        k = self.W_phi(x_t)     # (B, N, embed)

        # Scaled dot-product attention
        scale = q.size(-1) ** 0.5
        scores = torch.bmm(q, k.transpose(1, 2)) / scale  # (B, N, N)
        A_t = F.softmax(scores, dim=-1)                   # (B, N, N)

        return A_t, x_t
