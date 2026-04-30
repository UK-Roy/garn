"""
Relation Modeling Module (paper Section III-C-3, Fig. 4c).

Pipeline:
  full_states → f_cs MLPs → E_t
  E_t × A_t  → GCN (f_g, 2 layers) → C_t
  C_t        → LSTMs (f_L*) → H_t
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.attention import build_mlp


class GraphConvLayer(nn.Module):
    """Single GCN layer: C^(l+1) = σ(A C^(l) W_g^(l))."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, A, C):
        """A: (B,N,N), C: (B,N,in_dim) → (B,N,out_dim)."""
        return F.relu(self.W(torch.bmm(A, C)))


class RelationModeling(nn.Module):
    """
    Spatio-temporal relation modeling via GCN + LSTM.

    Full state inputs per agent type:
        robot_full:  (B, 9)
        obs_full:    (B, K, 5)
        ped_full:    (B, I, 9)
        grp_full:    (B, M, 7)

    Returns:
        H_t: (B, N, lstm_hidden)  LSTM hidden states across all agents
    """

    def __init__(self, cfg):
        super().__init__()
        rel = cfg['relation']
        gcn_dims = rel['gcn_dims']     # [256, 126]
        lstm_h = rel['lstm_hidden']    # 126

        # Embedding MLPs per agent type (f_cs in paper = "agent state embedding")
        robot_in = 9
        obs_in = 5
        ped_in = 9
        grp_in = 7

        # All agent types share common embedding dimension = gcn_dims[0]
        embed_dim = gcn_dims[0]  # 256

        self.f_cr = build_mlp(robot_in, rel['f_cr_dims'], embed_dim)  # (128,128)->256
        self.f_cs_obs = build_mlp(obs_in, rel['f_cs_dims'], embed_dim)
        self.f_cs_ped = build_mlp(ped_in, rel['f_cs_dims'], embed_dim)
        self.f_egp = build_mlp(grp_in, rel['f_egp_dims'], embed_dim)

        # Two-layer GCN
        self.gcn1 = GraphConvLayer(embed_dim, gcn_dims[0])
        self.gcn2 = GraphConvLayer(gcn_dims[0], gcn_dims[1])

        # One LSTM per agent type (f_L* in paper — shared hidden states)
        self.lstm_robot = nn.LSTMCell(gcn_dims[1], lstm_h)
        self.lstm_obs = nn.LSTMCell(gcn_dims[1], lstm_h)
        self.lstm_ped = nn.LSTMCell(gcn_dims[1], lstm_h)
        self.lstm_grp = nn.LSTMCell(gcn_dims[1], lstm_h)

        self.lstm_h = lstm_h

    def forward(self, robot_full, obs_full, ped_full, grp_full,
                A_t, hx=None):
        """
        robot_full: (B, 9)
        obs_full:   (B, K, 5)
        ped_full:   (B, I, 9)
        grp_full:   (B, M, 7)
        A_t:        (B, N, N) from attention module
        hx:         optional dict of LSTM hidden states from previous step

        Returns:
            H_t: (B, N, lstm_h)
            new_hx: updated LSTM hidden states
        """
        B = robot_full.size(0)
        K = obs_full.size(1)
        I = ped_full.size(1)
        M = grp_full.size(1)

        # Embed each agent into fixed-size vector
        e_r = self.f_cr(robot_full).unsqueeze(1)                              # (B,1,embed)
        e_o = self.f_cs_obs(obs_full.view(B * K, -1)).view(B, K, -1)         # (B,K,embed)
        e_p = self.f_cs_ped(ped_full.view(B * I, -1)).view(B, I, -1)         # (B,I,embed)
        e_g = self.f_egp(grp_full.view(B * M, -1)).view(B, M, -1)            # (B,M,embed)

        E_t = torch.cat([e_r, e_o, e_p, e_g], dim=1)   # (B, N, embed)

        # GCN: propagate spatial relations through attention-weighted graph
        C1 = self.gcn1(A_t, E_t)                        # (B, N, 256)
        C_t = self.gcn2(A_t, C1)                        # (B, N, 126)

        # Init hidden states if not provided
        if hx is None:
            hx = self._init_hx(B, C_t.device)

        # LSTM update per agent type (each agent type uses a shared LSTMCell)
        # We collect outputs into lists and stack at the end (no in-place ops,
        # autograd-safe).
        n_o, n_p, n_g = K, I, M

        # Reshape stored hidden states into (B, n, lstm_h) for per-agent indexing
        prev_ho = hx['ho'].view(B, n_o, self.lstm_h)
        prev_co = hx['co'].view(B, n_o, self.lstm_h)
        prev_hp = hx['hp'].view(B, n_p, self.lstm_h)
        prev_cp = hx['cp'].view(B, n_p, self.lstm_h)
        prev_hg = hx['hg'].view(B, n_g, self.lstm_h)
        prev_cg = hx['cg'].view(B, n_g, self.lstm_h)

        # Robot (single agent)
        hr, cr = self.lstm_robot(C_t[:, 0, :], (hx['hr'], hx['cr']))
        h_r_list = [hr.unsqueeze(1)]

        # Obstacles
        h_o_list, c_o_list = [], []
        for k in range(n_o):
            h_ok, c_ok = self.lstm_obs(
                C_t[:, 1 + k, :], (prev_ho[:, k, :], prev_co[:, k, :]))
            h_o_list.append(h_ok)
            c_o_list.append(c_ok)

        # Pedestrians
        h_p_list, c_p_list = [], []
        for i in range(n_p):
            h_pi, c_pi = self.lstm_ped(
                C_t[:, 1 + n_o + i, :], (prev_hp[:, i, :], prev_cp[:, i, :]))
            h_p_list.append(h_pi)
            c_p_list.append(c_pi)

        # Groups
        h_g_list, c_g_list = [], []
        for m in range(n_g):
            h_gm, c_gm = self.lstm_grp(
                C_t[:, 1 + n_o + n_p + m, :], (prev_hg[:, m, :], prev_cg[:, m, :]))
            h_g_list.append(h_gm)
            c_g_list.append(c_gm)

        # Stack per-agent hidden states into (B, n, lstm_h) — no in-place ops
        new_ho_t = torch.stack(h_o_list, dim=1) if h_o_list else prev_ho
        new_co_t = torch.stack(c_o_list, dim=1) if c_o_list else prev_co
        new_hp_t = torch.stack(h_p_list, dim=1) if h_p_list else prev_hp
        new_cp_t = torch.stack(c_p_list, dim=1) if c_p_list else prev_cp
        new_hg_t = torch.stack(h_g_list, dim=1) if h_g_list else prev_hg
        new_cg_t = torch.stack(c_g_list, dim=1) if c_g_list else prev_cg

        # Concatenate to form H_t = (B, N, lstm_h)
        H_t = torch.cat(
            [hr.unsqueeze(1), new_ho_t, new_hp_t, new_hg_t], dim=1)

        new_hx = {
            'hr': hr, 'cr': cr,
            'ho': new_ho_t.reshape(B * n_o, self.lstm_h),
            'co': new_co_t.reshape(B * n_o, self.lstm_h),
            'hp': new_hp_t.reshape(B * n_p, self.lstm_h),
            'cp': new_cp_t.reshape(B * n_p, self.lstm_h),
            'hg': new_hg_t.reshape(B * n_g, self.lstm_h),
            'cg': new_cg_t.reshape(B * n_g, self.lstm_h),
        }
        return H_t, new_hx

    def _init_hx(self, B, device):
        """Initialize zero LSTM hidden/cell states."""
        def z(n): return torch.zeros(B * n, self.lstm_h, device=device)
        return {
            'hr': torch.zeros(B, self.lstm_h, device=device),
            'cr': torch.zeros(B, self.lstm_h, device=device),
            'ho': z(10), 'co': z(10),   # padded to max_obstacles
            'hp': z(15), 'cp': z(15),   # padded to max_pedestrians
            'hg': z(6), 'cg': z(6),     # padded to max_groups
        }
