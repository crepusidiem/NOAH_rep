"""
Bidirectional Temporal Attention Embedding (TIME-01, TIME-02, TIME-03).

Implements the geometric temporal frequency bank and directional gap embeddings:
- Base bank: position p_i and age a_i (F=2, N_2 = d_h / 4 = 16) -> 64 dim
- Directional forward bank: delta_next_i (F=1, N_1 = d_h / 2 = 32) -> 64 dim
- Directional backward bank: delta_prev_i (F=1, N_1 = d_h / 2 = 32) -> 64 dim
- Updates:
    Q_i^* = Q_i + Psi_2([p_i, a_i]) + Psi_1([log1p(delta_next_i)])
    K_i^* = K_i + Psi_2([p_i, a_i]) + Psi_1([log1p(delta_prev_i)])
"""

import math
import numpy as np
import torch
import torch.nn as nn


def compute_temporal_frequencies(N_F: int, base: float = 10000.0) -> torch.Tensor:
    """
    omega_n = base^(-n / N_F) for n = 0, ..., N_F - 1
    """
    indices = torch.arange(N_F, dtype=torch.float32)
    return torch.pow(base, -indices / N_F)


class TemporalFeatureBank(nn.Module):
    """
    Encodes F features into d_h dimensions using sinusoidal basis functions.
    Requires 2 * F divides d_h.
    """
    def __init__(self, num_features: int, head_dim: int = 64, base: float = 10000.0):
        super().__init__()
        self.num_features = num_features
        self.head_dim = head_dim
        assert head_dim % (2 * num_features) == 0, (
            f"2 * F ({2 * num_features}) must divide head_dim ({head_dim})"
        )
        self.N_F = head_dim // (2 * num_features)
        frequencies = compute_temporal_frequencies(self.N_F, base=base)
        self.register_buffer("frequencies", frequencies)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: Tensor of shape [..., F] containing feature values.
        Returns:
            Tensor of shape [..., head_dim]
        """
        # u shape: [..., F, 1]
        u_expanded = u.unsqueeze(-1)
        # frequencies shape: [1, ..., 1, N_F]
        freq = self.frequencies.view(*([1] * (u.dim())), self.N_F)
        
        args = u_expanded * freq  # [..., F, N_F]
        sin_part = torch.sin(args)
        cos_part = torch.cos(args)
        
        # psi_F(u_i) = [sin_0..N_F-1, cos_0..N_F-1] -> shape [..., F, 2 * N_F]
        psi = torch.cat([sin_part, cos_part], dim=-1)
        
        # Psi_F(u) = concatenate across all F features -> shape [..., F * 2 * N_F] = [..., head_dim]
        Psi = psi.view(*u.shape[:-1], self.head_dim)
        return Psi


class BidirectionalTemporalEmbedding(nn.Module):
    """
    Computes Q and K temporal additions:
    - Base bank: [ordinal_position, age]
    - Query directional bank: [log1p(delta_next)]
    - Key directional bank: [log1p(delta_prev)]
    """
    def __init__(self, head_dim: int = 64, base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        # Base bank for 2 features: position and age
        self.base_bank = TemporalFeatureBank(num_features=2, head_dim=head_dim, base=base)
        # Directional banks for 1 feature: forward / backward log gaps
        self.fwd_bank = TemporalFeatureBank(num_features=1, head_dim=head_dim, base=base)
        self.bwd_bank = TemporalFeatureBank(num_features=1, head_dim=head_dim, base=base)

    def forward(
        self,
        positions: torch.Tensor,     # [B, L]
        ages: torch.Tensor,          # [B, L]
        delta_next: torch.Tensor,    # [B, L]
        delta_prev: torch.Tensor,    # [B, L]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            time_Q: [B, L, head_dim] to add to queries
            time_K: [B, L, head_dim] to add to keys
        """
        # Shifted log transforms: log(1 + delta)
        log_fwd = torch.log1p(torch.clamp(delta_next, min=0.0)).unsqueeze(-1)  # [B, L, 1]
        log_bwd = torch.log1p(torch.clamp(delta_prev, min=0.0)).unsqueeze(-1)  # [B, L, 1]
        
        base_features = torch.stack([positions, ages], dim=-1)  # [B, L, 2]
        
        psi_base = self.base_bank(base_features)  # [B, L, head_dim]
        psi_fwd = self.fwd_bank(log_fwd)          # [B, L, head_dim]
        psi_bwd = self.bwd_bank(log_bwd)          # [B, L, head_dim]
        
        time_Q = psi_base + psi_fwd  # [B, L, head_dim]
        time_K = psi_base + psi_bwd  # [B, L, head_dim]
        return time_Q, time_K
