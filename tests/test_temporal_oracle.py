"""
Stage 2A Mathematical and Temporal Oracles Unit Tests.

Covers:
- U-TIME-01: Geometric frequency bank decay and divisibility constraints.
- U-TIME-02: Shifted log transform behavior and zero-gap invariance.
- U-TIME-03: Dimension exactness (Base bank = 64 dim, Directional bank = 64 dim).
- U-TIME-04: Directional temporal asymmetry (Q forward vs K backward).
- U-TIME-05: Exact zero-gap boundary condition (sin(0) = 0, cos(0) = 1).
- U-NUM-01: Dyadic Fourier feature dimension (44) and dyadic powers (2^-7 to 2^14).
"""

import math
import numpy as np
import pytest
import torch

from noah_baseline.models.temporal import (
    compute_temporal_frequencies,
    TemporalFeatureBank,
    BidirectionalTemporalEmbedding,
)
from noah_baseline.data.numeric import DyadicFourierEncoder


def test_time_01_geometric_frequency_decay():
    """U-TIME-01: omega_n = base^(-n / N_F), omega_0 = 1, monotonic decay."""
    N_F = 16
    base = 10000.0
    freqs = compute_temporal_frequencies(N_F, base=base)
    assert freqs.shape == (N_F,)
    assert math.isclose(freqs[0].item(), 1.0, rel_tol=1e-6)
    expected_last = base ** (-(N_F - 1) / N_F)
    assert math.isclose(freqs[-1].item(), expected_last, rel_tol=1e-5)
    # Monotonically strictly decreasing
    diffs = freqs[1:] - freqs[:-1]
    assert torch.all(diffs < 0)


def test_time_02_shifted_log_transform():
    """U-TIME-02: log1p(0) = 0, nonnegative, strictly monotonic."""
    gaps = torch.tensor([0.0, 0.5, 1.0, 24.0, 72.0, 1000.0])
    log_gaps = torch.log1p(gaps)
    assert log_gaps[0].item() == 0.0
    assert torch.all(log_gaps >= 0.0)
    assert torch.all(log_gaps[1:] > log_gaps[:-1])


def test_time_03_dimension_exactness():
    """U-TIME-03: Base bank (F=2) and Directional banks (F=1) both output head_dim = 64."""
    head_dim = 64
    base_bank = TemporalFeatureBank(num_features=2, head_dim=head_dim)
    dir_bank = TemporalFeatureBank(num_features=1, head_dim=head_dim)

    assert base_bank.N_F == 16  # 64 / (2 * 2)
    assert dir_bank.N_F == 32   # 64 / (2 * 1)

    batch_size, seq_len = 4, 128
    u_base = torch.randn(batch_size, seq_len, 2)
    u_dir = torch.randn(batch_size, seq_len, 1)

    out_base = base_bank(u_base)
    out_dir = dir_bank(u_dir)

    assert out_base.shape == (batch_size, seq_len, 64)
    assert out_dir.shape == (batch_size, seq_len, 64)


def test_time_04_directional_asymmetry():
    """U-TIME-04: Q gets next gap; K gets prev gap; asymmetry guaranteed."""
    embedding = BidirectionalTemporalEmbedding(head_dim=64)
    B, L = 2, 10
    pos = torch.arange(L).repeat(B, 1).float()
    age = torch.full((B, L), 50.0)
    delta_next = torch.tensor([[1.0, 2.0, 0.5, 0.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0]] * B)
    delta_prev = torch.tensor([[0.0, 1.0, 2.0, 0.5, 0.0, 10.0, 1.0, 2.0, 3.0, 4.0]] * B)

    time_Q, time_K = embedding(pos, age, delta_next, delta_prev)

    assert time_Q.shape == (B, L, 64)
    assert time_K.shape == (B, L, 64)
    # Wherever delta_next != delta_prev, time_Q != time_K
    asymmetric_positions = delta_next != delta_prev
    diff = torch.norm(time_Q - time_K, dim=-1)
    assert torch.all(diff[asymmetric_positions] > 1e-4)


def test_time_05_zero_gap_boundary():
    """U-TIME-05: delta = 0 yields sin(0) = 0 and cos(0) = 1."""
    dir_bank = TemporalFeatureBank(num_features=1, head_dim=64)
    zero_val = torch.zeros(1, 1, 1)
    out = dir_bank(zero_val).squeeze()  # [64]
    
    # First 32 are sines (should be 0.0)
    assert torch.allclose(out[:32], torch.zeros(32), atol=1e-7)
    # Next 32 are cosines (should be 1.0)
    assert torch.allclose(out[32:], torch.ones(32), atol=1e-7)


def test_num_01_dyadic_fourier_encoder():
    """U-NUM-01: Encodes scalar into 44 dyadic Fourier features."""
    encoder = DyadicFourierEncoder(kappa=1.0)
    assert encoder.output_dim == 44

    # Single scalar
    feat = encoder.encode(120.0)
    assert isinstance(feat, np.ndarray)
    assert feat.shape == (44,)

    # Zero value check: sin(0) = 0, cos(0) = 1
    feat_zero = encoder.encode(0.0)
    assert np.allclose(feat_zero[:22], 0.0, atol=1e-7)
    assert np.allclose(feat_zero[22:], 1.0, atol=1e-7)

    # Batch array check
    batch_vals = np.array([12.5, 98.6, 140.0])
    batch_feats = encoder.encode(batch_vals)
    assert batch_feats.shape == (3, 44)
