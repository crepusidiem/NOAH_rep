"""
Unit Tests for Anti-Stall Autoregressive Rollout Engine (Step 1).
"""

import pytest
import torch
from src.data.schema import EventCategory, Modality
from src.models.noah import NOAH
from src.models.rollout import sample_top_p, generate_autoregressive_rollout


def test_top_p_sampling_behavior():
    """Verify that top-p sampling works deterministically at T=0 and within support at T>0."""
    logits = torch.tensor([10.0, 1.0, 0.1, -5.0])
    # T=0 must equal argmax (idx 0)
    assert sample_top_p(logits, temperature=0.0) == 0

    # With very low top_p=0.5, only idx 0 should be selected
    samples = [sample_top_p(logits, temperature=1.0, top_p=0.5) for _ in range(20)]
    assert all(s == 0 for s in samples)


def test_rollout_anti_stall_and_diversity():
    """Verify that anti-stall protocol advances time and avoids zero-gap sink."""
    torch.manual_seed(42)
    device = "cpu"
    model = NOAH(
        d_model=256,
        d_latent=128,
        n_layers=2,
        n_heads=2,
        head_dim=64,
        ffn_ratio=2,
    ).to(device)

    # Synthetic 6-token prefix
    prefix = {
        "category_ids": torch.zeros(1, 6, dtype=torch.int64),
        "modality_ids": torch.zeros(1, 6, dtype=torch.int64),
        "specifics_vec": torch.zeros(1, 6, 768),
        "has_specifics": torch.zeros(1, 6, dtype=torch.bool),
        "value_vec": torch.zeros(1, 6, 768),
        "has_value": torch.zeros(1, 6, dtype=torch.bool),
        "positions": torch.arange(6).unsqueeze(0).float(),
        "ages": torch.ones(1, 6).float() * 65.0,
        "delta_next": torch.zeros(1, 6).float(),
        "delta_prev": torch.zeros(1, 6).float(),
        "attention_mask": torch.ones(1, 6, dtype=torch.bool),
    }

    traj = generate_autoregressive_rollout(
        model=model,
        prefix_data=prefix,
        device=device,
        n_rollout_steps=15,
        temperature=0.8,
        top_p=0.9,
        max_consecutive_zero_gaps=3,
        min_forced_gap_hours=1.0,
    )

    assert len(traj) == 15
    # Must have positive cumulative hours (anti-stall active)
    assert traj[-1]["cumulative_hours"] > 0.0
    # Must not have > 3 consecutive zero gaps
    consec_zeros = 0
    max_observed_zeros = 0
    for step in traj:
        if step["predicted_gap_hours"] == 0.0:
            consec_zeros += 1
            max_observed_zeros = max(max_observed_zeros, consec_zeros)
        else:
            consec_zeros = 0
    assert max_observed_zeros <= 3, f"Observed {max_observed_zeros} consecutive zero gaps"
