"""
Unit and Oracle Tests for Variational Module & KL Divergence (LAT-01..05, LOSS-01).

Tests:
1. Analytical Gaussian KL identity: q == p implies KL == 0
2. Known closed-form KL: 1D q=N(0,1), p=N(0,4) -> log(2) - 0.375; 384 dims -> 384 * (log(2) - 0.375)
3. KL direction sensitivity: KL(q || p) != KL(p || q)
4. Smooth prior sigma squash: strictly bounded in [0.05, 2.0] with non-zero gradients
5. Reparameterization gradients reach inputs (h_prev, e_curr)
6. Forward inference uses prior only (no target embedding required)
"""

import math
import pytest
import torch
import torch.nn as nn

from noah_baseline.models.variational import (
    smooth_squash_sigma,
    compute_gaussian_kl,
    ConditionalPrior,
    InformedPosterior,
    VariationalLatentModule,
)


def test_gaussian_kl_identity():
    """U-L06: q == p gives KL == 0 exactly."""
    B, L, D = 2, 8, 384
    mu = torch.randn(B, L, D)
    sigma = torch.rand(B, L, D) + 0.1
    log_sigma = torch.log(sigma)
    
    kl = compute_gaussian_kl(
        mu_q=mu, sigma_q=sigma, log_sigma_q=log_sigma,
        mu_p=mu, sigma_p=sigma, log_sigma_p=log_sigma,
    )
    assert kl.shape == (B, L)
    assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-6)


def test_gaussian_kl_known_closed_form():
    """
    U-L07: 1D q = N(0, 1), p = N(0, 4) gives log(2) - 0.375.
    For D=384 dimensions, sum is exactly 384 * (log(2) - 0.375).
    """
    D = 384
    mu_q = torch.zeros(1, 1, D)
    sigma_q = torch.ones(1, 1, D)
    log_sigma_q = torch.zeros(1, 1, D)
    
    mu_p = torch.zeros(1, 1, D)
    sigma_p = torch.full((1, 1, D), 2.0)  # std = 2.0 -> var = 4.0
    log_sigma_p = torch.full((1, 1, D), math.log(2.0))
    
    kl = compute_gaussian_kl(
        mu_q=mu_q, sigma_q=sigma_q, log_sigma_q=log_sigma_q,
        mu_p=mu_p, sigma_p=sigma_p, log_sigma_p=log_sigma_p,
    )
    expected_1d = math.log(2.0) - 0.375
    expected_total = D * expected_1d
    
    assert math.isclose(kl.item(), expected_total, rel_tol=1e-5)


def test_gaussian_kl_asymmetry():
    """U-L06: KL divergence is asymmetric: KL(q || p) != KL(p || q)."""
    mu_q = torch.tensor([[[0.0]]])
    sigma_q = torch.tensor([[[1.0]]])
    log_sigma_q = torch.log(sigma_q)
    
    mu_p = torch.tensor([[[0.0]]])
    sigma_p = torch.tensor([[[2.0]]])
    log_sigma_p = torch.log(sigma_p)
    
    kl_qp = compute_gaussian_kl(mu_q, sigma_q, log_sigma_q, mu_p, sigma_p, log_sigma_p)
    kl_pq = compute_gaussian_kl(mu_p, sigma_p, log_sigma_p, mu_q, sigma_q, log_sigma_q)
    
    # KL(q||p) = log(2) - 0.375 = 0.318147
    # KL(p||q) = log(1/2) + (4 + 0)/2 - 0.5 = -log(2) + 2 - 0.5 = 1.5 - log(2) = 0.80685
    assert not math.isclose(kl_qp.item(), kl_pq.item(), rel_tol=1e-4)
    assert math.isclose(kl_pq.item(), 1.5 - math.log(2.0), rel_tol=1e-5)


def test_smooth_prior_sigma_squash_bounds_and_gradients():
    """U-L03: Prior sigma is strictly bounded within [0.05, 2.0] with non-zero gradient."""
    raw_s = torch.tensor([-100.0, -10.0, 0.0, 10.0, 100.0], requires_grad=True)
    sigma, log_sigma = smooth_squash_sigma(raw_s, sigma_min=0.05, sigma_max=2.0, in_log_space=True)
    
    # Check bounds with float32 numerical tolerance
    assert torch.all(sigma >= 0.05 - 1e-6)
    assert torch.all(sigma <= 2.0 + 1e-6)
    
    # Check midpoint at s = 0: log_sigma = (log(0.05) + log(2.0)) / 2 = log(sqrt(0.1))
    expected_mid = math.sqrt(0.05 * 2.0)
    assert math.isclose(sigma[2].item(), expected_mid, rel_tol=1e-4)
    
    # Check gradients non-zero at intermediate values
    loss = sigma.sum()
    loss.backward()
    assert raw_s.grad is not None
    assert raw_s.grad[2].item() > 0.0  # Gradient exists at center


def test_variational_reparameterization_gradients():
    """U-L05: Gradients backprop through sample z into h_prev and e_curr."""
    d_model, d_latent = 64, 32
    var_module = VariationalLatentModule(d_model=d_model, d_latent=d_latent)
    
    h_prev = torch.randn(2, 4, d_model, requires_grad=True)
    e_curr = torch.randn(2, 4, d_model, requires_grad=True)
    
    z_q, z_p, kl = var_module.forward_train(h_prev, e_curr)
    
    # Loss combining both latent samples and KL
    loss = z_q.sum() + z_p.sum() + kl.sum()
    loss.backward()
    
    assert h_prev.grad is not None
    assert e_curr.grad is not None
    assert torch.norm(h_prev.grad) > 0.0
    assert torch.norm(e_curr.grad) > 0.0


def test_inference_prior_isolation():
    """U-L11: Inference rollout only calls prior and succeeds without e_curr."""
    d_model, d_latent = 64, 32
    var_module = VariationalLatentModule(d_model=d_model, d_latent=d_latent)
    
    h_prev = torch.randn(2, 4, d_model)
    # Target e_curr is not provided
    z_sample = var_module.forward_inference(h_prev, temperature=1.0)
    z_mean = var_module.forward_inference(h_prev, temperature=0.0)
    
    assert z_sample.shape == (2, 4, d_latent)
    assert z_mean.shape == (2, 4, d_latent)
    # With tau=0, output is deterministic mean
    z_mean_2 = var_module.forward_inference(h_prev, temperature=0.0)
    assert torch.allclose(z_mean, z_mean_2)
