"""
NOAH Variational Latent Module & Objective (VAR-01, VAR-02, LOSS-01, LOSS-02).

Implements:
1. Conditional Prior p(z_t | h_{t-1}):
   - Maps history h_{t-1} [B, L, d_model] to (mu_p, sigma_p) [B, L, d_latent]
   - Smooth squash of sigma_p to [0.05, 2.0]
2. Informed Posterior q(z_t | h_{t-1}, e_t):
   - Maps [h_{t-1}, e_t] [B, L, 2 * d_model] to (mu_q, sigma_q) [B, L, d_latent]
3. Reparameterized sampling: z = mu + sigma * eps
4. Analytical Gaussian KL divergence: KL(q || p)
5. Dual reconstruction loss weighting:
   L = 0.3 * L_rec(q) + beta * KL(q || p) + 0.1 * L_rec(p)
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def smooth_squash_sigma(
    raw_s: torch.Tensor,
    sigma_min: float = 0.05,
    sigma_max: float = 2.0,
    in_log_space: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Smoothly squashes raw scale logits to [sigma_min, sigma_max].
    
    If in_log_space is True (A-20 candidate):
        log_sigma = log(sigma_min) + sigmoid(raw_s) * (log(sigma_max) - log(sigma_min))
        sigma = exp(log_sigma)
    Else:
        sigma = sigma_min + sigmoid(raw_s) * (sigma_max - sigma_min)
        log_sigma = log(sigma)
        
    Returns:
        (sigma, log_sigma)
    """
    sig = torch.sigmoid(raw_s)
    if in_log_space:
        log_min = math.log(sigma_min)
        log_max = math.log(sigma_max)
        log_sigma = log_min + sig * (log_max - log_min)
        sigma = torch.exp(log_sigma)
    else:
        sigma = sigma_min + sig * (sigma_max - sigma_min)
        log_sigma = torch.log(sigma)
    return sigma, log_sigma


def compute_gaussian_kl(
    mu_q: torch.Tensor,
    sigma_q: torch.Tensor,
    log_sigma_q: torch.Tensor,
    mu_p: torch.Tensor,
    sigma_p: torch.Tensor,
    log_sigma_p: torch.Tensor,
) -> torch.Tensor:
    """
    Analytical KL divergence between diagonal Gaussians:
        KL(q || p) = sum_j [ log(sigma_p_j / sigma_q_j) + (sigma_q_j^2 + (mu_q_j - mu_p_j)^2) / (2 * sigma_p_j^2) - 0.5 ]
    
    Returns:
        Tensor of shape [..., ] containing sum over latent dimension.
    """
    # [..., d_latent]
    var_p = sigma_p.pow(2)
    var_q = sigma_q.pow(2)
    diff_sq = (mu_q - mu_p).pow(2)
    
    kl_per_dim = (log_sigma_p - log_sigma_q) + (var_q + diff_sq) / (2.0 * var_p) - 0.5
    return kl_per_dim.sum(dim=-1)


def sample_diagonal_gaussian(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    eps: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Reparameterized sample: z = mu + (temperature * sigma) * eps"""
    if eps is None:
        eps = torch.randn_like(mu)
    return mu + (temperature * sigma) * eps


class ConditionalPrior(nn.Module):
    """
    Learned conditional prior p(z_t | h_{t-1}).
    Takes historical hidden representation h_{t-1} and outputs (mu_p, sigma_p, log_sigma_p).
    """
    def __init__(
        self,
        d_model: int = 768,
        d_latent: int = 384,
        hidden_dim: Optional[int] = None,
        sigma_min: float = 0.05,
        sigma_max: float = 2.0,
    ):
        super().__init__()
        hidden_dim = hidden_dim or d_model
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * d_latent),
        )

    def forward(self, h_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            h_prev: [B, L, d_model]
        Returns:
            mu_p: [B, L, d_latent]
            sigma_p: [B, L, d_latent] bounded in [sigma_min, sigma_max]
            log_sigma_p: [B, L, d_latent]
        """
        out = self.net(h_prev)
        mu_p, raw_s = torch.chunk(out, 2, dim=-1)
        sigma_p, log_sigma_p = smooth_squash_sigma(
            raw_s, sigma_min=self.sigma_min, sigma_max=self.sigma_max, in_log_space=True
        )
        return mu_p, sigma_p, log_sigma_p


class InformedPosterior(nn.Module):
    """
    Informed posterior q(z_t | h_{t-1}, e_t).
    Takes [h_{t-1}, e_t] (concatenation: 2 * d_model) and outputs (mu_q, sigma_q, log_sigma_q).
    """
    def __init__(
        self,
        d_model: int = 768,
        d_latent: int = 384,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        hidden_dim = hidden_dim or d_model
        self.net = nn.Sequential(
            nn.Linear(2 * d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * d_latent),
        )

    def forward(self, h_prev: torch.Tensor, e_curr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            h_prev: [B, L, d_model] historical state
            e_curr: [B, L, d_model] current event target embedding
        Returns:
            mu_q: [B, L, d_latent]
            sigma_q: [B, L, d_latent]
            log_sigma_q: [B, L, d_latent]
        """
        inp = torch.cat([h_prev, e_curr], dim=-1)
        out = self.net(inp)
        mu_q, raw_log_sigma = torch.chunk(out, 2, dim=-1)
        
        # Numerically stabilized log_sigma for posterior (A-21)
        log_sigma_q = torch.clamp(raw_log_sigma, min=-10.0, max=10.0)
        sigma_q = torch.exp(log_sigma_q)
        return mu_q, sigma_q, log_sigma_q


class VariationalLatentModule(nn.Module):
    """
    Complete NOAH variational module connecting conditional prior and posterior.
    Handles sampling, KL calculation, and dual path evaluation.
    """
    def __init__(
        self,
        d_model: int = 768,
        d_latent: int = 384,
        sigma_min: float = 0.05,
        sigma_max: float = 2.0,
    ):
        super().__init__()
        self.prior = ConditionalPrior(d_model=d_model, d_latent=d_latent, sigma_min=sigma_min, sigma_max=sigma_max)
        self.posterior = InformedPosterior(d_model=d_model, d_latent=d_latent)

    def sample(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        eps: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Reparameterized sample: z = mu + (temperature * sigma) * eps"""
        if eps is None:
            eps = torch.randn_like(mu)
        return mu + (temperature * sigma) * eps

    def forward_train(
        self,
        h_prev: torch.Tensor,
        e_curr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Training forward pass:
        Returns:
            z_q: Sampled latent from posterior q(z_t | h_{t-1}, e_t)
            z_p: Sampled latent from prior p(z_t | h_{t-1})
            kl: KL(q || p) of shape [B, L]
        """
        mu_p, sigma_p, log_sigma_p = self.prior(h_prev)
        mu_q, sigma_q, log_sigma_q = self.posterior(h_prev, e_curr)
        
        # Shared or independent epsilon for reparameterization (A-22)
        eps_q = torch.randn_like(mu_q)
        eps_p = torch.randn_like(mu_p)
        
        z_q = self.sample(mu_q, sigma_q, eps=eps_q)
        z_p = self.sample(mu_p, sigma_p, eps=eps_p)
        
        kl = compute_gaussian_kl(
            mu_q=mu_q, sigma_q=sigma_q, log_sigma_q=log_sigma_q,
            mu_p=mu_p, sigma_p=sigma_p, log_sigma_p=log_sigma_p,
        )
        return z_q, z_p, kl

    def forward_inference(
        self,
        h_prev: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Inference rollout forward pass:
        Uses history and prior only. True target embedding is strictly unavailable.
        """
        mu_p, sigma_p, _ = self.prior(h_prev)
        if temperature == 0.0:
            return mu_p
        return self.sample(mu_p, sigma_p, temperature=temperature)
