"""
NOAH: Neural Overarching Architecture for Healthcare (Complete Model).
Implements the end-to-end architecture conforming to the paper specification:
- Input projection of multimodal events (ARCH-01, NUM-01, NUM-02)
- Bidirectional temporal causal transformer backbone (ARCH-01..03, TIME-01..05)
- Variational latent module with conditional prior & informed posterior (LAT-01..05)
- Hierarchical multimodal decoder (DEC-01..06)
- Dual-path reconstruction & KL objective (LOSS-01..04)
- Prior-only autoregressive rollout (ROLL-01..03)
"""

import math
from typing import Dict, NamedTuple, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.schema import EventCategory, Modality
from src.models.projections import ModalityInputProjector
from src.models.transformer import NOAHTransformerBackbone
from src.models.variational import (
    ConditionalPrior,
    InformedPosterior,
    compute_gaussian_kl,
    sample_diagonal_gaussian,
)
from src.models.decoder import HierarchicalMultimodalDecoder, DecoderOutputs


class NOAHLossBreakdown(NamedTuple):
    total_loss: torch.Tensor
    loss_rec_post: torch.Tensor
    loss_rec_prior: torch.Tensor
    loss_kl: torch.Tensor
    loss_category: torch.Tensor
    loss_modality: torch.Tensor
    loss_specifics_presence: torch.Tensor
    loss_specifics_vec: torch.Tensor
    loss_value: torch.Tensor
    loss_time_gate: torch.Tensor
    loss_time_reg: torch.Tensor


class NOAH(nn.Module):
    """
    Complete NOAH Model.
    """
    def __init__(
        self,
        d_model: int = 256,
        d_latent: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        head_dim: int = 64,
        ffn_ratio: int = 4,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        num_categories: int = 31,
        num_modalities: int = 6,
        text_dim: int = 768,
        cxr_dim: int = 768,
        ecg_dim: int = 192,
        echo_dim: int = 512,
        numeric_dim: int = 44,
        temporal_layers: int = 4,
        layer_scale_init: float = 1e-4,
        alpha_post: float = 0.3,
        gamma_prior: float = 0.1,
        category_class_weights: Optional[torch.Tensor] = None,
        modality_class_weights: Optional[torch.Tensor] = None,
        specifics_presence_class_weights: Optional[torch.Tensor] = None,
        category_modality_support: Optional[torch.Tensor] = None,
        drop_path: float = 0.0,

    ):
        super().__init__()
        self.d_model = d_model
        self.d_latent = d_latent
        self.alpha_post = alpha_post
        self.gamma_prior = gamma_prior
        self.register_buffer("category_class_weights", category_class_weights, persistent=False)

        self.register_buffer("modality_class_weights", modality_class_weights, persistent=False)
        self.register_buffer("specifics_presence_class_weights", specifics_presence_class_weights, persistent=False)
        self.register_buffer("category_modality_support", category_modality_support, persistent=False)

        # 1. Modality Input Projector
        self.projector = ModalityInputProjector(
            d_model=d_model,
            num_categories=num_categories,
            num_modalities=num_modalities,
            numeric_dim=numeric_dim,
            text_dim=text_dim,
            cxr_dim=cxr_dim,
            ecg_dim=ecg_dim,
            echo_dim=echo_dim,
            dropout=dropout,
        )

        # 2. Causal Transformer Backbone
        self.backbone = NOAHTransformerBackbone(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            head_dim=head_dim,
            ffn_ratio=ffn_ratio,
            dropout=dropout,
            attn_dropout=attn_dropout,
            temporal_layers=temporal_layers,
            layer_scale_init=layer_scale_init,
            drop_path=drop_path,
        )

        # 3. Variational Latent Module (LAT-01..05)
        self.prior = ConditionalPrior(
            d_model=d_model,
            d_latent=d_latent,
            hidden_dim=d_model,
            sigma_min=0.05,
            sigma_max=2.0,
        )
        self.posterior = InformedPosterior(
            d_model=d_model,
            d_latent=d_latent,
            hidden_dim=d_model,
        )

        # 4. Hierarchical Multimodal Decoder (DEC-01..06)
        self.decoder = HierarchicalMultimodalDecoder(
            d_latent=d_latent,
            hidden_dim=d_latent,
            num_categories=num_categories,
            num_modalities=num_modalities,
            text_dim=text_dim,
            cxr_dim=cxr_dim,
            ecg_dim=ecg_dim,
            echo_dim=echo_dim,
            numeric_dim=numeric_dim,
        )

    def forward(
        self,
        category_ids: torch.Tensor,       # [B, L]
        modality_ids: torch.Tensor,       # [B, L]
        positions: torch.Tensor,          # [B, L]
        ages: torch.Tensor,               # [B, L]
        delta_prev: torch.Tensor,         # [B, L]
        delta_next: torch.Tensor,         # [B, L]
        has_next_gap: torch.Tensor,       # [B, L] bool
        specifics_vec: torch.Tensor,      # [B, L, 768]
        has_specifics: torch.Tensor,      # [B, L] bool
        value_vec: torch.Tensor,          # [B, L, max_val_dim]
        has_value: torch.Tensor,          # [B, L] bool
        attention_mask: torch.Tensor,     # [B, L] bool
        loss_mask: torch.Tensor,          # [B, L] bool
        beta: float = 1.0,                # KL annealing weight
    ) -> Tuple[torch.Tensor, NOAHLossBreakdown, Dict[str, torch.Tensor]]:
        """
        Executes teacher-forced forward pass and computes the dual-path loss.
        """
        B, L = category_ids.shape

        # 1. Project input tokens e_t: [B, L, d_model]
        e = self.projector(
            category_ids=category_ids,
            modality_ids=modality_ids,
            specifics_vec=specifics_vec,
            has_specifics=has_specifics,
            value_vec=value_vec,
            has_value=has_value,
        )

        # 2. Causal Transformer backbone produces patient history representations h
        # h shape: [B, L, d_model]
        h = self.backbone(
            e=e,
            positions=positions,
            ages=ages,
            delta_next=delta_next,
            delta_prev=delta_prev,
            attention_mask=attention_mask,
        )

        # 3. Historical shift alignment (ALIGN-01, ALIGN-02)
        # Target event x_t is at index t in [1 .. L-1].
        # Its historical context is h_{t-1} at index t-1 in [0 .. L-2].
        # For sequence length L, targets are evaluated on indices [1 .. L-1]
        h_prev = h[:, :-1, :]              # [B, L-1, d_model]
        e_target = e[:, 1:, :]             # [B, L-1, d_model]

        # Target attributes shifted to [1 .. L-1]
        cat_tgt = category_ids[:, 1:]
        mod_tgt = modality_ids[:, 1:]
        spec_vec_tgt = specifics_vec[:, 1:]
        has_spec_tgt = has_specifics[:, 1:]
        val_vec_tgt = value_vec[:, 1:]
        has_val_tgt = has_value[:, 1:]
        delta_tgt = delta_next[:, 1:]
        has_gap_tgt = has_next_gap[:, 1:]
        mask_tgt = loss_mask[:, 1:] & attention_mask[:, 1:] & attention_mask[:, :-1]        # [B, L-1] bool loss ownership mask

        # 4. Variational Distributions
        # Context Prior: p(z_t | h_{t-1})
        mu_p, sigma_p, log_sigma_p = self.prior(h_prev)
        # Informed Posterior: q(z_t | h_{t-1}, e_t)
        mu_q, sigma_q, log_sigma_q = self.posterior(h_prev, e_target)

        # Analytical KL Divergence: [B, L-1]
        kl_div = compute_gaussian_kl(
            mu_q=mu_q,
            sigma_q=sigma_q,
            log_sigma_q=log_sigma_q,
            mu_p=mu_p,
            sigma_p=sigma_p,
            log_sigma_p=log_sigma_p,
        )

        # Reparameterized sampling (LAT-03)
        z_post = sample_diagonal_gaussian(mu_q, sigma_q)   # [B, L-1, d_latent]
        z_prior = sample_diagonal_gaussian(mu_p, sigma_p)  # [B, L-1, d_latent]

        # 5. Dual Reconstruction Passes (LOSS-01)
        dec_post = self.decoder(z_post)
        dec_prior = self.decoder(z_prior)

        # 6. Compute Component Reconstruction Losses
        loss_post, b_post = self._compute_reconstruction_loss(
            dec_out=dec_post,
            category_targets=cat_tgt,
            modality_targets=mod_tgt,
            specifics_targets=spec_vec_tgt,
            has_specifics=has_spec_tgt,
            value_targets=val_vec_tgt,
            has_value=has_val_tgt,
            delta_targets=delta_tgt,
            has_next_gap=has_gap_tgt,
            loss_mask=mask_tgt,
        )

        loss_prior, _ = self._compute_reconstruction_loss(
            dec_out=dec_prior,
            category_targets=cat_tgt,
            modality_targets=mod_tgt,
            specifics_targets=spec_vec_tgt,
            has_specifics=has_spec_tgt,
            value_targets=val_vec_tgt,
            has_value=has_val_tgt,
            delta_targets=delta_tgt,
            has_next_gap=has_gap_tgt,
            loss_mask=mask_tgt,
        )

        # Masked KL loss
        valid_count = mask_tgt.sum().clamp(min=1.0)
        kl_loss = (kl_div * mask_tgt.float()).sum() / valid_count

        # Total Dual-Path Objective (LOSS-01)
        total_loss = self.alpha_post * loss_post + beta * kl_loss + self.gamma_prior * loss_prior

        loss_breakdown = NOAHLossBreakdown(
            total_loss=total_loss,
            loss_rec_post=loss_post,
            loss_rec_prior=loss_prior,
            loss_kl=kl_loss,
            loss_category=b_post["category"],
            loss_modality=b_post["modality"],
            loss_specifics_presence=b_post["specifics_presence"],
            loss_specifics_vec=b_post["specifics_vec"],
            loss_value=b_post["value"],
            loss_time_gate=b_post["time_gate"],
            loss_time_reg=b_post["time_reg"],
        )

        aux = {
            "h": h,
            "mu_p": mu_p,
            "sigma_p": sigma_p,
            "mu_q": mu_q,
            "sigma_q": sigma_q,
            "kl_per_token": kl_div,
        }

        return total_loss, loss_breakdown, aux

    def _compute_reconstruction_loss(
        self,
        dec_out: DecoderOutputs,
        category_targets: torch.Tensor,
        modality_targets: torch.Tensor,
        specifics_targets: torch.Tensor,
        has_specifics: torch.Tensor,
        value_targets: torch.Tensor,
        has_value: torch.Tensor,
        delta_targets: torch.Tensor,
        has_next_gap: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Computes composite reconstruction loss across all multimodal heads obeying masks.
        """
        valid_mask = loss_mask.bool()
        denom = valid_mask.sum().clamp(min=1.0).float()

        # 1. Category NLL
        c_logits = dec_out.category_logits[valid_mask]  # [N, num_cats]
        c_tgt = category_targets[valid_mask]            # [N]
        cat_weight = self.category_class_weights.to(dtype=c_logits.dtype, device=c_logits.device) if self.category_class_weights is not None else None
        loss_cat = F.cross_entropy(c_logits, c_tgt, weight=cat_weight) if valid_mask.any() else torch.tensor(0.0, device=loss_mask.device)

        # 2. Modality NLL
        m_logits = dec_out.modality_logits[valid_mask]  # [N, num_mods]
        m_tgt = modality_targets[valid_mask]            # [N]
        loss_mod = F.cross_entropy(m_logits, m_tgt, weight=self.modality_class_weights) if valid_mask.any() else torch.tensor(0.0, device=loss_mask.device)

        # 3. Specifics Presence BCE
        pres_logits = dec_out.specifics_presence_logits.squeeze(-1)[valid_mask]
        pres_tgt = has_specifics[valid_mask].float()
        pres_weights = None
        if self.specifics_presence_class_weights is not None:
            pres_weights = self.specifics_presence_class_weights[pres_tgt.long()]
        loss_pres = F.binary_cross_entropy_with_logits(pres_logits, pres_tgt, weight=pres_weights) if valid_mask.any() else torch.tensor(0.0, device=loss_mask.device)

        # 4. Specifics Vector Regression (only where has_specifics is True)
        spec_mask = valid_mask & has_specifics.bool()
        if spec_mask.any():
            pred_s = dec_out.specifics_vec[spec_mask]
            tgt_s = specifics_targets[spec_mask]
            loss_spec_vec = F.mse_loss(pred_s, tgt_s)
        else:
            loss_spec_vec = torch.tensor(0.0, device=loss_mask.device)

        # 5. Modality Value Regression
        val_mask = valid_mask & has_value.bool()
        loss_val = torch.tensor(0.0, device=loss_mask.device)
        if val_mask.any():
            # Numeric values
            num_mask = val_mask & (modality_targets == Modality.NUMERIC)
            if num_mask.any():
                loss_val = loss_val + F.mse_loss(
                    dec_out.numeric_vec[num_mask],
                    value_targets[num_mask][:, :44]
                )
            # CXR values
            cxr_mask = val_mask & (modality_targets == Modality.CXR)
            if cxr_mask.any():
                loss_val = loss_val + F.mse_loss(
                    dec_out.cxr_vec[cxr_mask],
                    value_targets[cxr_mask][:, :768]
                )
            # Text values
            text_mask = val_mask & (modality_targets == Modality.TEXT)
            if text_mask.any():
                loss_val = loss_val + F.mse_loss(
                    dec_out.text_vec[text_mask],
                    value_targets[text_mask][:, :768]
                )
            # ECG values
            ecg_mask = val_mask & (modality_targets == Modality.ECG)
            if ecg_mask.any():
                loss_val = loss_val + F.mse_loss(
                    dec_out.ecg_vec[ecg_mask],
                    value_targets[ecg_mask][:, :192]
                )

            echo_mask = val_mask & (modality_targets == Modality.ECHO)
            if echo_mask.any():
                loss_val = loss_val + F.mse_loss(
                    dec_out.echo_vec[echo_mask], value_targets[echo_mask][:, :512]
                )

        # 6. Time (Outgoing delta_next hurdle)
        # Supervised only where has_next_gap is True (excludes terminal events)
        time_mask = valid_mask & has_next_gap.bool()
        if time_mask.any():
            t_gate_logits = dec_out.time_gate_logits.squeeze(-1)[time_mask]
            is_pos_gap = (delta_targets[time_mask] > 0.0).float()
            loss_time_gate = F.binary_cross_entropy_with_logits(t_gate_logits, is_pos_gap)

            # Continuous regression only on positive gaps
            pos_mask = time_mask & (delta_targets > 0.0)
            if pos_mask.any():
                pred_log = dec_out.log_gap_reg.squeeze(-1)[pos_mask]
                tgt_log = torch.log1p(delta_targets[pos_mask])
                loss_time_reg = F.mse_loss(pred_log, tgt_log)
            else:
                loss_time_reg = torch.tensor(0.0, device=loss_mask.device)
        else:
            loss_time_gate = torch.tensor(0.0, device=loss_mask.device)
            loss_time_reg = torch.tensor(0.0, device=loss_mask.device)

        total_rec = (
            loss_cat +
            loss_mod +
            loss_pres +
            loss_spec_vec +
            loss_val +
            loss_time_gate +
            loss_time_reg
        )

        breakdown = {
            "category": loss_cat.detach(),
            "modality": loss_mod.detach(),
            "specifics_presence": loss_pres.detach(),
            "specifics_vec": loss_spec_vec.detach(),
            "value": loss_val.detach(),
            "time_gate": loss_time_gate.detach(),
            "time_reg": loss_time_reg.detach(),
        }

        return total_rec, breakdown
