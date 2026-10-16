"""
Hierarchical Multimodal Decoder for NOAH (DEC-01 through DEC-06).

Factorizes the event probability distribution into a cascading feature MLP:
1. Category logits:
     f_t^{cat} = F_cat(z_t)
     c_logits = H_cat(f_t^{cat})
2. Specifics presence & regression:
     f_t^{spec} = F_spec(z_t + f_t^{cat})
     s_presence_logit = H_pres(f_t^{spec})
     s_vec = H_spec(f_t^{spec}) (768-D)
3. Modality logits:
     m_logits = H_m(z_t + f_t^{spec})
4. Value heads (selected by modality):
     v_numeric = H_v^{num}(z_t + f_t^{spec}) (44-D Fourier coefficients)
     v_cxr     = H_v^{cxr}(z_t + f_t^{spec}) (768-D)
     v_text    = H_v^{text}(z_t + f_t^{spec}) (768-D)
     v_ecg     = H_v^{ecg}(z_t + f_t^{spec}) (192-D)
     v_echo    = H_v^{echo}(z_t + f_t^{spec}) (512-D)
5. Outgoing Time gap (delta_next):
     time_gate_logit = H_delta_gate(z_t)
     log_gap_reg     = H_delta_reg(z_t)
"""

import math
from typing import Dict, NamedTuple, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.schema import EventCategory, Modality


class DecoderOutputs(NamedTuple):
    category_logits: torch.Tensor          # [B, L, num_categories]
    specifics_presence_logits: torch.Tensor # [B, L, 1]
    specifics_vec: torch.Tensor            # [B, L, 768]
    modality_logits: torch.Tensor          # [B, L, num_modalities]
    numeric_vec: torch.Tensor              # [B, L, 44]
    cxr_vec: torch.Tensor                  # [B, L, 768]
    text_vec: torch.Tensor                 # [B, L, 768]
    ecg_vec: torch.Tensor                  # [B, L, 192]
    echo_vec: torch.Tensor                 # [B, L, 512]
    time_gate_logits: torch.Tensor         # [B, L, 1]
    log_gap_reg: torch.Tensor              # [B, L, 1]


class HierarchicalMultimodalDecoder(nn.Module):
    """
    Hierarchical multimodal decoder executing the cascading feature cascade (DEC-01..06).
    """
    def __init__(
        self,
        d_latent: int = 128,
        hidden_dim: int = 256,
        num_categories: int = 31,
        num_modalities: int = 6,
        text_dim: int = 768,
        cxr_dim: int = 768,
        ecg_dim: int = 192,
        echo_dim: int = 512,
        numeric_dim: int = 44,
    ):
        super().__init__()
        self.d_latent = d_latent
        self.hidden_dim = hidden_dim
        self.num_categories = num_categories
        self.num_modalities = num_modalities

        # 1. Category Cascade: F_cat & H_cat
        self.F_cat = nn.Sequential(
            nn.Linear(d_latent, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.H_cat = nn.Linear(hidden_dim, num_categories)

        # 2. Specifics Cascade: F_spec, H_pres & H_spec
        # Consumes (z_t + f_t^{cat})
        # Note: if d_latent != hidden_dim, we project z_t to hidden_dim for addition
        self.z_to_hidden = (
            nn.Linear(d_latent, hidden_dim) if d_latent != hidden_dim else nn.Identity()
        )
        self.F_spec = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.H_pres = nn.Linear(hidden_dim, 1)
        self.H_spec = nn.Linear(hidden_dim, text_dim)

        # 3. Modality Cascade: H_m
        # Consumes (z_t + f_t^{spec})
        self.H_m = nn.Linear(hidden_dim, num_modalities)

        # 4. Modality Value Heads
        # Each consumes (z_t + f_t^{spec})
        self.H_v_numeric = nn.Linear(hidden_dim, numeric_dim)
        self.H_v_cxr = nn.Linear(hidden_dim, cxr_dim)
        self.H_v_text = nn.Linear(hidden_dim, text_dim)
        self.H_v_ecg = nn.Linear(hidden_dim, ecg_dim)
        self.H_v_echo = nn.Linear(hidden_dim, echo_dim)

        # 5. Outgoing Time Head: consumes z_t directly
        self.time_mlp = nn.Sequential(
            nn.Linear(d_latent, hidden_dim),
            nn.GELU(),
        )
        self.H_delta_gate = nn.Linear(hidden_dim, 1)
        self.H_delta_reg = nn.Linear(hidden_dim, 1)

    def forward(self, z: torch.Tensor) -> DecoderOutputs:
        """
        Executes hierarchical decoding for a batch of latent vectors.
        
        Args:
            z: Latent tensor of shape [B, L, d_latent]
        Returns:
            DecoderOutputs namedtuple containing all predicted logits and continuous regressions.
        """
        # Align z to hidden_dim if dimensions differ
        z_hid = self.z_to_hidden(z)  # [B, L, hidden_dim]

        # 1. Category
        f_cat = self.F_cat(z)                 # [B, L, hidden_dim]
        c_logits = self.H_cat(f_cat)          # [B, L, num_categories]

        # 2. Specifics
        f_spec = self.F_spec(z_hid + f_cat)   # [B, L, hidden_dim]
        pres_logits = self.H_pres(f_spec)     # [B, L, 1]
        s_vec = self.H_spec(f_spec)           # [B, L, text_dim]

        # 3. Modality
        m_features = z_hid + f_spec           # [B, L, hidden_dim]
        m_logits = self.H_m(m_features)       # [B, L, num_modalities]

        # 4. Modality Values
        num_vec = self.H_v_numeric(m_features) # [B, L, 44]
        cxr_vec = self.H_v_cxr(m_features)     # [B, L, 768]
        text_vec = self.H_v_text(m_features)   # [B, L, 768]
        ecg_vec = self.H_v_ecg(m_features)     # [B, L, 192]
        echo_vec = self.H_v_echo(m_features)   # [B, L, 512]

        # 5. Time (outgoing gap)
        t_feat = self.time_mlp(z)             # [B, L, hidden_dim]
        time_gate = self.H_delta_gate(t_feat) # [B, L, 1]
        log_gap = self.H_delta_reg(t_feat)    # [B, L, 1]

        return DecoderOutputs(
            category_logits=c_logits,
            specifics_presence_logits=pres_logits,
            specifics_vec=s_vec,
            modality_logits=m_logits,
            numeric_vec=num_vec,
            cxr_vec=cxr_vec,
            text_vec=text_vec,
            ecg_vec=ecg_vec,
            echo_vec=echo_vec,
            time_gate_logits=time_gate,
            log_gap_reg=log_gap,
        )

    def decode_time(
        self,
        time_gate_logits: torch.Tensor,
        log_gap_reg: torch.Tensor,
        gate_threshold: float = 0.5,
    ) -> torch.Tensor:
        """
        DEC-03: Hard zero hurdle at inference.
        If gate sigmoid <= gate_threshold: delta = 0.0.
        Else: delta = expm1(clamp(log_gap_reg, min=0.0)).
        """
        gate_prob = torch.sigmoid(time_gate_logits)
        is_positive = gate_prob > gate_threshold
        # Clamp log gap to non-negative so expm1 >= 0
        clamped_log = torch.clamp(log_gap_reg, min=0.0)
        gap = torch.where(is_positive, torch.expm1(clamped_log), torch.zeros_like(clamped_log))
        return gap.squeeze(-1) if gap.shape[-1] == 1 else gap
