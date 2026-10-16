"""
Multimodal Input Projections for NOAH (ARCH-01, NUM-01, NUM-02).

Projects diverse clinical inputs into the unified model hidden dimension d_model:
1. Category Embedding (31 clinical categories)
2. Modality Type Embedding (6 modalities: categorical, numeric, cxr, text, ecg, echo)
3. Text Specifics Projection (BioLORD-2023, 768-D -> d_model)
4. Modality Value Projections:
   - Numeric: 44-D dyadic Fourier features -> d_model
   - CXR: 768-D RAD-DINO CLS -> d_model
   - Text: 768-D BioLORD text embedding -> d_model
   - ECG: 192-D OTIS patch-mean -> d_model
   - ECHO: 512-D EchoPrime CLS (plug-and-play hook) -> d_model
"""

import math
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.schema import EventCategory, Modality


class ModalityInputProjector(nn.Module):
    """
    Unified input projection module mapping multimodal clinical tokens to d_model.
    """
    def __init__(
        self,
        d_model: int = 256,
        num_categories: int = 31,
        num_modalities: int = 6,
        numeric_dim: int = 44,
        text_dim: int = 768,
        cxr_dim: int = 768,
        ecg_dim: int = 192,
        echo_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_categories = num_categories
        self.num_modalities = num_modalities

        # 1. Discrete embeddings
        self.category_embed = nn.Embedding(num_categories, d_model)
        self.modality_embed = nn.Embedding(num_modalities, d_model)

        # 2. Specifics projection (clinical text phrases/names)
        self.specifics_proj = nn.Sequential(
            nn.Linear(text_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # 3. Modality-specific value projections
        self.numeric_proj = nn.Linear(numeric_dim, d_model)
        self.text_proj = nn.Linear(text_dim, d_model)
        self.cxr_proj = nn.Linear(cxr_dim, d_model)
        self.ecg_proj = nn.Linear(ecg_dim, d_model)
        self.echo_proj = nn.Linear(echo_dim, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        category_ids: torch.Tensor,       # [B, L]
        modality_ids: torch.Tensor,       # [B, L]
        specifics_vec: torch.Tensor,      # [B, L, 768]
        has_specifics: torch.Tensor,      # [B, L] bool
        value_vec: torch.Tensor,          # [B, L, max_val_dim]
        has_value: torch.Tensor,          # [B, L] bool
    ) -> torch.Tensor:
        """
        Embeds and sums categorical, modality, specifics, and value features.
        
        Returns:
            Tensor of shape [B, L, d_model]
        """
        B, L = category_ids.shape

        # 1. Category and Modality base embeddings
        e_cat = self.category_embed(category_ids)    # [B, L, d_model]
        e_mod = self.modality_embed(modality_ids)    # [B, L, d_model]

        # 2. Specifics projection (masked by has_specifics)
        # Avoid running specifics_proj on all tokens if possible, or project all and mask
        e_spec_raw = self.specifics_proj(specifics_vec)  # [B, L, d_model]
        spec_mask = has_specifics.unsqueeze(-1).float()  # [B, L, 1]
        e_spec = e_spec_raw * spec_mask

        # 3. Modality Value Projections
        # value_vec is padded to max_val_dim (e.g. 768).
        # We route by modality_id for tokens where has_value is True
        e_val = torch.zeros(B, L, self.d_model, device=category_ids.device, dtype=e_cat.dtype)

        # Boolean masks per modality with active values
        val_mask = has_value.bool()

        # Numeric (Modality.NUMERIC = 1) -> uses first 44 dimensions
        num_mask = val_mask & (modality_ids == Modality.NUMERIC)
        if num_mask.any():
            num_inputs = value_vec[num_mask][:, :44]
            e_val[num_mask] = self.numeric_proj(num_inputs)

        # CXR (Modality.CXR = 2) -> uses 768 dimensions
        cxr_mask = val_mask & (modality_ids == Modality.CXR)
        if cxr_mask.any():
            cxr_inputs = value_vec[cxr_mask][:, :768]
            e_val[cxr_mask] = self.cxr_proj(cxr_inputs)

        # Text (Modality.TEXT = 3) -> uses 768 dimensions
        text_mask = val_mask & (modality_ids == Modality.TEXT)
        if text_mask.any():
            text_inputs = value_vec[text_mask][:, :768]
            e_val[text_mask] = self.text_proj(text_inputs)

        # ECG (Modality.ECG = 4) -> uses 192 dimensions
        ecg_mask = val_mask & (modality_ids == Modality.ECG)
        if ecg_mask.any():
            ecg_inputs = value_vec[ecg_mask][:, :192]
            e_val[ecg_mask] = self.ecg_proj(ecg_inputs)

        # ECHO (Modality.ECHO = 5) -> uses 512 dimensions
        echo_mask = val_mask & (modality_ids == Modality.ECHO)
        if echo_mask.any():
            echo_inputs = value_vec[echo_mask][:, :512]
            e_val[echo_mask] = self.echo_proj(echo_inputs)

        # Combine all representations
        e_total = e_cat + e_mod + e_spec + e_val
        e_total = self.layer_norm(e_total)
        e_total = self.dropout(e_total)

        return e_total
