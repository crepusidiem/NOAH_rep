"""
Bidirectional Temporal Transformer Backbone for NOAH (ARCH-01, ARCH-02, ARCH-03, TIME-01..05).

Features:
- Causal multi-head self-attention with pre-LayerNorm.
- Directional Temporal Attention additions (Q^* = Q + T^Q, K^* = K + T^K) in layers 0-3.
- LayerScale initialized to 1e-4 on residual branches (ARCH-03).
- Strictly causal content masking: j <= i.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.temporal import BidirectionalTemporalEmbedding, TemporalFeatureBank


class DropPath(nn.Module):
    """Per-example stochastic residual depth; constant rate is assumption A-18."""
    def __init__(self, probability=0.0):
        super().__init__()
        if not 0 <= probability < 1:
            raise ValueError("drop_path must be in [0, 1)")
        self.probability = probability

    def forward(self, x):
        if not self.training or self.probability == 0:
            return x
        keep = 1 - self.probability
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


class TemporalAttentionLayer(nn.Module):
    """
    Multi-head causal self-attention layer with temporal Q/K addition.
    """
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        head_dim: int = 64,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        inject_temporal: bool = True,
        layer_scale_init: float = 1e-4,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.inner_dim = n_heads * head_dim
        self.inject_temporal = inject_temporal
        self.scale = 1.0 / math.sqrt(head_dim)

        self.q_proj = nn.Linear(d_model, self.inner_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.inner_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.inner_dim, bias=False)
        self.out_proj = nn.Linear(self.inner_dim, d_model, bias=False)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(dropout)

        # LayerScale parameter
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(d_model))

    def forward(
        self,
        x: torch.Tensor,                                      # [B, L, d_model]
        time_Q: Optional[torch.Tensor] = None,                # [B, L, head_dim]
        time_K: Optional[torch.Tensor] = None,                # [B, L, head_dim]
        causal_mask: Optional[torch.Tensor] = None,           # [L, L] bool
        padding_mask: Optional[torch.Tensor] = None,          # [B, L] bool (True = valid, False = pad)
    ) -> torch.Tensor:
        B, L, _ = x.shape

        # Linear projections
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, L, D]
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, L, D]
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, L, D]

        # Inject temporal features into Q and K in designated layers (ARCH-02)
        if self.inject_temporal and time_Q is not None and time_K is not None:
            # Broadcast over attention heads: time_Q shape [B, 1, L, D]
            q = q + time_Q.unsqueeze(1)
            k = k + time_K.unsqueeze(1)

        # Scaled dot-product attention via PyTorch SDPA (FlashAttention / memory-efficient SDPA)
        dropout_p = self.attn_dropout.p if self.training else 0.0

        if padding_mask is not None or causal_mask is not None:
            # When padding mask is provided, combine causal tril mask and key padding mask
            if causal_mask is None:
                causal_mask = torch.ones(L, L, device=x.device, dtype=torch.bool).tril()
            if padding_mask is not None:
                attn_mask = causal_mask.unsqueeze(0).unsqueeze(0) & padding_mask.unsqueeze(1).unsqueeze(2)
            else:
                attn_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p)
            out = torch.nan_to_num(out, nan=0.0)
        else:
            # Pure causal fast path (enables optimal FlashAttention triangular tiling)
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dropout_p)
        out = out.transpose(1, 2).contiguous().view(B, L, self.inner_dim)
        out = self.out_proj(out)
        out = self.proj_dropout(out)

        # LayerScale
        return self.gamma * out


class TransformerBlock(nn.Module):
    """
    Standard Transformer Decoder Block with Pre-LayerNorm and LayerScale.
    """
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        head_dim: int = 64,
        ffn_ratio: int = 4,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        inject_temporal: bool = True,
        layer_scale_init: float = 1e-4,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.drop_path = DropPath(drop_path)
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = TemporalAttentionLayer(
            d_model=d_model,
            n_heads=n_heads,
            head_dim=head_dim,
            dropout=dropout,
            attn_dropout=attn_dropout,
            inject_temporal=inject_temporal,
            layer_scale_init=layer_scale_init,
        )

        self.ln2 = nn.LayerNorm(d_model)
        ffn_dim = d_model * ffn_ratio
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.gamma_ffn = nn.Parameter(layer_scale_init * torch.ones(d_model))

    def forward(
        self,
        x: torch.Tensor,
        time_Q: Optional[torch.Tensor] = None,
        time_K: Optional[torch.Tensor] = None,
        causal_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-LN Self-Attention with residual connection
        normed1 = self.ln1(x)
        attn_out = self.attn(
            normed1,
            time_Q=time_Q,
            time_K=time_K,
            causal_mask=causal_mask,
            padding_mask=padding_mask,
        )
        x = x + self.drop_path(attn_out)

        # Pre-LN Feed-Forward Network with residual connection
        normed2 = self.ln2(x)
        ffn_out = self.gamma_ffn * self.ffn(normed2)
        x = x + self.drop_path(ffn_out)
        return x


class NOAHTransformerBackbone(nn.Module):
    """
    NOAH Causal Transformer Backbone.
    Computes patient history representations h_t.
    """
    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        head_dim: int = 64,
        ffn_ratio: int = 4,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        temporal_layers: int = 4,  # Layers 0..3 have temporal additions (ARCH-02)
        layer_scale_init: float = 1e-4,
        drop_path: float = 0.0,
        input_temporal: bool = True,
    ):
        super().__init__()
        self.input_time_bank = TemporalFeatureBank(4, d_model) if input_temporal else None
        self.d_model = d_model
        self.n_layers = n_layers
        self.temporal_layers = temporal_layers

        # Bidirectional Temporal Embedding bank
        self.temporal_emb = BidirectionalTemporalEmbedding(head_dim=head_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                head_dim=head_dim,
                ffn_ratio=ffn_ratio,
                dropout=dropout,
                attn_dropout=attn_dropout,
                inject_temporal=(i < temporal_layers),
                layer_scale_init=layer_scale_init,
                drop_path=drop_path,
            )
            for i in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        e: torch.Tensor,              # [B, L, d_model] Input token embeddings
        positions: torch.Tensor,      # [B, L]
        ages: torch.Tensor,           # [B, L]
        delta_next: torch.Tensor,     # [B, L]
        delta_prev: torch.Tensor,     # [B, L]
        attention_mask: torch.Tensor, # [B, L] bool (True = valid, False = pad)
    ) -> torch.Tensor:
        """
        Executes causal transformer and returns hidden state h of shape [B, L, d_model].
        """
        B, L, _ = e.shape
        device = e.device

        # Compute Q and K temporal additions
        time_Q, time_K = self.temporal_emb(
            positions=positions,
            ages=ages,
            delta_next=delta_next,
            delta_prev=delta_prev,
        )

        # Causal mask: tril matrix where j <= i is True
        causal_mask = torch.tril(torch.ones(L, L, device=device, dtype=torch.bool))

        x = e
        if self.input_time_bank is not None:
            x = x + self.input_time_bank(torch.stack([
                positions, ages, torch.log1p(delta_prev.clamp_min(0)),
                torch.log1p(delta_next.clamp_min(0)),
            ], dim=-1))
        for block in self.blocks:
            x = block(
                x=x,
                time_Q=time_Q,
                time_K=time_K,
                causal_mask=causal_mask,
                padding_mask=attention_mask,
            )

        h = self.final_norm(x)
        return h
