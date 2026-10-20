"""Frozen OTIS encoder adapter with strict checkpoint-key accounting.

Matches author patch/transformer operations, including both learned attention
projections. Position adaptation must be explicitly selected and is a candidate.
Adapted OTIS operations: Copyright (c)2024 Özgün Turgut, MIT; see OTIS_LICENSE.txt.
Author evidence: oetu/otis revision30282e9, models_vit.py and util/{patch_embed,
transformer,pos_embed}. No classification or pretraining decoder output is used.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def temporal_positions(checkpoint, patches, *, policy):
    if checkpoint.ndim != 3 or checkpoint.shape[0] != 1 or patches < 1:
        raise ValueError('Invalid OTIS position table')
    old = checkpoint.shape[1] - 1
    if patches <= old:
        return checkpoint[:, :patches+1].clone()
    if policy == 'upstream_linear':
        extended = F.interpolate(checkpoint[:, 1:].transpose(1, 2), size=patches,
                                 mode='linear', align_corners=False).transpose(1, 2)
        return torch.cat((checkpoint[:, :1], extended), dim=1)
    if policy != 'analytic_extension':
        raise ValueError('Explicit OTIS temporal position policy required')
    width = checkpoint.shape[-1]
    omega = np.arange(width//2, dtype=np.float64) / (width/2.)
    omega = 1. / 10000**omega
    phase = np.einsum('m,d->md', np.arange(patches, dtype=np.float64), omega)
    values = np.concatenate((np.sin(phase), np.cos(phase)), axis=1)
    values = np.concatenate((np.zeros((1, width)), values)).astype(np.float32)
    result = torch.from_numpy(values).unsqueeze(0).to(checkpoint)
    if not torch.allclose(result[:, :old+1], checkpoint, atol=5e-6, rtol=1e-6):
        raise ValueError('Stored table is not the claimed analytic bank; cannot extend it')
    # Retain every saved entry exactly; only append analytically extended positions.
    result[:, :old+1] = checkpoint
    return result


def normalize_ecg(waveform, leads, *, lead_order, sample_rate):
    a = np.asarray(waveform, dtype=np.float64)
    if sample_rate != 500 or a.shape != (5000, 12) or len(set(leads)) != 12 or set(leads) != set(lead_order):
        raise ValueError('ECG requires known unique twelve leads at500Hz for10seconds')
    a = a[:, [leads.index(lead) for lead in lead_order]].T
    if not np.isfinite(a).all():
        raise ValueError('Nonfinite waveform; no fabricated zero leads')
    mean, std = a.mean(), a.std()
    if std <= 1e-9:
        raise ValueError('Constant waveform cannot be encoded as missing zeros')
    keep = np.abs((a-mean)/(std+1e-9)) <= 3.
    mean = a[keep].mean(); std = a[keep].std()
    if std <= 1e-9:
        raise ValueError('Degenerate robust waveform statistics')
    return np.clip((a-mean)/(std+1e-9), -4., 4.).astype(np.float32)[None]


class PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(1, 192, kernel_size=(1, 24), stride=(1, 24))
        self.norm = nn.LayerNorm(192)

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        return F.gelu(self.norm(x)).permute(0, 3, 1, 2)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(192, 576)
        self.mha = nn.MultiheadAttention(192, 3, batch_first=True)
        self.proj = nn.Linear(192, 192)

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        return self.proj(self.mha(q, k, v, need_weights=False)[0])


class Mlp(nn.Module):
    def __init__(self):
        super().__init__(); self.fc1 = nn.Linear(192, 768); self.fc2 = nn.Linear(768, 192)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(192, eps=1e-6); self.norm2 = nn.LayerNorm(192, eps=1e-6)
        self.attn = Attention(); self.mlp = Mlp()

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class OTISEncoder(nn.Module):
    def __init__(self, state, domain_offset, *, position_policy):
        super().__init__()
        self.patch_embed = PatchEmbed(); self.blocks = nn.ModuleList([Block() for _ in range(12)])
        self.norm = nn.LayerNorm(192, eps=1e-6)
        self.cls_token = nn.Parameter(torch.empty(1, 1, 192))
        self.pos_embed_x = nn.Parameter(torch.empty_like(state['pos_embed_x']), requires_grad=False)
        self.pos_embed_y = nn.Embedding(*state['pos_embed_y.weight'].shape, padding_idx=0)
        expected = set(self.state_dict()); extras = set(state)-expected
        allowed = ('decoder_', 'projector.', 'predictor.')
        if any(k != 'mask_token' and not k.startswith(allowed) for k in extras):
            raise ValueError('Unexpected non-encoder checkpoint keys')
        if expected-set(state):
            raise ValueError('Missing pretrained encoder weights')
        self.load_state_dict({k: state[k] for k in expected}, strict=True)
        if state['patch_embed.proj.weight'].shape != (192, 1, 1, 24):
            raise ValueError('Cannot substitute or reinitialize OTIS patches')
        self.excluded_pretraining_keys = sorted(extras)
        self.domain_offset = int(domain_offset)
        if self.domain_offset < 0 or self.domain_offset+12 >= self.pos_embed_y.num_embeddings:
            raise ValueError('Invalid trained ECG domain offsets')
        self.position_policy = position_policy
        self.register_buffer('extended_time', temporal_positions(self.pos_embed_x.detach(), 208, policy=position_policy))
        self.requires_grad_(False); self.eval()

    @classmethod
    def from_checkpoint(cls, path: Path, *, position_policy):
        unsafe = torch.serialization.get_unsafe_globals_in_checkpoint(path)
        if set(unsafe)-{'argparse.Namespace'}:
            raise ValueError('Unexpected serialized checkpoint globals')
        with torch.serialization.safe_globals([argparse.Namespace]):
            ckpt = torch.load(path, weights_only=True, map_location='cpu')
        if tuple(ckpt['domains'].get('ecg', ())) != (1, 12, 5000):
            raise ValueError('No trained twelve-lead ECG domain')
        return cls(ckpt['model'], ckpt['domain_offsets']['ecg'], position_policy=position_policy)

    def patches(self, waveform):
        if waveform.shape[1:] != (1, 12, 5000) or not torch.isfinite(waveform).all():
            raise ValueError('Invalid full ECG input')
        x = self.patch_embed(waveform)
        # Trained variate indices use offset+1 because embedding0 is padding.
        indices = torch.arange(1+self.domain_offset, 13+self.domain_offset, device=x.device)
        vertical = F.pad(self.pos_embed_y(indices), (0, 96)).T[None, :, :, None]
        horizontal = F.pad(self.extended_time[:, 1:], (96, 0)).transpose(1, 2)[:, :, None, :]
        x = (x+vertical+horizontal).flatten(2).transpose(1, 2)
        token = self.cls_token + F.pad(self.extended_time[:, :1], (96, 0))
        x = torch.cat((token.expand(x.shape[0], -1, -1), x), dim=1)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)[:, 1:]

    def forward(self, waveform):
        return self.patches(waveform).mean(dim=1)
