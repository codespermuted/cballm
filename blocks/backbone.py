"""Backbone 블록 — 핵심 예측 모듈들.

모든 backbone: (B, T, d_model) → (B, H, 1)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .base import BaseBackbone


class LinearBackbone(BaseBackbone):
    """단순 선형. DLinear 스타일."""

    def __init__(self, seq_len: int, pred_len: int, d_model: int):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)  # (B, T*d_model)
        self.linear = nn.Linear(seq_len * d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(self.flatten(x))  # (B, H)
        return out.unsqueeze(-1)  # (B, H, 1)


class MLPBackbone(BaseBackbone):
    """Channel-independent MLP — 각 feature 독립 처리 후 합산.

    전체 flatten 대신, 각 d_model 채널을 독립적으로 seq_len → pred_len 매핑.
    DLinear과 유사하지만 비선형 레이어 추가.
    """

    def __init__(self, seq_len: int, pred_len: int, d_model: int,
                 hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        # 각 채널 독립: seq_len → pred_len (DLinear 스타일)
        self.channel_mlp = nn.Sequential(
            nn.Linear(seq_len, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_len),
        )
        # 채널 합산: d_model → 1
        self.channel_mix = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        x = x.permute(0, 2, 1)           # (B, d_model, T)
        x = self.channel_mlp(x)          # (B, d_model, H)
        x = x.permute(0, 2, 1)           # (B, H, d_model)
        return self.channel_mix(x)        # (B, H, 1)


class PatchMLPBackbone(BaseBackbone):
    """PatchTST 스타일 patching + channel-independent MLP.

    패치 단위로 local pattern을 포착한 뒤 MLP로 예측.
    """

    def __init__(self, seq_len: int, pred_len: int, d_model: int,
                 patch_len: int = 16, stride: int = 8, hidden_dim: int = 256):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = (seq_len - patch_len) // stride + 1

        # patch → d_model projection
        self.patch_proj = nn.Linear(patch_len * d_model, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

        # flatten patches → prediction
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(self.n_patches * hidden_dim, pred_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        # unfold patches: (B, n_patches, patch_len, D)
        patches = x.unfold(1, self.patch_len, self.stride)  # (B, n_patches, D, patch_len)
        patches = patches.permute(0, 1, 3, 2).reshape(B, self.n_patches, -1)  # (B, n_patches, patch_len*D)

        patches = self.norm(self.patch_proj(patches))  # (B, n_patches, hidden)
        return self.head(patches).unsqueeze(-1)  # (B, H, 1)


class TransformerBackbone(BaseBackbone):
    """Lightweight Transformer encoder + channel-independent head."""

    def __init__(self, seq_len: int, pred_len: int, d_model: int,
                 n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        # Channel-independent projection: 각 d_model 채널을 T → H 매핑
        self.temporal_proj = nn.Linear(seq_len, pred_len)
        self.channel_mix = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc = self.encoder(x)             # (B, T, d_model)
        enc = enc.permute(0, 2, 1)        # (B, d_model, T)
        out = self.temporal_proj(enc)      # (B, d_model, H)
        out = out.permute(0, 2, 1)        # (B, H, d_model)
        return self.channel_mix(out)      # (B, H, 1)


# ── Registry ──

BACKBONE_REGISTRY: dict[str, type[BaseBackbone]] = {
    "Linear": LinearBackbone,
    "PatchMLP": PatchMLPBackbone,
    # MLP, Transformer: 학습 불안정 (epoch 0 early stop). 단독 품질 검증 후 복귀 예정.
}
