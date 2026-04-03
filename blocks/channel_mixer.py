"""ChannelMixer 블록 — 변수(채널)축 mixing (v2).

모든 channel mixer: (B, H, d_model) → (B, H, d_model)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseChannelMixer


class FeatureMLPMix(BaseChannelMixer):
    """TSMixer IC — Feature 축 MLP mixing. 변수간 상관이 높을 때."""

    def __init__(self, d_model: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.norm(x))


class InvertedAttentionMix(BaseChannelMixer):
    """iTransformer — 변수를 토큰으로 취급한 Attention.

    변수 수가 많고 상호작용이 복잡할 때 SOTA.
    """

    def __init__(self, d_model: int, n_heads: int = 4,
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, d_model) — d_model 차원이 변수 정보를 담고 있음
        return self.encoder(x)


CHANNEL_MIXER_REGISTRY: dict[str, type[BaseChannelMixer]] = {
    "FeatureMLPMix": FeatureMLPMix,
    "InvertedAttentionMix": InvertedAttentionMix,
}
