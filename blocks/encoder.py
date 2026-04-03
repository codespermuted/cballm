"""Encoder 블록 — 입력 피쳐를 d_model 차원으로 변환 (v2).

LinearProjection: (B, T, C) → (B, T, d_model)
PatchEmbedding:   (B, T, C) → (B, n_patch, d_model)  ← shape이 다름!
FourierEmbedding: (B, T, C) → (B, T, d_model)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .base import BaseEncoder


class LinearProjection(BaseEncoder):
    """단순 선형 projection. DLinear 스타일."""

    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class PatchEmbedding(BaseEncoder):
    """PatchTST 스타일 패치 임베딩.

    시퀀스를 겹치는 패치로 분할 후 d_model로 projection.
    출력 shape이 (B, n_patch, d_model)로 다름에 주의.
    """

    def __init__(self, n_features: int, d_model: int,
                 patch_len: int = 16, stride: int = 8):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.proj = nn.Linear(patch_len * n_features, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        # unfold: (B, n_patch, C, patch_len)
        patches = x.unfold(1, self.patch_len, self.stride)
        # (B, n_patch, C, patch_len) → (B, n_patch, patch_len * C)
        patches = patches.permute(0, 1, 3, 2).reshape(
            B, patches.shape[1], self.patch_len * C
        )
        return self.norm(self.proj(patches))

    @property
    def output_is_patch(self) -> bool:
        return True


class FourierEmbedding(BaseEncoder):
    """Fourier basis로 시간 피쳐를 인코딩 + linear projection."""

    def __init__(self, n_features: int, d_model: int,
                 n_harmonics: int = 3, learnable_freq: bool = False):
        super().__init__()
        self.n_harmonics = n_harmonics
        fourier_dim = n_features * n_harmonics * 2
        self.proj = nn.Linear(n_features + fourier_dim, d_model)

        if learnable_freq:
            self.freq = nn.Parameter(
                torch.arange(1, n_harmonics + 1, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "freq",
                torch.arange(1, n_harmonics + 1, dtype=torch.float32),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = [x]
        for k in self.freq:
            parts.append(torch.sin(2 * math.pi * k * x))
            parts.append(torch.cos(2 * math.pi * k * x))
        return self.proj(torch.cat(parts, dim=-1))


ENCODER_REGISTRY: dict[str, type[BaseEncoder]] = {
    "LinearProjection": LinearProjection,
    "PatchEmbedding": PatchEmbedding,
    "FourierEmbedding": FourierEmbedding,
    # v1 호환
    "Linear": LinearProjection,
    "Fourier": FourierEmbedding,
}
