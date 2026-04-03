"""Encoder 블록 — 입력 피쳐를 d_model 차원으로 변환.

모든 encoder: (B, T, raw_features) → (B, T, d_model)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .base import BaseEncoder


class LinearEncoder(BaseEncoder):
    """단순 선형 projection."""

    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class FourierTimeEncoder(BaseEncoder):
    """Fourier basis로 시간 피쳐를 인코딩 + linear projection.

    입력에서 raw time features(hour, day_of_week 등)를 sin/cos로 변환한 뒤
    나머지 피쳐와 합쳐서 d_model로 projection.
    """

    def __init__(self, n_features: int, d_model: int,
                 n_time_features: int = 0, n_harmonics: int = 3):
        super().__init__()
        self.n_time_features = n_time_features
        self.n_harmonics = n_harmonics

        # Fourier 변환 후 차원: time features × harmonics × 2(sin/cos) + 나머지 features
        fourier_dim = n_time_features * n_harmonics * 2
        total_dim = (n_features - n_time_features) + fourier_dim
        self.proj = nn.Linear(total_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_time_features > 0:
            time_feats = x[..., :self.n_time_features]  # (B, T, n_time)
            other_feats = x[..., self.n_time_features:]  # (B, T, n_other)

            # Fourier encoding
            fourier_parts = []
            for k in range(1, self.n_harmonics + 1):
                fourier_parts.append(torch.sin(2 * math.pi * k * time_feats))
                fourier_parts.append(torch.cos(2 * math.pi * k * time_feats))
            fourier = torch.cat(fourier_parts, dim=-1)  # (B, T, n_time*harmonics*2)

            x = torch.cat([other_feats, fourier], dim=-1)

        return self.proj(x)


class RevINEncoder(BaseEncoder):
    """Reversible Instance Normalization + projection.

    비정상 시계열에 효과적. 입력을 정규화하고, 예측 후 역변환.
    """

    def __init__(self, n_features: int, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.proj = nn.Linear(n_features, d_model)
        # affine parameters (learnable)
        self.affine_weight = nn.Parameter(torch.ones(n_features))
        self.affine_bias = nn.Parameter(torch.zeros(n_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Instance normalization along time axis
        self._mean = x.mean(dim=1, keepdim=True).detach()
        self._std = (x.std(dim=1, keepdim=True) + self.eps).detach()
        x_norm = (x - self._mean) / self._std
        x_norm = x_norm * self.affine_weight + self.affine_bias
        return self.proj(x_norm)

    def reverse(self, pred: torch.Tensor, target_idx: int = 0) -> torch.Tensor:
        """예측값을 원래 스케일로 역변환. (B, H, 1) → (B, H, 1)"""
        mean = self._mean[:, :, target_idx:target_idx+1]  # (B, 1, 1)
        std = self._std[:, :, target_idx:target_idx+1]
        return pred * std + mean


ENCODER_REGISTRY: dict[str, type[BaseEncoder]] = {
    "Linear": LinearEncoder,
    "Fourier": FourierTimeEncoder,
    # RevIN 비활성화: dataset-level normalization과 이중 적용 문제.
    # Trainer가 이미 train-set 기준 standardization을 하므로 RevIN 불필요.
    # 필요 시 Trainer의 normalization을 RevIN으로 교체하는 구조 변경 필요.
}
