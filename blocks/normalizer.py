"""Normalizer 블록 — 인스턴스/통계 정규화 + 역변환.

모든 normalizer: (B, T, C) → (B, T, C), reverse 포함.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseNormalizer


class RevIN(BaseNormalizer):
    """Reversible Instance Normalization (Kim et al., 2022).

    비정상 시계열의 분포 shift를 instance-level로 정규화하고,
    예측 후 역변환으로 원래 스케일 복원.
    """

    def __init__(self, n_features: int, affine: bool = True,
                 subtract_last: bool = False, eps: float = 1e-5):
        super().__init__()
        self.n_features = n_features
        self.affine = affine
        self.subtract_last = subtract_last
        self.eps = eps

        if affine:
            self.affine_weight = nn.Parameter(torch.ones(n_features))
            self.affine_bias = nn.Parameter(torch.zeros(n_features))

        self._mean: torch.Tensor | None = None
        self._std: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        if self.subtract_last:
            self._mean = x[:, -1:, :].detach()
        else:
            self._mean = x.mean(dim=1, keepdim=True).detach()
        self._std = (x.std(dim=1, keepdim=True) + self.eps).detach()

        x_norm = (x - self._mean) / self._std

        if self.affine:
            x_norm = x_norm * self.affine_weight + self.affine_bias

        return x_norm

    def reverse(self, pred: torch.Tensor, target_idx: int | None = None) -> torch.Tensor:
        """예측값을 원래 스케일로 역변환.

        Args:
            pred: (B, H, output_dim)
            target_idx: output_dim=1일 때 어떤 채널의 통계를 쓸지. None이면 전체.
        """
        if self._mean is None or self._std is None:
            return pred

        if target_idx is not None:
            mean = self._mean[:, :, target_idx:target_idx + 1]
            std = self._std[:, :, target_idx:target_idx + 1]
        else:
            mean = self._mean
            std = self._std

        if self.affine and target_idx is not None:
            w = self.affine_weight[target_idx:target_idx + 1]
            b = self.affine_bias[target_idx:target_idx + 1]
            pred = (pred - b) / (w + self.eps)
        elif self.affine:
            pred = (pred - self.affine_bias) / (self.affine_weight + self.eps)

        return pred * std + mean


class RobustScaler(BaseNormalizer):
    """Robust Scaler — IQR 기반 정규화. 이상치에 강건."""

    def __init__(self, n_features: int, q_low: float = 25.0, q_high: float = 75.0):
        super().__init__()
        self.n_features = n_features
        self.q_low = q_low
        self.q_high = q_high
        self._median: torch.Tensor | None = None
        self._iqr: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        self._median = x.median(dim=1, keepdim=True).values.detach()
        q_lo = torch.quantile(x, self.q_low / 100.0, dim=1, keepdim=True).detach()
        q_hi = torch.quantile(x, self.q_high / 100.0, dim=1, keepdim=True).detach()
        self._iqr = (q_hi - q_lo + 1e-8).detach()

        return (x - self._median) / self._iqr

    def reverse(self, pred: torch.Tensor, target_idx: int | None = None) -> torch.Tensor:
        if self._median is None or self._iqr is None:
            return pred
        if target_idx is not None:
            median = self._median[:, :, target_idx:target_idx + 1]
            iqr = self._iqr[:, :, target_idx:target_idx + 1]
        else:
            median = self._median
            iqr = self._iqr
        return pred * iqr + median


class BatchInstanceNorm(BaseNormalizer):
    """Batch-Instance Normalization — BN/IN learnable interpolation.

    gate = sigmoid(rho)
    output = gate * BN(x) + (1-gate) * IN(x)

    RevIN 대비 배치 통계를 함께 활용.
    배치 간 분포가 유사하면 BN이 유리, instance마다 다르면 IN이 유리.
    학습 가능한 gate가 자동으로 최적 비율을 결정.
    """

    def __init__(self, n_features: int, eps: float = 1e-5):
        super().__init__()
        self.n_features = n_features
        self.eps = eps

        # learnable interpolation parameter
        self.rho = nn.Parameter(torch.zeros(1, 1, n_features))

        # BatchNorm (time축 기준)
        self.bn = nn.BatchNorm1d(n_features, eps=eps)

        # reverse용 통계 저장
        self._mean: torch.Tensor | None = None
        self._std: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        # instance statistics
        self._mean = x.mean(dim=1, keepdim=True).detach()
        self._std = (x.std(dim=1, keepdim=True) + self.eps).detach()
        x_in = (x - self._mean) / self._std

        # batch statistics (BatchNorm1d expects (B, C, T))
        x_bn = self.bn(x.permute(0, 2, 1)).permute(0, 2, 1)

        # interpolation
        gate = torch.sigmoid(self.rho)
        return gate * x_bn + (1.0 - gate) * x_in

    def reverse(self, pred: torch.Tensor, target_idx: int | None = None) -> torch.Tensor:
        """역변환 — instance 통계 기준으로 복원."""
        if self._mean is None or self._std is None:
            return pred
        if target_idx is not None:
            mean = self._mean[:, :, target_idx:target_idx + 1]
            std = self._std[:, :, target_idx:target_idx + 1]
        else:
            mean = self._mean
            std = self._std
        return pred * std + mean


NORMALIZER_REGISTRY: dict[str, type[BaseNormalizer]] = {
    "RevIN": RevIN,
    "RobustScaler": RobustScaler,
    "BatchInstanceNorm": BatchInstanceNorm,
}
