"""Loss 블록 — 예측 손실 함수.

모든 loss: (pred, target) → scalar
pred: (B, H, 1), target: (B, H, 1)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseLoss


class MAELoss(BaseLoss):
    """Mean Absolute Error."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (pred - target).abs().mean()


class MSELoss(BaseLoss):
    """Mean Squared Error."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return ((pred - target) ** 2).mean()


class HuberLoss(BaseLoss):
    """Huber Loss — MAE/MSE 하이브리드."""

    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.huber = nn.HuberLoss(delta=delta)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.huber(pred, target)


class QuantileLoss(BaseLoss):
    """Quantile Loss — 비대칭 손실."""

    def __init__(self, quantile: float = 0.5):
        super().__init__()
        self.q = quantile

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = target - pred
        return torch.max(self.q * error, (self.q - 1) * error).mean()


class AsymmetricLoss(BaseLoss):
    """비대칭 손실 — 과대/과소 예측에 다른 가중치."""

    def __init__(self, over_weight: float = 1.0, under_weight: float = 2.0):
        super().__init__()
        self.over_weight = over_weight
        self.under_weight = under_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = pred - target
        weights = torch.where(error > 0, self.over_weight, self.under_weight)
        return (weights * error.abs()).mean()


class SmoothnessRegLoss(BaseLoss):
    """기본 손실 + smoothness regularization."""

    def __init__(self, base_loss: BaseLoss | None = None, lambda_smooth: float = 0.1):
        super().__init__()
        self.base = base_loss or MAELoss()
        self.lambda_smooth = lambda_smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        base_loss = self.base(pred, target)
        # L2 on prediction differences (smoothness)
        diffs = pred[:, 1:, :] - pred[:, :-1, :]
        smooth_loss = (diffs ** 2).mean()
        return base_loss + self.lambda_smooth * smooth_loss


LOSS_REGISTRY: dict[str, type[BaseLoss]] = {
    "MAE": MAELoss,
    "MSE": MSELoss,
    "Huber": HuberLoss,
    "Quantile": QuantileLoss,
    "Asymmetric": AsymmetricLoss,
    "SmoothnessReg": SmoothnessRegLoss,
}
