"""Constraint 블록 — 출력에 물리/도메인 제약 적용.

모든 constraint: (B, H, 1) → (B, H, 1)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseConstraint


class PositivityConstraint(BaseConstraint):
    """출력이 항상 양수. Softplus 적용."""

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.softplus = nn.Softplus(beta=beta)

    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        return self.softplus(pred)


class ClampConstraint(BaseConstraint):
    """출력을 [min, max] 범위로 제한."""

    def __init__(self, min_val: float = float("-inf"), max_val: float = float("inf")):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        return torch.clamp(pred, self.min_val, self.max_val)


class MonotonicConstraint(BaseConstraint):
    """예측값이 단조 증가/감소하도록 강제.

    cumsum trick: 차분을 softplus로 비음수화 후 누적합.
    """

    def __init__(self, increasing: bool = True):
        super().__init__()
        self.increasing = increasing
        self.softplus = nn.Softplus()

    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        # pred: (B, H, 1)
        base = pred[:, 0:1, :]  # 시작점
        diffs = pred[:, 1:, :] - pred[:, :-1, :]  # 차분

        if self.increasing:
            diffs = self.softplus(diffs)  # 비음수
        else:
            diffs = -self.softplus(-diffs)  # 비양수

        monotonic = torch.cat([base, base + torch.cumsum(diffs, dim=1)], dim=1)
        return monotonic


class SmoothnessConstraint(BaseConstraint):
    """출력의 급격한 변화를 완화. Exponential smoothing 적용."""

    def __init__(self, alpha: float = 0.3):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        B, H, C = pred.shape
        smoothed = torch.zeros_like(pred)
        smoothed[:, 0, :] = pred[:, 0, :]
        for t in range(1, H):
            smoothed[:, t, :] = self.alpha * pred[:, t, :] + (1 - self.alpha) * smoothed[:, t-1, :]
        return smoothed


CONSTRAINT_REGISTRY: dict[str, type[BaseConstraint]] = {
    "Positivity": PositivityConstraint,
    "Clamp": ClampConstraint,
    "Monotonic": MonotonicConstraint,
    "Smoothness": SmoothnessConstraint,
}
