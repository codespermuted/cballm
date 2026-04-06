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


class VolatilityGate(BaseConstraint):
    """변동성 기반 gate — 고변동 구간의 예측을 감쇠/강화.

    raw_input에서 rolling std를 변동성 proxy로 계산하고,
    Sigmoid(Linear(vol))로 gate를 생성하여 예측에 곱함.

    window: rolling std 윈도우 (기본 pred_len)
    mode:
      "dampen" — 고변동 구간 예측을 감쇠 (보수적)
      "amplify" — 고변동 구간 예측을 강화
    """

    def __init__(self, d_model: int = 1, window: int = 24,
                 mode: str = "dampen"):
        super().__init__()
        self.window = window
        self.mode = mode
        self.gate_fc = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, pred: torch.Tensor,
                vol_feature: torch.Tensor | None = None) -> torch.Tensor:
        """(B, H, output_dim) → (B, H, output_dim)

        vol_feature가 없으면 pred 자체의 변동성으로 추정.
        """
        if vol_feature is not None:
            # vol_feature: (B, H, 1) — 외부에서 전달된 변동성 지표
            gate = self.gate_fc(vol_feature)
        else:
            # pred의 rolling 변동성을 proxy로
            B, H, C = pred.shape
            if H >= 3:
                # 간이 rolling std (kernel=3)
                k = min(3, H)
                unfolded = pred.unfold(1, k, 1)  # (B, H-k+1, C, k)
                vol = unfolded.std(dim=-1)  # (B, H-k+1, C)
                # padding으로 길이 맞추기
                pad_size = H - vol.shape[1]
                vol = nn.functional.pad(vol.permute(0, 2, 1), (pad_size, 0), mode="reflect")
                vol = vol.permute(0, 2, 1)  # (B, H, C)
            else:
                vol = torch.zeros_like(pred)

            # 채널 평균 → 단일 변동성
            vol_scalar = vol.mean(dim=-1, keepdim=True)  # (B, H, 1)
            gate = self.gate_fc(vol_scalar)

        if self.mode == "dampen":
            # 고변동 → gate 높음 → (1-gate) 적용 → 감쇠
            return pred * (1.0 - 0.5 * gate)
        else:
            # 고변동 → gate 높음 → 강화
            return pred * (0.5 + gate)


CONSTRAINT_REGISTRY: dict[str, type[BaseConstraint]] = {
    "Positivity": PositivityConstraint,
    "Clamp": ClampConstraint,
    "Monotonic": MonotonicConstraint,
    "Smoothness": SmoothnessConstraint,
    "VolatilityGate": VolatilityGate,
}
