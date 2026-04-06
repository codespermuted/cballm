"""Decomposition 블록 — 시계열 분해 (trend/seasonal 분리).

MovingAvgDecomp: DLinear 논문의 moving average decomposition.
  Input (B, T, C) → seasonal (B, T, C), trend (B, T, C)

Builder에서 decomposer 슬롯으로 사용.
decomposer가 있으면 ForecastModel이 trend/seasonal을 각각 mixer에 보내고 합산.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MovingAvgDecomp(nn.Module):
    """DLinear의 Moving Average Decomposition.

    trend = AvgPool1d(x, kernel_size)  (양쪽 패딩으로 길이 보존)
    seasonal = x - trend

    kernel_size 가이드:
    - dominant_period가 있으면 kernel_size = dominant_period
    - 없으면 kernel_size = 25 (DLinear 논문 기본값)
    - 항상 홀수로 강제
    """

    def __init__(self, kernel_size: int = 25):
        super().__init__()
        # 홀수 강제
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = kernel_size
        self.avg_pool = nn.AvgPool1d(
            kernel_size=kernel_size, stride=1,
            padding=0,  # 수동 패딩
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, T, C) → (seasonal: (B,T,C), trend: (B,T,C))"""
        # AvgPool1d expects (B, C, T)
        B, T, C = x.shape
        pad = self.kernel_size // 2
        # causal padding: 왼쪽(과거)만 replicate, 오른쪽(미래)은 0이 아닌 마지막 값 복제
        # → 미래 정보 누출 방지 (LEAK-safe)
        x_pad = x.permute(0, 2, 1)  # (B, C, T)
        x_pad = nn.functional.pad(x_pad, (pad * 2, 0), mode="replicate")
        trend = self.avg_pool(x_pad).permute(0, 2, 1)  # (B, T, C)

        seasonal = x - trend
        return seasonal, trend


DECOMPOSITION_REGISTRY: dict[str, type[nn.Module]] = {
    "MovingAvgDecomp": MovingAvgDecomp,
}
