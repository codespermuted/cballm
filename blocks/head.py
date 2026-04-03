"""Head 블록 — 최종 출력 projection (v2).

모든 head: (B, H, d_model) → (B, H, output_dim)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseHead


class LinearHead(BaseHead):
    """기본 Linear Head. head에서 복잡도 높이는 것은 거의 효과 없음."""

    def __init__(self, d_model: int, output_dim: int = 1):
        super().__init__()
        self.proj = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class FlattenLinearHead(BaseHead):
    """NHITS/NBEATS 스타일 Flatten + Linear."""

    def __init__(self, pred_len: int, d_model: int, output_dim: int = 1):
        super().__init__()
        self.pred_len = pred_len
        self.flatten = nn.Flatten(start_dim=1)
        self.linear = nn.Linear(pred_len * d_model, pred_len * output_dim)
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        out = self.linear(self.flatten(x))
        return out.view(B, self.pred_len, self.output_dim)


HEAD_REGISTRY: dict[str, type[BaseHead]] = {
    "LinearHead": LinearHead,
    "FlattenLinearHead": FlattenLinearHead,
}
