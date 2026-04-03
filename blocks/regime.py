"""Regime 블록 — 다중 regime 처리.

RegimeGate: backbone N개를 감싸고, 입력에 따라 가중합.
시그니처: (B, T, d_model) → (B, H, 1)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseRegimeGate, BaseBackbone


class RegimeGate(BaseRegimeGate):
    """Soft regime gate — N개의 backbone을 입력 기반으로 가중합.

    각 regime에 독립 backbone을 두고, gating network가 가중치를 결정.
    """

    def __init__(self, backbones: list[BaseBackbone], seq_len: int, d_model: int):
        super().__init__()
        self.n_regimes = len(backbones)
        self.backbones = nn.ModuleList(backbones)

        # Gating network: 입력 시퀀스 → regime 확률
        self.gate = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(seq_len * d_model, 64),
            nn.GELU(),
            nn.Linear(64, self.n_regimes),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gating weights: (B, n_regimes)
        weights = self.gate(x)

        # 각 backbone의 예측
        preds = torch.stack([bb(x) for bb in self.backbones], dim=-1)  # (B, H, 1, n_regimes)
        preds = preds.squeeze(2)  # (B, H, n_regimes)

        # 가중합
        weights = weights.unsqueeze(1)  # (B, 1, n_regimes)
        out = (preds * weights).sum(dim=-1, keepdim=True)  # (B, H, 1)
        return out


class HardRegimeGate(BaseRegimeGate):
    """Hard regime gate — argmax로 단일 backbone 선택 (추론용)."""

    def __init__(self, backbones: list[BaseBackbone], seq_len: int, d_model: int):
        super().__init__()
        self.n_regimes = len(backbones)
        self.backbones = nn.ModuleList(backbones)

        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(seq_len * d_model, 64),
            nn.GELU(),
            nn.Linear(64, self.n_regimes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # 학습 시: Gumbel-Softmax로 미분 가능
            logits = self.classifier(x)
            weights = nn.functional.gumbel_softmax(logits, tau=1.0, hard=False)
            preds = torch.stack([bb(x) for bb in self.backbones], dim=-1).squeeze(2)
            weights = weights.unsqueeze(1)
            return (preds * weights).sum(dim=-1, keepdim=True)
        else:
            # 추론 시: argmax
            logits = self.classifier(x)
            idx = logits.argmax(dim=-1)  # (B,)
            preds = torch.stack([bb(x) for bb in self.backbones], dim=-1).squeeze(2)  # (B, H, n)
            # 각 배치에서 선택된 backbone만
            B = x.shape[0]
            return preds[torch.arange(B), :, idx].unsqueeze(-1)


REGIME_REGISTRY: dict[str, type[BaseRegimeGate]] = {
    "SoftGate": RegimeGate,
    "HardGate": HardRegimeGate,
}
