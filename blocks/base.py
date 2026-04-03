"""블록 베이스 클래스 — 모든 블록이 준수하는 텐서 인터페이스."""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


# ── Encoder: (B, T, raw_features) → (B, T, d_model) ──

class BaseEncoder(nn.Module, ABC):
    """입력 피쳐를 d_model 차원으로 변환."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, raw_features) → (B, T, d_model)"""
        ...


# ── Decomposer: (B, T, d_model) → List[(B, T, d_model)] ──

class BaseDecomposer(nn.Module, ABC):
    """시계열을 여러 컴포넌트로 분해."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """(B, T, d_model) → List[(B, T, d_model)]"""
        ...


# ── Backbone: (B, T, d_model) → (B, H, 1) ──

class BaseBackbone(nn.Module, ABC):
    """핵심 예측 모델. d_model → prediction_length."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, d_model) → (B, H, 1)"""
        ...


# ── RegimeGate: Backbone wrapper ──

class BaseRegimeGate(nn.Module, ABC):
    """Backbone N개를 감싸고, regime에 따라 가중합."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, d_model) → (B, H, 1)  — 내부에서 backbone 호출."""
        ...


# ── Constraint: (B, H, 1) → (B, H, 1) ──

class BaseConstraint(nn.Module, ABC):
    """출력에 물리/도메인 제약 적용."""

    @abstractmethod
    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        """(B, H, 1) → (B, H, 1)"""
        ...


# ── Loss: (pred, target) → scalar ──

class BaseLoss(nn.Module, ABC):
    """예측 손실."""

    @abstractmethod
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """(B, H, 1), (B, H, 1) → scalar"""
        ...
