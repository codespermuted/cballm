"""블록 베이스 클래스 — 모든 블록이 준수하는 텐서 인터페이스 (v2)."""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


# ── Normalizer: (B, T, C) → (B, T, C), has_reverse ──

class BaseNormalizer(nn.Module, ABC):
    """인스턴스/통계 정규화. reverse로 원래 스케일 복원."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, C) → (B, T, C)"""
        ...

    @abstractmethod
    def reverse(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H, output_dim) → (B, H, output_dim)"""
        ...


# ── Encoder: (B, T, C) → (B, T, d_model) or (B, n_patch, d_model) ──

class BaseEncoder(nn.Module, ABC):
    """입력 피쳐를 d_model 차원으로 변환."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, C) → (B, T, d_model) or (B, n_patch, d_model)"""
        ...


# ── TemporalMixer: (B, T|n_patch, d_model) → (B, H, d_model) ──

class BaseTemporalMixer(nn.Module, ABC):
    """시간축 mixing. 입력 시퀀스를 예측 길이로 변환."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T|n_patch, d_model) → (B, H, d_model)"""
        ...


# ── ChannelMixer: (B, H, d_model) → (B, H, d_model) ──

class BaseChannelMixer(nn.Module, ABC):
    """변수(채널)축 mixing. 변수 간 상호작용을 학습."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H, d_model) → (B, H, d_model)"""
        ...


# ── Head: (B, H, d_model) → (B, H, output_dim) ──

class BaseHead(nn.Module, ABC):
    """최종 출력 projection."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H, d_model) → (B, H, output_dim)"""
        ...


# ── Constraint: (B, H, output_dim) → (B, H, output_dim) ──

class BaseConstraint(nn.Module, ABC):
    """출력에 물리/도메인 제약 적용."""

    @abstractmethod
    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        """(B, H, output_dim) → (B, H, output_dim)"""
        ...


# ── Loss: (pred, target) → scalar ──

class BaseLoss(nn.Module, ABC):
    """예측 손실."""

    @abstractmethod
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """(B, H, output_dim), (B, H, output_dim) → scalar"""
        ...


# ── v1 호환 ──

BaseBackbone = BaseTemporalMixer  # v1의 Backbone은 v2의 TemporalMixer
BaseRegimeGate = BaseTemporalMixer  # v1 호환
BaseDecomposer = BaseEncoder  # v1 호환 placeholder
