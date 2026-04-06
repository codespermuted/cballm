"""Head 블록 — 최종 출력 projection (v2.1).

모든 head: (B, H, d_model) → (B, H, output_dim)

DistributionalHead: 분포 기반 확률 예측.
  distribution 옵션: gaussian, student_t, log_normal, mixture_gaussian
  학습 시: (params_dict) 반환 → 대응 NLL loss와 연동
  추론 시 point_forecast=True: mean만 반환 (기존 인터페이스)
"""
from __future__ import annotations

import math

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


class DistributionalHead(BaseHead):
    """분포 기반 확률 예측 Head.

    distribution 옵션:
      - "gaussian":          mean + std (기본)
      - "student_t":         mean + std + df (heavy tail)
      - "log_normal":        mean + std (양수 제약 + right skew)
      - "mixture_gaussian":  k개 (mean, std, weight) (multi-modal)

    학습 시: dict {"mean": ..., "std": ..., ...} 반환 → 대응 NLL loss와 연동
    추론 시 point_forecast=True: mean만 반환 (기존 LinearHead 인터페이스)
    """

    def __init__(self, d_model: int, output_dim: int = 1,
                 distribution: str = "gaussian",
                 point_forecast: bool = True,
                 n_components: int = 3):
        super().__init__()
        self.distribution = distribution
        self.point_forecast = point_forecast
        self.output_dim = output_dim
        self.n_components = n_components
        self.softplus = nn.Softplus()

        if distribution == "gaussian":
            self.mean_proj = nn.Linear(d_model, output_dim)
            self.std_proj = nn.Linear(d_model, output_dim)

        elif distribution == "student_t":
            self.mean_proj = nn.Linear(d_model, output_dim)
            self.std_proj = nn.Linear(d_model, output_dim)
            self.df_proj = nn.Linear(d_model, output_dim)

        elif distribution == "log_normal":
            self.mu_proj = nn.Linear(d_model, output_dim)
            self.sigma_proj = nn.Linear(d_model, output_dim)

        elif distribution == "mixture_gaussian":
            k = n_components
            self.mean_proj = nn.Linear(d_model, output_dim * k)
            self.std_proj = nn.Linear(d_model, output_dim * k)
            self.weight_proj = nn.Linear(d_model, k)

        else:
            raise ValueError(f"Unknown distribution: {distribution}")

    def forward(self, x: torch.Tensor) -> torch.Tensor | dict:
        """(B, H, d_model) → (B, H, output_dim) or dict of params."""
        if self.distribution == "gaussian":
            return self._forward_gaussian(x)
        elif self.distribution == "student_t":
            return self._forward_student_t(x)
        elif self.distribution == "log_normal":
            return self._forward_log_normal(x)
        elif self.distribution == "mixture_gaussian":
            return self._forward_mixture(x)

    def _forward_gaussian(self, x: torch.Tensor):
        mean = self.mean_proj(x)
        std = self.softplus(self.std_proj(x)) + 1e-6

        if self.training or not self.point_forecast:
            return {"distribution": "gaussian", "mean": mean, "std": std}
        return mean

    def _forward_student_t(self, x: torch.Tensor):
        mean = self.mean_proj(x)
        std = self.softplus(self.std_proj(x)) + 1e-6
        # df > 2 for finite variance, softplus + 2.1 ensures this
        df = self.softplus(self.df_proj(x)) + 2.1

        if self.training or not self.point_forecast:
            return {"distribution": "student_t", "mean": mean, "std": std, "df": df}
        return mean

    def _forward_log_normal(self, x: torch.Tensor):
        mu = self.mu_proj(x)
        sigma = self.softplus(self.sigma_proj(x)) + 1e-6

        if self.training or not self.point_forecast:
            return {"distribution": "log_normal", "mu": mu, "sigma": sigma}
        # point forecast: E[X] = exp(mu + sigma^2/2)
        return torch.exp(mu + sigma ** 2 / 2)

    def _forward_mixture(self, x: torch.Tensor):
        B, H, _ = x.shape
        k = self.n_components
        od = self.output_dim

        means = self.mean_proj(x).view(B, H, od, k)       # (B, H, od, k)
        stds = self.softplus(self.std_proj(x)).view(B, H, od, k) + 1e-6
        weights = torch.softmax(self.weight_proj(x), dim=-1)  # (B, H, k)

        if self.training or not self.point_forecast:
            return {
                "distribution": "mixture_gaussian",
                "means": means, "stds": stds, "weights": weights,
            }
        # point forecast: weighted mean
        # weights: (B,H,k) → (B,H,1,k)
        w = weights.unsqueeze(2)
        return (means * w).sum(dim=-1)  # (B, H, od)


HEAD_REGISTRY: dict[str, type[BaseHead]] = {
    "LinearHead": LinearHead,
    "FlattenLinearHead": FlattenLinearHead,
    "DistributionalHead": DistributionalHead,
}
