"""Loss 블록 — 예측 손실 함수 (v2.1).

모든 loss: (pred, target) → scalar
pred: (B, H, 1) 또는 dict (DistributionalHead 출력)
target: (B, H, 1)

분포 NLL 손실:
  GaussianNLL, StudentTNLL, LogNormalNLL, MixtureGaussianNLL
  → DistributionalHead의 분포별 params dict를 받아 NLL 계산
  → 단일 텐서 입력 시 MAE fallback
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .base import BaseLoss


class MAELoss(BaseLoss):
    """Mean Absolute Error."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if isinstance(pred, dict):
            pred = pred.get("mean", pred.get("mu", list(pred.values())[0]))
        return (pred - target).abs().mean()


class MSELoss(BaseLoss):
    """Mean Squared Error."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if isinstance(pred, dict):
            pred = pred.get("mean", pred.get("mu", list(pred.values())[0]))
        return ((pred - target) ** 2).mean()


class HuberLoss(BaseLoss):
    """Huber Loss — MAE/MSE 하이브리드."""

    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.huber = nn.HuberLoss(delta=delta)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if isinstance(pred, dict):
            pred = pred.get("mean", pred.get("mu", list(pred.values())[0]))
        return self.huber(pred, target)


class QuantileLoss(BaseLoss):
    """Quantile Loss — 비대칭 손실."""

    def __init__(self, quantile: float = 0.5):
        super().__init__()
        self.q = quantile

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if isinstance(pred, dict):
            pred = pred.get("mean", pred.get("mu", list(pred.values())[0]))
        error = target - pred
        return torch.max(self.q * error, (self.q - 1) * error).mean()


class AsymmetricLoss(BaseLoss):
    """비대칭 손실 — 과대/과소 예측에 다른 가중치."""

    def __init__(self, over_weight: float = 1.0, under_weight: float = 2.0):
        super().__init__()
        self.over_weight = over_weight
        self.under_weight = under_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if isinstance(pred, dict):
            pred = pred.get("mean", pred.get("mu", list(pred.values())[0]))
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
        if isinstance(pred, dict):
            mean = pred.get("mean", pred.get("mu"))
        else:
            mean = pred
        base_loss = self.base(mean, target)
        diffs = mean[:, 1:, :] - mean[:, :-1, :]
        smooth_loss = (diffs ** 2).mean()
        return base_loss + self.lambda_smooth * smooth_loss


# ══════════════════════════════════════════════════════════════
#  Distribution NLL Losses
# ══════════════════════════════════════════════════════════════

class GaussianNLLLoss(BaseLoss):
    """Gaussian NLL. pred: dict(mean, std) 또는 tuple(mean, std)."""

    def forward(self, pred, target: torch.Tensor) -> torch.Tensor:
        mean, std = _extract_mean_std(pred)
        if std is None:
            return (mean - target).abs().mean()
        variance = std ** 2
        nll = 0.5 * (torch.log(variance) + (target - mean) ** 2 / variance)
        return nll.mean()


class StudentTNLLLoss(BaseLoss):
    """Student-t NLL. Heavy tail 데이터에 적합.

    pred: dict(mean, std, df)
    NLL = -log_prob from torch.distributions.StudentT
    """

    def forward(self, pred, target: torch.Tensor) -> torch.Tensor:
        if isinstance(pred, dict) and "df" in pred:
            mean = pred["mean"]
            std = pred["std"]
            df = pred["df"]
            dist = torch.distributions.StudentT(df=df, loc=mean, scale=std)
            return -dist.log_prob(target).mean()
        # fallback
        mean, std = _extract_mean_std(pred)
        if std is None:
            return (mean - target).abs().mean()
        return GaussianNLLLoss().forward(pred, target)


class LogNormalNLLLoss(BaseLoss):
    """Log-Normal NLL. 양수 + right skew 데이터에 적합.

    pred: dict(mu, sigma)  — log-space parameters
    target must be > 0.
    NLL = -log_prob from torch.distributions.LogNormal
    """

    def forward(self, pred, target: torch.Tensor) -> torch.Tensor:
        if isinstance(pred, dict) and "mu" in pred:
            mu = pred["mu"]
            sigma = pred["sigma"]
            # target을 양수로 clamp (log(0) 방지)
            target_safe = target.clamp(min=1e-6)
            dist = torch.distributions.LogNormal(loc=mu, scale=sigma)
            return -dist.log_prob(target_safe).mean()
        # fallback
        mean, std = _extract_mean_std(pred)
        if std is None:
            return (mean - target).abs().mean()
        return GaussianNLLLoss().forward(pred, target)


class MixtureGaussianNLLLoss(BaseLoss):
    """Mixture of Gaussians NLL. Multi-modal 데이터에 적합.

    pred: dict(means, stds, weights)
      means: (B, H, output_dim, k)
      stds:  (B, H, output_dim, k)
      weights: (B, H, k) — softmax된 혼합 비율

    NLL = -log(sum_k weight_k * N(target | mean_k, std_k))
    """

    def forward(self, pred, target: torch.Tensor) -> torch.Tensor:
        if isinstance(pred, dict) and "means" in pred:
            means = pred["means"]     # (B, H, od, k)
            stds = pred["stds"]       # (B, H, od, k)
            weights = pred["weights"] # (B, H, k)

            # target: (B, H, od) → (B, H, od, 1)
            t = target.unsqueeze(-1)

            # per-component log prob: -0.5*(log(2pi*std^2) + (t-mean)^2/std^2)
            var = stds ** 2
            log_prob = -0.5 * (math.log(2 * math.pi) + torch.log(var) + (t - means) ** 2 / var)
            # sum over output_dim: (B, H, k)
            log_prob = log_prob.sum(dim=2)

            # log-sum-exp with weights: log(sum_k w_k * exp(log_prob_k))
            log_weights = torch.log(weights + 1e-8)
            log_mixture = torch.logsumexp(log_weights + log_prob, dim=-1)  # (B, H)

            return -log_mixture.mean()
        # fallback
        mean, std = _extract_mean_std(pred)
        if std is None:
            return (mean - target).abs().mean()
        return GaussianNLLLoss().forward(pred, target)


# ══════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════

def _extract_mean_std(pred) -> tuple:
    """pred에서 mean, std를 추출. dict/tuple/tensor 모두 대응."""
    if isinstance(pred, dict):
        mean = pred.get("mean", pred.get("mu"))
        std = pred.get("std", pred.get("sigma"))
        return mean, std
    if isinstance(pred, tuple) and len(pred) == 2:
        return pred[0], pred[1]
    return pred, None


LOSS_REGISTRY: dict[str, type[BaseLoss]] = {
    "MAE": MAELoss,
    "MSE": MSELoss,
    "Huber": HuberLoss,
    "Quantile": QuantileLoss,
    "Asymmetric": AsymmetricLoss,
    "SmoothnessReg": SmoothnessRegLoss,
    "GaussianNLL": GaussianNLLLoss,
    "StudentTNLL": StudentTNLLLoss,
    "LogNormalNLL": LogNormalNLLLoss,
    "MixtureGaussianNLL": MixtureGaussianNLLLoss,
}
