"""CompositeScore — 다차원 모델 평가.

단일 val MAE 대신 5개 지표의 가중합으로 모델을 평가.
Critic 판정과 Greedy Ensemble 선택에 사용.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CompositeScore:
    """다차원 모델 평가 점수.

    각 지표는 0~1로 정규화 (높을수록 좋음).
    """
    mae_normalized: float = 0.0       # 1 - mae/baseline_mae (weight 0.4)
    horizon_uniformity: float = 1.0   # 1 - (max-min)/mean step MAE (weight 0.2)
    residual_cleanliness: float = 1.0 # 잔차 품질 (weight 0.2)
    distribution_fit: float = 1.0     # KS test p-value (weight 0.1)
    stability: float = 1.0            # 라운드 간 일관성 (weight 0.1)

    WEIGHTS = (0.4, 0.2, 0.2, 0.1, 0.1)

    def weighted_score(self) -> float:
        """가중 종합 점수. 0~1, 높을수록 좋음."""
        w = self.WEIGHTS
        return (w[0] * self.mae_normalized +
                w[1] * self.horizon_uniformity +
                w[2] * self.residual_cleanliness +
                w[3] * self.distribution_fit +
                w[4] * self.stability)

    def summary(self) -> str:
        return (f"score={self.weighted_score():.3f} "
                f"(mae={self.mae_normalized:.2f}, "
                f"horizon={self.horizon_uniformity:.2f}, "
                f"residual={self.residual_cleanliness:.2f}, "
                f"dist={self.distribution_fit:.2f}, "
                f"stab={self.stability:.2f})")

    def to_dict(self) -> dict:
        return {
            "weighted_score": round(self.weighted_score(), 4),
            "mae_normalized": round(self.mae_normalized, 4),
            "horizon_uniformity": round(self.horizon_uniformity, 4),
            "residual_cleanliness": round(self.residual_cleanliness, 4),
            "distribution_fit": round(self.distribution_fit, 4),
            "stability": round(self.stability, 4),
        }


def compute_composite_score(
    val_mae: float,
    baseline_mae: float,
    val_mae_by_step: list[float] | None = None,
    residual_diagnosis=None,
    dist_fit_pvalue: float | None = None,
    disagreement_stability: float = 1.0,
) -> CompositeScore:
    """학습 결과에서 CompositeScore를 계산.

    Args:
        val_mae: 현재 모델의 val MAE
        baseline_mae: naive baseline MAE (last-value repeat)
        val_mae_by_step: 각 prediction step의 MAE
        residual_diagnosis: ResidualDiagnosis 객체
        dist_fit_pvalue: 분포 적합도 KS test p-value
        disagreement_stability: 이전 라운드 대비 일관성 (0~1)
    """
    # 1. MAE normalized (0~1, 높을수록 좋음)
    if baseline_mae > 0:
        mae_norm = max(0.0, min(1.0, 1.0 - val_mae / baseline_mae))
    else:
        mae_norm = 0.5

    # 2. Horizon uniformity
    if val_mae_by_step and len(val_mae_by_step) >= 4:
        arr = np.array(val_mae_by_step)
        mean_step = arr.mean()
        if mean_step > 0:
            uniformity = max(0.0, 1.0 - (arr.max() - arr.min()) / mean_step)
        else:
            uniformity = 1.0
    else:
        uniformity = 1.0  # 정보 없으면 중립

    # 3. Residual cleanliness
    if residual_diagnosis is not None:
        n_issues = 0
        if residual_diagnosis.has_autocorrelation:
            n_issues += 1
        if residual_diagnosis.has_heteroscedasticity:
            n_issues += 1
        if abs(residual_diagnosis.residual_skew) > 1.0:
            n_issues += 1
        cleanliness = {0: 1.0, 1: 0.7, 2: 0.4, 3: 0.2}.get(n_issues, 0.1)
    else:
        cleanliness = 1.0

    # 4. Distribution fit
    if dist_fit_pvalue is not None:
        dist_fit = min(1.0, dist_fit_pvalue)
    else:
        dist_fit = 1.0

    # 5. Stability
    stability = max(0.0, min(1.0, disagreement_stability))

    return CompositeScore(
        mae_normalized=mae_norm,
        horizon_uniformity=uniformity,
        residual_cleanliness=cleanliness,
        distribution_fit=dist_fit,
        stability=stability,
    )
