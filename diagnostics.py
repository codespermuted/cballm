"""CBALLM 진단 모듈 — 4종 rule-based 진단기.

CurveAnalyzer:        학습 곡선 패턴 진단 (loss history)
HorizonDiagnoser:     예측 horizon별 성능 분석 (step MAE)
FeatureDiagnoser:     feature 유효성 진단 (permutation importance)
DisagreementDiagnoser: 라운드 간 예측 일관성 (predictions 비교)

LLM 불사용. 모든 판정은 통계량과 규칙으로 수행.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ══════════════════════════════════════════════════════════════
#  CurveAnalyzer — 학습 곡선 패턴 진단
# ══════════════════════════════════════════════════════════════

@dataclass
class CurveDiagnosis:
    """학습 곡선 진단 결과."""
    pattern: str = "healthy"
    # healthy, overfitting, lr_instability, early_plateau, still_improving
    train_val_gap: float = 0.0       # 최종 train/val loss 비율
    convergence_speed: str = "normal"  # fast, normal, slow, not_converged
    best_epoch_ratio: float = 0.0    # best_epoch / max_epochs
    loss_volatility: float = 0.0     # val_loss rolling(5) std / mean
    suggested_action: str = "maintain"

    def summary(self, max_len: int = 80) -> str:
        speed_map = {"fast": "빠름", "normal": "보통", "slow": "느림", "not_converged": "미수렴"}
        s = f"학습곡선: {self.pattern}, 수렴={speed_map.get(self.convergence_speed, self.convergence_speed)}"
        if self.pattern != "healthy":
            s += f" → {self.suggested_action}"
        return s[:max_len]


class CurveAnalyzer:
    """학습 곡선 패턴을 rule-based로 진단."""

    @staticmethod
    def analyze(train_losses: list[float], val_losses: list[float],
                best_epoch: int, max_epochs: int) -> CurveDiagnosis:
        if not train_losses or not val_losses:
            return CurveDiagnosis()

        n = len(val_losses)
        best_epoch_ratio = best_epoch / max(max_epochs, 1)

        # train/val gap
        train_final = train_losses[-1] if train_losses else 0
        val_final = val_losses[-1] if val_losses else 1
        train_val_gap = val_final / max(train_final, 1e-8)

        # val loss volatility: rolling(5) std / mean
        if n >= 5:
            window = min(5, n)
            recent = np.array(val_losses[-window:])
            loss_volatility = float(recent.std() / max(recent.mean(), 1e-8))
        else:
            loss_volatility = 0.0

        # convergence speed
        if best_epoch_ratio < 0.2:
            convergence_speed = "fast"
        elif best_epoch_ratio < 0.7:
            convergence_speed = "normal"
        elif best_epoch == max_epochs - 1 or best_epoch == n - 1:
            convergence_speed = "not_converged"
        else:
            convergence_speed = "slow"

        # pattern 판정
        best_val = val_losses[best_epoch] if best_epoch < len(val_losses) else val_final

        # overfitting: val이 best 이후 10% 이상 악화 AND train 계속 하락
        if n > best_epoch + 3:
            post_best_val = np.mean(val_losses[best_epoch + 1:])
            train_still_dropping = len(train_losses) > 3 and train_losses[-1] < train_losses[-3]
            if post_best_val > best_val * 1.1 and train_still_dropping:
                return CurveDiagnosis(
                    pattern="overfitting", train_val_gap=train_val_gap,
                    convergence_speed=convergence_speed, best_epoch_ratio=best_epoch_ratio,
                    loss_volatility=loss_volatility,
                    suggested_action="weight_decay 강화, dropout 추가",
                )

        # lr_instability: val_loss diff 부호 변경 > 50%
        # 짧은 history(< 20 epoch)에서는 noise에 민감하므로 비활성화
        if n >= 20:
            diffs = np.diff(val_losses)
            sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
            if sign_changes / max(len(diffs) - 1, 1) > 0.45:
                return CurveDiagnosis(
                    pattern="lr_instability", train_val_gap=train_val_gap,
                    convergence_speed=convergence_speed, best_epoch_ratio=best_epoch_ratio,
                    loss_volatility=loss_volatility,
                    suggested_action="lr 낮추기, warmup epoch 추가",
                )

        # early_plateau: best_epoch < 30% AND 이후 변화 < 1%
        if best_epoch_ratio < 0.3 and n > best_epoch + 5:
            post_range = val_losses[best_epoch:]
            change = abs(max(post_range) - min(post_range)) / max(best_val, 1e-8)
            if change < 0.01:
                return CurveDiagnosis(
                    pattern="early_plateau", train_val_gap=train_val_gap,
                    convergence_speed=convergence_speed, best_epoch_ratio=best_epoch_ratio,
                    loss_volatility=loss_volatility,
                    suggested_action="scheduler 변경 (cosine → plateau)",
                )

        # still_improving: 마지막 3 epoch 평균 하락 중
        if n >= 3:
            last3 = val_losses[-3:]
            if last3[-1] < last3[0] and (best_epoch >= n - 3 or convergence_speed == "not_converged"):
                return CurveDiagnosis(
                    pattern="still_improving", train_val_gap=train_val_gap,
                    convergence_speed=convergence_speed, best_epoch_ratio=best_epoch_ratio,
                    loss_volatility=loss_volatility,
                    suggested_action="max_epochs 증가",
                )

        # healthy
        return CurveDiagnosis(
            pattern="healthy", train_val_gap=train_val_gap,
            convergence_speed=convergence_speed, best_epoch_ratio=best_epoch_ratio,
            loss_volatility=loss_volatility,
            suggested_action="maintain",
        )


# ══════════════════════════════════════════════════════════════
#  HorizonDiagnoser — 예측 horizon별 성능 분석
# ══════════════════════════════════════════════════════════════

@dataclass
class HorizonDiagnosis:
    """Horizon별 성능 진단 결과."""
    mae_by_step: list[float] = field(default_factory=list)
    degradation_rate: float = 1.0   # mae_last_quarter / mae_first_quarter
    collapse_point: int | None = None  # 연속 3 step MAE > 평균 2배인 첫 지점
    pattern: str = "uniform"  # uniform, gradual, cliff, oscillating

    def summary(self, max_len: int = 80) -> str:
        if self.pattern == "uniform":
            return f"Horizon: 균일 (degradation={self.degradation_rate:.1f}x)"[:max_len]
        if self.pattern == "cliff":
            return f"Horizon: step {self.collapse_point} 이후 급락 → attention window 확장"[:max_len]
        if self.pattern == "gradual":
            return f"Horizon: 점진 악화 ({self.degradation_rate:.1f}x) → seq_len 확장"[:max_len]
        return f"Horizon: 진동 패턴 → ConvMix 시도"[:max_len]


class HorizonDiagnoser:
    """Horizon별 step MAE를 분석."""

    @staticmethod
    def analyze(mae_by_step: list[float]) -> HorizonDiagnosis:
        if not mae_by_step or len(mae_by_step) < 4:
            return HorizonDiagnosis(mae_by_step=mae_by_step)

        arr = np.array(mae_by_step)
        n = len(arr)
        q = max(1, n // 4)

        first_q = arr[:q].mean()
        last_q = arr[-q:].mean()
        degradation = last_q / max(first_q, 1e-8)

        # collapse point: 연속 3 step MAE > first_quarter 평균의 3배
        # 3배 기준으로 해야 gradual increase와 구분 가능
        collapse = None
        for i in range(n - 2):
            if all(arr[i:i+3] > first_q * 3):
                collapse = i
                break

        # 진동 판정: 인접 step 간 부호 교대 비율
        step_diffs = np.diff(arr)
        if len(step_diffs) > 2:
            sign_alternations = np.sum(np.diff(np.sign(step_diffs)) != 0)
            alternation_ratio = sign_alternations / max(len(step_diffs) - 1, 1)
        else:
            alternation_ratio = 0.0

        # pattern 판정
        if collapse is not None:
            pattern = "cliff"
        elif alternation_ratio > 0.5:
            pattern = "oscillating"
        elif degradation < 1.3:
            pattern = "uniform"
        else:
            pattern = "gradual"

        return HorizonDiagnosis(
            mae_by_step=mae_by_step,
            degradation_rate=round(degradation, 2),
            collapse_point=collapse,
            pattern=pattern,
        )


# ══════════════════════════════════════════════════════════════
#  FeatureDiagnoser — feature 유효성 진단
# ══════════════════════════════════════════════════════════════

@dataclass
class FeatureDiagnosis:
    """Feature 유효성 진단 결과."""
    useful: list[str] = field(default_factory=list)      # importance 상위
    useless: list[str] = field(default_factory=list)     # 제거해도 변화 < 1%
    suggested_add: list[str] = field(default_factory=list)
    suggested_drop: list[str] = field(default_factory=list)

    def summary(self, max_len: int = 80) -> str:
        parts = []
        if self.useful:
            parts.append(f"{','.join(self.useful[:3])} 유효")
        if self.useless:
            parts.append(f"{','.join(self.useless[:2])} 무효")
        actions = []
        if self.suggested_drop:
            actions.append(f"drop {','.join(self.suggested_drop[:2])}")
        if self.suggested_add:
            actions.append(f"add {','.join(self.suggested_add[:2])}")
        if actions:
            parts.append(" → " + ", ".join(actions))
        return ("Feature: " + " / ".join(parts))[:max_len] if parts else ""


class FeatureDiagnoser:
    """Permutation importance 기반 feature 유효성 분석.

    val set에서만 계산. feature 20개 이상이면 top-k sampling.
    """

    @staticmethod
    def analyze(model, val_loader, loss_fn, feature_names: list[str],
                device: str = "cpu", top_k: int = 20) -> FeatureDiagnosis:
        """model + val_loader로 permutation importance 계산."""
        import torch

        if not feature_names:
            return FeatureDiagnosis()

        n_features = len(feature_names)
        # feature 수 제한
        if n_features > top_k:
            indices = list(range(min(top_k, n_features)))
        else:
            indices = list(range(n_features))

        # baseline MAE
        model.eval()
        baseline_mae = FeatureDiagnoser._compute_mae(model, val_loader, device)

        # 각 feature를 shuffle하여 MAE 변화 측정
        importance: dict[str, float] = {}
        for idx in indices:
            shuffled_mae = FeatureDiagnoser._compute_mae_shuffled(
                model, val_loader, device, shuffle_idx=idx,
            )
            delta = (shuffled_mae - baseline_mae) / max(baseline_mae, 1e-8)
            importance[feature_names[idx]] = delta

        # 분류
        sorted_feats = sorted(importance.items(), key=lambda x: -x[1])
        useful = [f for f, d in sorted_feats[:3] if d > 0.01]
        useless = [f for f, d in sorted_feats if d < 0.01]

        return FeatureDiagnosis(
            useful=useful,
            useless=useless,
            suggested_drop=useless[:3],
        )

    @staticmethod
    def analyze_from_importance(importance: dict[str, float],
                                 available_templates: list[str] | None = None,
                                 has_heteroscedasticity: bool = False) -> FeatureDiagnosis:
        """사전 계산된 importance dict로 분석 (model 불필요)."""
        sorted_feats = sorted(importance.items(), key=lambda x: -x[1])
        useful = [f for f, d in sorted_feats[:3] if d > 0.01]
        useless = [f for f, d in sorted_feats if d < 0.01]

        suggested_add = []
        if available_templates:
            if has_heteroscedasticity and "volatility" in available_templates:
                suggested_add.append("volatility")
            for t in available_templates:
                if t not in [f.split("_")[0] for f in useful + useless]:
                    suggested_add.append(t)
                if len(suggested_add) >= 2:
                    break

        return FeatureDiagnosis(
            useful=useful,
            useless=useless,
            suggested_add=suggested_add[:2],
            suggested_drop=useless[:3],
        )

    @staticmethod
    def _compute_mae(model, val_loader, device: str) -> float:
        import torch
        preds, targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                if isinstance(pred, dict):
                    pred = pred.get("mean", pred.get("mu"))
                preds.append(pred.cpu())
                targets.append(yb.cpu())
        preds = torch.cat(preds)
        targets = torch.cat(targets)
        return (preds - targets).abs().mean().item()

    @staticmethod
    def _compute_mae_shuffled(model, val_loader, device: str,
                               shuffle_idx: int) -> float:
        import torch
        preds, targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.clone()
                # feature shuffle: 해당 컬럼을 배치 내에서 shuffle
                perm = torch.randperm(xb.shape[0])
                xb[:, :, shuffle_idx] = xb[perm, :, shuffle_idx]
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                if isinstance(pred, dict):
                    pred = pred.get("mean", pred.get("mu"))
                preds.append(pred.cpu())
                targets.append(yb.cpu())
        preds = torch.cat(preds)
        targets = torch.cat(targets)
        return (preds - targets).abs().mean().item()


# ══════════════════════════════════════════════════════════════
#  DisagreementDiagnoser — 라운드 간 예측 일관성
# ══════════════════════════════════════════════════════════════

@dataclass
class DisagreementDiagnosis:
    """라운드 간 예측 일관성 진단."""
    high_disagreement_segments: list[str] = field(default_factory=list)
    consensus_segments: list[str] = field(default_factory=list)
    stability: float = 1.0  # 0~1

    def summary(self, max_len: int = 80) -> str:
        if self.stability > 0.8:
            return f"일관성: 안정 (stability={self.stability:.2f})"[:max_len]
        segs = ",".join(self.high_disagreement_segments[:2])
        return f"일관성: {segs} 예측 불안정 (stability={self.stability:.2f})"[:max_len]


class DisagreementDiagnoser:
    """라운드 간 val predictions를 비교하여 일관성 진단."""

    SEGMENT_LABELS = ["초반", "중반", "후반", "말미"]

    @staticmethod
    def analyze(prev_predictions: np.ndarray | None,
                curr_predictions: np.ndarray | None) -> DisagreementDiagnosis:
        """두 라운드의 val predictions를 비교."""
        if prev_predictions is None or curr_predictions is None:
            return DisagreementDiagnosis()

        # shape 맞추기 (작은 쪽에 맞춤)
        n = min(len(prev_predictions), len(curr_predictions))
        if n == 0:
            return DisagreementDiagnosis()

        prev = prev_predictions[:n]
        curr = curr_predictions[:n]

        # 전체 MAE 차이
        overall_diff = np.abs(prev - curr).mean()
        overall_scale = np.abs(curr).mean() + 1e-8

        # 구간별 비교
        n_seg = min(4, n)
        seg_size = max(1, n // n_seg)
        labels = DisagreementDiagnoser.SEGMENT_LABELS

        high_disagreement = []
        consensus = []

        for i in range(n_seg):
            seg_prev = prev[i * seg_size:(i + 1) * seg_size]
            seg_curr = curr[i * seg_size:(i + 1) * seg_size]
            seg_diff = np.abs(seg_prev - seg_curr).mean()
            seg_ratio = seg_diff / overall_scale

            label = labels[i] if i < len(labels) else f"seg{i}"
            if seg_ratio > 0.1:  # 10% 이상 차이
                high_disagreement.append(label)
            else:
                consensus.append(label)

        n_disagree = len(high_disagreement)
        stability = 1.0 - (n_disagree / max(n_seg, 1))

        return DisagreementDiagnosis(
            high_disagreement_segments=high_disagreement,
            consensus_segments=consensus,
            stability=round(stability, 2),
        )
