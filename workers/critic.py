"""Critic — rule-based 결과 분석 (v2.1). 적응형 threshold + 피드백 분기.

v2.1 변경사항:
- 하드코딩 threshold → norm_MSE 기반 적응형
- 데이터 스케일에 무관하게 동작
- extreme 구간 분석 세분화
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass


@dataclass
class CriticVerdict:
    """Critic의 구조화된 판단 결과 (v2.1)."""
    verdict: str  # DONE | RETRY_HP | RETRY_RECIPE | RETRY_BLOCK
    best_model: str
    best_metric: dict
    normal_metric: dict
    extreme_metric: dict
    analysis: str
    suggestions: list[str]
    ceiling_reached: bool
    iteration: int

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "best_model": self.best_model,
            "best_metric": self.best_metric,
            "normal_metric": self.normal_metric,
            "extreme_metric": self.extreme_metric,
            "analysis": self.analysis,
            "suggestions": self.suggestions,
            "ceiling_reached": self.ceiling_reached,
            "iteration": self.iteration,
        }


class Critic:
    """Rule-based Critic (v2.1) — 적응형 threshold.

    판단 분기:
    - DONE: 최종 결과 확정
    - RETRY_HP: HP 미세 조정 (Architect Step 2~6 재실행)
    - RETRY_RECIPE: 다른 레시피 시도 (Architect Step 1 재실행)
    - RETRY_BLOCK: 블록 교체 (KG에 대체 블록 쿼리 → Architect 선택)

    적응형 threshold:
    - norm_MSE 기반으로 개선폭/극단비율 판단 (스케일 무관)
    - norm_MSE < 0.1: 이미 우수 → ceiling 판단 관대
    - norm_MSE > 0.5: 개선 여지 큼 → retry 적극적
    """
    name = "critic"
    description = "Rule-based 결과 분석 + 적응형 피드백 분기 (v2.1)"

    # 기본값 (norm_MSE 없을 때 fallback)
    DEFAULT_IMPROVEMENT_THRESHOLD = 0.02
    DEFAULT_EXTREME_RATIO = 2.0
    DEFAULT_VAL_TEST_RATIO = 2.0

    # norm_MSE 구간별 적응형 threshold
    ADAPTIVE_THRESHOLDS = {
        # norm_MSE < 0.1: 이미 좋음 → 작은 개선도 의미 있고, ceiling 빨리 인정
        "excellent": {
            "norm_mse_upper": 0.1,
            "improvement_threshold": 0.01,   # 1% 개선이면 충분
            "extreme_ratio_threshold": 2.5,  # 극단 비율 관대
            "val_test_ratio": 2.5,
        },
        # 0.1 <= norm_MSE < 0.3: 보통 → 표준 기준
        "good": {
            "norm_mse_upper": 0.3,
            "improvement_threshold": 0.02,
            "extreme_ratio_threshold": 2.0,
            "val_test_ratio": 2.0,
        },
        # 0.3 <= norm_MSE < 0.5: 미흡 → retry 적극적
        "fair": {
            "norm_mse_upper": 0.5,
            "improvement_threshold": 0.03,
            "extreme_ratio_threshold": 1.8,
            "val_test_ratio": 1.8,
        },
        # norm_MSE >= 0.5: 나쁨 → 강하게 retry
        "poor": {
            "norm_mse_upper": float("inf"),
            "improvement_threshold": 0.05,
            "extreme_ratio_threshold": 1.5,
            "val_test_ratio": 1.5,
        },
    }

    def __init__(self, cwd: str = "", rules: str = "",
                 prev_mae: float | None = None,
                 prev_norm_mse: float | None = None):
        self.cwd = cwd
        self.prev_mae = prev_mae
        self.prev_norm_mse = prev_norm_mse

    def run(self, task: str) -> dict:
        """학습 결과를 파싱하여 rule-based 판단."""
        metrics = self._extract_metrics(task)
        iteration = self._extract_iteration(task)
        max_iterations = self._extract_max_iterations(task)

        verdict = self._judge(metrics, iteration, max_iterations)

        response = json.dumps(verdict.to_dict(), ensure_ascii=False, indent=2)

        return {
            "worker": self.name,
            "response": response,
            "code": None,
            "execution_result": None,
        }

    def _get_thresholds(self, norm_mse: float | None) -> dict:
        """norm_MSE 기반 적응형 threshold 결정."""
        if norm_mse is None:
            return {
                "improvement_threshold": self.DEFAULT_IMPROVEMENT_THRESHOLD,
                "extreme_ratio_threshold": self.DEFAULT_EXTREME_RATIO,
                "val_test_ratio": self.DEFAULT_VAL_TEST_RATIO,
                "level": "unknown",
            }
        for level, cfg in self.ADAPTIVE_THRESHOLDS.items():
            if norm_mse < cfg["norm_mse_upper"]:
                return {**cfg, "level": level}
        # fallback
        return {**list(self.ADAPTIVE_THRESHOLDS.values())[-1], "level": "poor"}

    def _judge(self, metrics: dict, iteration: int, max_iterations: int) -> CriticVerdict:
        """v2.1 Critic Decision Tree — 적응형 threshold."""
        overall = metrics.get("overall", {})
        normal = metrics.get("normal", {})
        extreme = metrics.get("extreme", {})
        best_model = metrics.get("best_model", "unknown")
        naive_metric = metrics.get("naive", {})

        mae = overall.get("MAE")
        norm_mse = metrics.get("norm", {}).get("MSE")
        best_epoch = metrics.get("best_epoch")
        suggestions: list[str] = []
        analysis_parts: list[str] = []
        ceiling_reached = False

        # 적응형 threshold 결정
        thresholds = self._get_thresholds(norm_mse)
        improvement_threshold = thresholds["improvement_threshold"]
        extreme_ratio_threshold = thresholds["extreme_ratio_threshold"]
        val_test_ratio_threshold = thresholds["val_test_ratio"]
        perf_level = thresholds["level"]

        if norm_mse is not None:
            analysis_parts.append(f"norm_MSE={norm_mse:.4f} (level={perf_level})")

        # ── Check 1: naive보다 나쁨 → ERROR ──
        naive_mae = naive_metric.get("MAE")
        if mae is not None and naive_mae is not None and mae > naive_mae:
            analysis_parts.append(
                f"model MAE={mae:.4f} > naive MAE={naive_mae:.4f}. naive보다 나쁨."
            )
            suggestions.append("데이터/파이프라인 점검 필요. leakage 또는 설정 오류 의심.")
            return CriticVerdict(
                verdict="RETRY_RECIPE", best_model=best_model,
                best_metric=overall, normal_metric=normal, extreme_metric=extreme,
                analysis=". ".join(analysis_parts), suggestions=suggestions,
                ceiling_reached=False, iteration=iteration,
            )

        # ── Check 2: 메트릭 없음 ──
        if not overall or mae is None:
            analysis_parts.append("메트릭 추출 실패")
            suggestions.append("데이터 경로와 컬럼명 확인")
            return CriticVerdict(
                verdict="RETRY_HP", best_model=best_model,
                best_metric=overall, normal_metric=normal, extreme_metric=extreme,
                analysis=". ".join(analysis_parts), suggestions=suggestions,
                ceiling_reached=False, iteration=iteration,
            )

        # ── Check 3: best_epoch=0 → 학습 미시작 ──
        if best_epoch is not None and best_epoch == 0:
            analysis_parts.append("best_epoch=0. 학습 미시작.")
            suggestions.append("lr 낮추기 또는 warmup 추가 (capacity=high이면 warmup_epochs=10 권장)")
            return CriticVerdict(
                verdict="RETRY_HP", best_model=best_model,
                best_metric=overall, normal_metric=normal, extreme_metric=extreme,
                analysis=". ".join(analysis_parts), suggestions=suggestions,
                ceiling_reached=False, iteration=iteration,
            )

        # ── Check 4: val-test 갭 과대 (적응형) ──
        val_mae = metrics.get("val", {}).get("MAE")
        if val_mae and mae and val_mae > 0:
            val_test_ratio = mae / val_mae
            if val_test_ratio > val_test_ratio_threshold:
                analysis_parts.append(
                    f"val-test 갭 과대: test/val={val_test_ratio:.2f} "
                    f"(threshold={val_test_ratio_threshold:.1f} at {perf_level})"
                )
                if val_test_ratio > 3.0:
                    suggestions.append("심각한 분포 차이. 더 단순한 모델 + RevIN + weight_decay 강화")
                else:
                    suggestions.append("더 단순한 모델 또는 RevIN 권장")
                return CriticVerdict(
                    verdict="RETRY_HP", best_model=best_model,
                    best_metric=overall, normal_metric=normal, extreme_metric=extreme,
                    analysis=". ".join(analysis_parts), suggestions=suggestions,
                    ceiling_reached=False, iteration=iteration,
                )

        # ── Check 5: 마지막 iteration ──
        if iteration >= max_iterations:
            analysis_parts.append(f"최대 반복({max_iterations}) 도달")
            return CriticVerdict(
                verdict="DONE", best_model=best_model,
                best_metric=overall, normal_metric=normal, extreme_metric=extreme,
                analysis=". ".join(analysis_parts), suggestions=suggestions,
                ceiling_reached=True, iteration=iteration,
            )

        analysis_parts.append(f"전체 MAE: {mae:.4f}")

        # ── Check 6: Extreme 구간 성능 (적응형) ──
        extreme_ratio = 1.0
        if normal and extreme:
            normal_mae = normal.get("MAE", mae)
            extreme_mae = extreme.get("MAE", mae)
            if normal_mae > 0:
                extreme_ratio = extreme_mae / normal_mae
            analysis_parts.append(
                f"Normal MAE: {normal_mae:.4f}, Extreme MAE: {extreme_mae:.4f} "
                f"(비율: {extreme_ratio:.2f}, threshold={extreme_ratio_threshold:.1f})"
            )

            if extreme_ratio > extreme_ratio_threshold * 1.5 and iteration < max_iterations:
                # 극단 비율이 threshold의 1.5배 초과 → RETRY_BLOCK (capacity 상향)
                suggestions.append(
                    f"극단 구간 성능 심각 (비율={extreme_ratio:.1f}). "
                    "TemporalMixer capacity 상향 또는 AsymmetricLoss 시도."
                )
                return CriticVerdict(
                    verdict="RETRY_BLOCK", best_model=best_model,
                    best_metric=overall, normal_metric=normal, extreme_metric=extreme,
                    analysis=". ".join(analysis_parts), suggestions=suggestions,
                    ceiling_reached=False, iteration=iteration,
                )
            elif extreme_ratio > extreme_ratio_threshold and iteration < max_iterations:
                # threshold 초과하지만 심각하지는 않음 → RETRY_HP (loss 조정)
                suggestions.append(
                    f"극단 구간 개선 필요 (비율={extreme_ratio:.1f}). "
                    "AsymmetricLoss under_weight 상향 또는 Huber loss 시도."
                )

        # ── Check 7: 이전 대비 개선 (적응형) ──
        if self.prev_mae is not None:
            improvement = (self.prev_mae - mae) / self.prev_mae
            analysis_parts.append(
                f"이전 대비 개선: {improvement:.2%} "
                f"(threshold={improvement_threshold:.1%} at {perf_level})"
            )

            if improvement < improvement_threshold:
                analysis_parts.append("개선폭 미미 — ceiling")
                return CriticVerdict(
                    verdict="DONE", best_model=best_model,
                    best_metric=overall, normal_metric=normal, extreme_metric=extreme,
                    analysis=". ".join(analysis_parts), suggestions=suggestions,
                    ceiling_reached=True, iteration=iteration,
                )

        # norm_MSE 기반 추가 개선 가이드
        if self.prev_norm_mse is not None and norm_mse is not None:
            norm_improvement = (self.prev_norm_mse - norm_mse) / self.prev_norm_mse
            analysis_parts.append(f"norm_MSE 개선: {norm_improvement:.2%}")

        # ── Check 8: 첫 iteration → baseline 확보 → 다른 레시피 ──
        if iteration == 1:
            analysis_parts.append("baseline 확보. 다른 레시피 시도.")
            if norm_mse and norm_mse > 0.5:
                suggestions.append("norm_MSE > 0.5: capacity 높은 레시피 + scheduler=cosine_warmup 권장")
            elif norm_mse and norm_mse > 0.3:
                suggestions.append("norm_MSE > 0.3: capacity 중간 이상 레시피 시도")
            else:
                suggestions.append("baseline 양호. 다른 레시피로 비교")
            suggestions.append("다음 레시피를 시도하여 비교")
            return CriticVerdict(
                verdict="RETRY_RECIPE", best_model=best_model,
                best_metric=overall, normal_metric=normal, extreme_metric=extreme,
                analysis=". ".join(analysis_parts), suggestions=suggestions,
                ceiling_reached=False, iteration=iteration,
            )

        # ── Default: HP 미세조정 ──
        if norm_mse and norm_mse > 0.3:
            suggestions.append("norm_MSE 아직 높음 — scheduler/warmup 조정 또는 d_model 증가 시도")
        suggestions.append("HP 미세 조정 시도")
        return CriticVerdict(
            verdict="RETRY_HP", best_model=best_model,
            best_metric=overall, normal_metric=normal, extreme_metric=extreme,
            analysis=". ".join(analysis_parts), suggestions=suggestions,
            ceiling_reached=False, iteration=iteration,
        )

    def _extract_metrics(self, text: str) -> dict:
        metrics: dict = {}

        metrics_match = re.search(r'METRICS:\s*(\{[^}]+\})', text)
        if metrics_match:
            try:
                metrics["overall"] = json.loads(metrics_match.group(1))
            except json.JSONDecodeError:
                pass

        for metric_name in ["MAE", "MSE", "RMSE", "MAPE"]:
            pattern = rf'{metric_name}[:\s=]+([0-9]+\.?[0-9]*)'
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                metrics.setdefault("overall", {})[metric_name] = float(match.group(1))

        model_match = re.search(r'BEST_MODEL[:\s=]+(\S+)', text)
        if model_match:
            metrics["best_model"] = model_match.group(1)

        norm_match = re.search(r'METRICS_NORM:\s*(\{[^}]+\})', text)
        if norm_match:
            try:
                metrics["norm"] = json.loads(norm_match.group(1))
            except json.JSONDecodeError:
                pass

        normal_match = re.search(r'NORMAL[_ ]MAE[:\s=]+([0-9]+\.?[0-9]*)', text, re.IGNORECASE)
        extreme_match = re.search(r'EXTREME[_ ]MAE[:\s=]+([0-9]+\.?[0-9]*)', text, re.IGNORECASE)
        if normal_match:
            metrics["normal"] = {"MAE": float(normal_match.group(1))}
        if extreme_match:
            metrics["extreme"] = {"MAE": float(extreme_match.group(1))}

        # naive baseline
        naive_match = re.search(r'NAIVE[_ ]MAE[:\s=]+([0-9]+\.?[0-9]*)', text, re.IGNORECASE)
        if naive_match:
            metrics["naive"] = {"MAE": float(naive_match.group(1))}

        # best_epoch
        epoch_match = re.search(r'best.epoch[:\s=]+(\d+)', text, re.IGNORECASE)
        if epoch_match:
            metrics["best_epoch"] = int(epoch_match.group(1))

        return metrics

    def _extract_iteration(self, text: str) -> int:
        match = re.search(r'Iteration[:\s]*(\d+)', text, re.IGNORECASE)
        return int(match.group(1)) if match else 1

    def _extract_max_iterations(self, text: str) -> int:
        match = re.search(r'Iteration[:\s]*\d+/(\d+)', text, re.IGNORECASE)
        return int(match.group(1)) if match else 3
