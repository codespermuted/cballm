"""Critic — rule-based 결과 분석 (v2). 피드백 분기: DONE/RETRY_HP/RETRY_RECIPE/RETRY_BLOCK."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass


@dataclass
class CriticVerdict:
    """Critic의 구조화된 판단 결과 (v2)."""
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
    """Rule-based Critic (v2).

    판단 분기:
    - DONE: 최종 결과 확정
    - RETRY_HP: HP 미세 조정 (Architect Step 2~6 재실행)
    - RETRY_RECIPE: 다른 레시피 시도 (Architect Step 1 재실행)
    - RETRY_BLOCK: 블록 교체 (KG에 대체 블록 쿼리 → Architect 선택)
    """
    name = "critic"
    description = "Rule-based 결과 분석 + 피드백 분기 (v2)"

    MAE_IMPROVEMENT_THRESHOLD = 0.02
    EXTREME_MAE_RATIO = 2.0
    CEILING_VARIANCE = 0.005
    VAL_TEST_RATIO_THRESHOLD = 2.0

    def __init__(self, cwd: str = "", rules: str = "",
                 prev_mae: float | None = None):
        self.cwd = cwd
        self.prev_mae = prev_mae

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

    def _judge(self, metrics: dict, iteration: int, max_iterations: int) -> CriticVerdict:
        """v2 Critic Decision Tree."""
        overall = metrics.get("overall", {})
        normal = metrics.get("normal", {})
        extreme = metrics.get("extreme", {})
        best_model = metrics.get("best_model", "unknown")
        naive_metric = metrics.get("naive", {})

        mae = overall.get("MAE")
        best_epoch = metrics.get("best_epoch")
        suggestions: list[str] = []
        analysis_parts: list[str] = []
        ceiling_reached = False

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
            suggestions.append("lr 낮추기 또는 warmup 추가")
            return CriticVerdict(
                verdict="RETRY_HP", best_model=best_model,
                best_metric=overall, normal_metric=normal, extreme_metric=extreme,
                analysis=". ".join(analysis_parts), suggestions=suggestions,
                ceiling_reached=False, iteration=iteration,
            )

        # ── Check 4: val-test 갭 과대 ──
        val_mae = metrics.get("val", {}).get("MAE")
        if val_mae and mae and val_mae > 0:
            val_test_ratio = mae / val_mae
            if val_test_ratio > self.VAL_TEST_RATIO_THRESHOLD:
                analysis_parts.append(
                    f"val-test 갭 과대: test/val={val_test_ratio:.2f}"
                )
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

        # ── Check 6: Extreme 구간 성능 ──
        extreme_ratio = 1.0
        if normal and extreme:
            normal_mae = normal.get("MAE", mae)
            extreme_mae = extreme.get("MAE", mae)
            if normal_mae > 0:
                extreme_ratio = extreme_mae / normal_mae
            analysis_parts.append(
                f"Normal MAE: {normal_mae:.4f}, Extreme MAE: {extreme_mae:.4f} "
                f"(비율: {extreme_ratio:.2f})"
            )

            if extreme_ratio > 3.0 and iteration < max_iterations:
                suggestions.append(
                    f"극단 구간 성능 불량 (비율={extreme_ratio:.1f}). "
                    "TemporalMixer capacity 상향 시도."
                )
                return CriticVerdict(
                    verdict="RETRY_BLOCK", best_model=best_model,
                    best_metric=overall, normal_metric=normal, extreme_metric=extreme,
                    analysis=". ".join(analysis_parts), suggestions=suggestions,
                    ceiling_reached=False, iteration=iteration,
                )

        # ── Check 7: 이전 대비 개선 ──
        if self.prev_mae is not None:
            improvement = (self.prev_mae - mae) / self.prev_mae
            analysis_parts.append(f"이전 대비 개선: {improvement:.2%}")

            if improvement < self.MAE_IMPROVEMENT_THRESHOLD:
                analysis_parts.append("개선폭 미미 — ceiling")
                return CriticVerdict(
                    verdict="DONE", best_model=best_model,
                    best_metric=overall, normal_metric=normal, extreme_metric=extreme,
                    analysis=". ".join(analysis_parts), suggestions=suggestions,
                    ceiling_reached=True, iteration=iteration,
                )

        # ── Check 8: 첫 iteration → baseline 확보 → 다른 레시피 ──
        if iteration == 1:
            analysis_parts.append("baseline 확보. 다른 레시피 시도.")
            norm_mse = metrics.get("norm", {}).get("MSE")
            if norm_mse and norm_mse > 0.3:
                suggestions.append("norm_MSE 높음 — capacity 높은 레시피 시도")
            suggestions.append("다음 레시피를 시도하여 비교")
            return CriticVerdict(
                verdict="RETRY_RECIPE", best_model=best_model,
                best_metric=overall, normal_metric=normal, extreme_metric=extreme,
                analysis=". ".join(analysis_parts), suggestions=suggestions,
                ceiling_reached=False, iteration=iteration,
            )

        # ── Default: HP 미세조정 ──
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
