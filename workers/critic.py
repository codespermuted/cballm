"""Critic — rule-based 결과 분석. LLM 없이 결정론적 판단."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass


@dataclass
class CriticVerdict:
    """Critic의 구조화된 판단 결과."""
    verdict: str  # DONE | RETRY_FEATURES | RETRY_MODELS | RETRY_BOTH
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
    """Rule-based Critic. LLM을 사용하지 않고 결정론적으로 평가한다."""
    name = "critic"
    description = "Rule-based 결과 분석 + 피드백 라우팅"

    # ── 판단 기준 (범용 — 데이터 특성에 따라 자동 조절) ──
    MAE_IMPROVEMENT_THRESHOLD = 0.02   # 2% 이상 개선되어야 의미 있음
    EXTREME_MAE_RATIO = 2.0            # extreme MAE / normal MAE > 2.0이면 extreme 대응 필요
    CEILING_VARIANCE = 0.005           # 모델 간 MAE 차이 < 0.5%면 ceiling

    def __init__(self, cwd: str = "", rules: str = "",
                 prev_mae: float | None = None):
        self.cwd = cwd
        self.prev_mae = prev_mae  # 이전 iteration MAE (개선폭 판단용)

    def run(self, task: str) -> dict:
        """학습 결과 텍스트를 파싱하여 rule-based 판단을 내린다."""
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

    def _extract_metrics(self, text: str) -> dict:
        """학습 결과 텍스트에서 메트릭을 추출한다."""
        metrics = {}

        # METRICS: {"MAE": 0.xxx, ...} 패턴
        metrics_match = re.search(r'METRICS:\s*(\{[^}]+\})', text)
        if metrics_match:
            try:
                metrics["overall"] = json.loads(metrics_match.group(1))
            except json.JSONDecodeError:
                pass

        # 개별 메트릭 추출 (다양한 포맷 대응)
        for metric_name in ["MAE", "MSE", "RMSE", "MAPE"]:
            pattern = rf'{metric_name}[:\s=]+([0-9]+\.?[0-9]*)'
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                metrics.setdefault("overall", {})[metric_name] = float(match.group(1))

        # BEST_MODEL 추출
        model_match = re.search(r'BEST_MODEL[:\s=]+(\S+)', text)
        if model_match:
            metrics["best_model"] = model_match.group(1)

        # 정규화 메트릭 (벤치마크 비교용)
        norm_match = re.search(r'METRICS_NORM:\s*(\{[^}]+\})', text)
        if norm_match:
            try:
                metrics["norm"] = json.loads(norm_match.group(1))
            except json.JSONDecodeError:
                pass

        # Normal/Extreme 분리 메트릭 (있으면)
        normal_match = re.search(r'NORMAL[_ ]MAE[:\s=]+([0-9]+\.?[0-9]*)', text, re.IGNORECASE)
        extreme_match = re.search(r'EXTREME[_ ]MAE[:\s=]+([0-9]+\.?[0-9]*)', text, re.IGNORECASE)
        if normal_match:
            metrics["normal"] = {"MAE": float(normal_match.group(1))}
        # "N/A"면 extreme 샘플 없음 → 메트릭 없음
        if extreme_match:
            metrics["extreme"] = {"MAE": float(extreme_match.group(1))}

        return metrics

    def _extract_iteration(self, text: str) -> int:
        match = re.search(r'Iteration[:\s]*(\d+)', text, re.IGNORECASE)
        return int(match.group(1)) if match else 1

    def _extract_max_iterations(self, text: str) -> int:
        match = re.search(r'Iteration[:\s]*\d+/(\d+)', text, re.IGNORECASE)
        return int(match.group(1)) if match else 3

    def _judge(self, metrics: dict, iteration: int, max_iterations: int) -> CriticVerdict:
        """규칙 기반 판단."""
        overall = metrics.get("overall", {})
        normal = metrics.get("normal", {})
        extreme = metrics.get("extreme", {})
        best_model = metrics.get("best_model", "unknown")

        mae = overall.get("MAE")
        suggestions = []
        analysis_parts = []
        ceiling_reached = False

        # ── 판단 로직 ──

        # Case 1: 메트릭 자체가 없음 → 코드 실행 실패
        if not overall or mae is None:
            analysis_parts.append("메트릭 추출 실패 — 학습 코드가 정상 실행되지 않았을 가능성")
            suggestions.append("데이터 로드 경로와 컬럼명을 확인하고, 단순 모델(LightGBM)부터 시도")
            suggestions.append("코드 실행 에러 로그를 확인하고 import/경로 문제 수정")
            verdict = "RETRY_BOTH"

        # Case 2: 마지막 iteration → 무조건 DONE
        elif iteration >= max_iterations:
            analysis_parts.append(f"최대 반복({max_iterations})회 도달. 현재 결과로 최종 보고.")
            verdict = "DONE"

        # Case 3: 정상적 메트릭 존재
        else:
            analysis_parts.append(f"전체 MAE: {mae:.4f}")

            # Extreme 대응력 점검
            if normal and extreme:
                normal_mae = normal.get("MAE", mae)
                extreme_mae = extreme.get("MAE", mae)
                ratio = extreme_mae / normal_mae if normal_mae > 0 else 0
                analysis_parts.append(f"Normal MAE: {normal_mae:.4f}, Extreme MAE: {extreme_mae:.4f} (비율: {ratio:.2f})")

                if ratio > self.EXTREME_MAE_RATIO:
                    suggestions.append(f"극단 구간 MAE가 정상 대비 {ratio:.1f}배 — asymmetric loss 또는 regime gate 적용 권장")
            elif normal and not extreme:
                analysis_parts.append(f"Normal MAE: {normal.get('MAE', 'N/A'):.4f}, Extreme: 해당 구간 샘플 없음")

            # 정규화 메트릭 (벤치마크 비교용)
            norm = metrics.get("norm", {})
            if norm:
                analysis_parts.append(f"정규화 MSE: {norm.get('MSE', 'N/A')}, MAE: {norm.get('MAE', 'N/A')} (벤치마크 비교 기준)")

            # 이전 MAE 대비 개선폭 판단
            if self.prev_mae is not None and mae is not None:
                improvement = (self.prev_mae - mae) / self.prev_mae
                analysis_parts.append(f"이전 대비 개선: {improvement:.2%}")

                if improvement < self.MAE_IMPROVEMENT_THRESHOLD:
                    analysis_parts.append("개선폭 미미 — ceiling 가능성")
                    ceiling_reached = True
                    verdict = "DONE"
                else:
                    analysis_parts.append("유의미한 개선 — 추가 시도 가치 있음")
                    suggestions.append("현재 접근 유지하면서 피쳐 조합 변경 시도")
                    verdict = "RETRY_FEATURES"
            elif iteration == 1:
                # 첫 iteration: baseline 확보, 다음에 개선 시도
                suggestions.append("현재 baseline 대비 피쳐 조합 변경 시도")
                suggestions.append("앙상블 또는 다른 모델 아키텍처 시도")
                verdict = "RETRY_FEATURES"
            else:
                analysis_parts.append("추가 개선 여지 제한적 — 현재 결과로 마무리 권장")
                verdict = "DONE"

        return CriticVerdict(
            verdict=verdict,
            best_model=best_model,
            best_metric=overall,
            normal_metric=normal,
            extreme_metric=extreme,
            analysis=". ".join(analysis_parts),
            suggestions=suggestions,
            ceiling_reached=ceiling_reached,
            iteration=iteration,
        )
