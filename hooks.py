"""CBALLM Hooks — Pre/Post Train 진단 모듈 (전부 rule-based).

패턴 A: claw-code의 hooks.rs에서 차용.
- PreTrainHook:  블록 조합 사전 검증 (ALLOW/DENY/WARN)
- PostTrainHook: 잔차 진단 + 블록 기여도 + 진단 지시문 생성

LLM 불사용. 모든 진단은 통계 검정과 규칙으로 수행.
"""
from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field, asdict

import numpy as np


# ══════════════════════════════════════════════════════════════
#  Exit Codes
# ══════════════════════════════════════════════════════════════

class HookResult:
    ALLOW = "ALLOW"
    DENY = "DENY"
    WARN = "WARN"


# ══════════════════════════════════════════════════════════════
#  Diagnosis — 구조화된 진단 스키마
# ══════════════════════════════════════════════════════════════

@dataclass
class Diagnosis:
    """PostTrainHook의 구조화된 진단 결과.

    각 필드의 판정 기준 (rule-based):

    failure_mode:
      - "underfitting_global": norm_MSE > 0.5 AND 잔차 자기상관 있음
      - "underfitting_tail":  극단 구간 MAE/전체 MAE > 2.0
      - "overfitting_spike":  val-test 갭 > 2.0 AND best_epoch < epochs/3
      - "plateau":            이전 대비 개선 < 1%
      - "training_failure":   best_epoch == 0 또는 메트릭 없음
      - "acceptable":         위 조건 모두 해당 없음

    residual_pattern:
      - "autocorrelated":          Ljung-Box p < 0.05 (잔차에 패턴 남음)
      - "heteroscedastic":         구간별 분산 비 > 3.0 (이분산성)
      - "biased":                  |잔차 skew| > 1.0 (한쪽 편향)
      - "distribution_mismatch":   KS test p < 0.05 (선택 분포와 잔차 불일치)
      - "clean":                   위 조건 모두 해당 없음

    underperforming_regime:
      - "high_volatility":    후반 구간 MAE가 전체의 2배 이상
      - "low_demand":         초반 구간 MAE가 전체의 2배 이상
      - "transition":         중반 구간이 최악
      - "none":               구간 차이 미미

    suggested_direction:
      - "increase_capacity":  underfitting_global
      - "add_asymmetric_loss": underfitting_tail 또는 heteroscedastic
      - "reduce_capacity":    overfitting_spike
      - "change_recipe":      plateau
      - "fix_training":       training_failure
      - "add_volatility_block": high_volatility regime
      - "change_distribution": distribution_mismatch
      - "maintain":           acceptable + clean

    block_blacklist:
      - 이번 라운드에서 BlockAttribution verdict == "ineffective"인 블록들
    """
    failure_mode: str = "acceptable"
    residual_pattern: str = "clean"
    underperforming_regime: str = "none"
    suggested_direction: str = "maintain"
    block_blacklist: list[str] = field(default_factory=list)

    def to_prompt_directive(self) -> str:
        """ArchitectPromptBuilder Section 3용 pre-digested 텍스트.

        Architect에게 "왜 + 무엇을" 1-3줄로 전달.
        """
        if self.failure_mode == "acceptable" and self.residual_pattern == "clean":
            return "이전 라운드 양호. 다른 레시피로 비교."

        parts = []

        # failure mode → 핵심 문제
        mode_desc = {
            "underfitting_global": "전체 성능 부족 (norm_MSE 높음, 잔차에 패턴 남음)",
            "underfitting_tail": "극단 구간 예측 실패",
            "overfitting_spike": "과적합 — val 대비 test 성능 급락",
            "plateau": "성능 정체 — 이전 라운드 대비 개선 미미",
            "training_failure": "학습 실패 — lr/warmup 조정 필요",
            "acceptable": "성능 양호 (잔차 이슈 확인)",
        }
        if self.failure_mode in mode_desc and self.failure_mode != "acceptable":
            parts.append(f"문제: {mode_desc[self.failure_mode]}")

        # residual pattern → 잔차 특성
        pattern_desc = {
            "autocorrelated": "잔차에 시간 패턴 남음 -> temporal capacity 부족",
            "heteroscedastic": "잔차 분산 불균일 -> 구간별 다른 전략 필요",
            "biased": "예측 편향 -> loss 함수 또는 constraint 조정",
            "distribution_mismatch": "선택 분포와 잔차 불일치 -> 다른 분포 시도",
        }
        if self.residual_pattern in pattern_desc:
            parts.append(f"잔차: {pattern_desc[self.residual_pattern]}")

        # regime → 취약 구간
        if self.underperforming_regime != "none":
            regime_desc = {
                "high_volatility": "고변동 구간에서 성능 저하",
                "low_demand": "저수요 구간에서 성능 저하",
                "transition": "전환 구간에서 성능 저하",
            }
            parts.append(f"취약구간: {regime_desc.get(self.underperforming_regime, self.underperforming_regime)}")

        # direction → 액션
        direction_desc = {
            "increase_capacity": "-> capacity 높은 블록으로 교체",
            "add_asymmetric_loss": "-> AsymmetricLoss 또는 Huber 시도",
            "reduce_capacity": "-> 더 단순한 모델 + 정규화 강화",
            "change_recipe": "-> 완전히 다른 레시피 시도",
            "fix_training": "-> lr 낮추기, warmup 추가, scheduler 변경",
            "add_volatility_block": "-> ConvMix 또는 regime gate 시도",
            "change_distribution": "-> 다른 분포 선택 (student_t/log_normal/mixture)",
            "maintain": "-> 현재 구성 유지",
        }
        if self.suggested_direction in direction_desc and self.suggested_direction != "maintain":
            parts.append(f"방향: {direction_desc[self.suggested_direction]}")

        # blacklist
        if self.block_blacklist:
            parts.append(f"회피: {', '.join(self.block_blacklist)}")

        return "\n".join(parts)


# ══════════════════════════════════════════════════════════════
#  Dataclass Interfaces
# ══════════════════════════════════════════════════════════════

@dataclass
class ResidualDiagnosis:
    """잔차 분석 결과."""
    has_autocorrelation: bool = False     # Ljung-Box: 잔차에 패턴이 남아있는가
    lb_pvalue: float = 1.0
    has_heteroscedasticity: bool = False  # Breusch-Pagan: 이분산성이 있는가
    bp_pvalue: float = 1.0
    segment_mae: dict = field(default_factory=dict)  # 구간별 MAE
    worst_segment: str = ""
    worst_segment_mae: float = 0.0
    overall_mae: float = 0.0
    residual_skew: float = 0.0
    residual_kurtosis: float = 0.0

    def summary(self) -> str:
        parts = []
        if self.has_autocorrelation:
            parts.append(f"잔차 자기상관 존재 (LB p={self.lb_pvalue:.3f})")
        if self.has_heteroscedasticity:
            parts.append(f"잔차 이분산성 존재 (BP p={self.bp_pvalue:.3f})")
        if self.worst_segment and self.overall_mae > 0:
            ratio = self.worst_segment_mae / self.overall_mae
            parts.append(f"최악 구간: {self.worst_segment} (MAE 비율={ratio:.1f}x)")
        if abs(self.residual_skew) > 1.0:
            direction = "과소예측 편향" if self.residual_skew > 0 else "과대예측 편향"
            parts.append(f"잔차 {direction} (skew={self.residual_skew:.2f})")
        return "; ".join(parts) if parts else "잔차 정상"


@dataclass
class BlockAttribution:
    """블록별 기여도 추적."""
    block_name: str
    slot: str            # encoder, temporal_mixer, channel_mixer, normalizer
    contribution: float  # 0~1 (ablation 기반 추정)
    verdict: str         # "effective" | "marginal" | "ineffective"

    def summary(self) -> str:
        return f"{self.slot}={self.block_name}: {self.verdict} ({self.contribution:.2f})"


@dataclass
class PreTrainVerdict:
    """PreTrainHook 결과."""
    exit_code: str  # ALLOW | DENY | WARN
    reason: str
    similar_round: int | None = None  # DENY 시 유사했던 라운드

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PostTrainDiagnosis:
    """PostTrainHook 종합 진단."""
    residual: ResidualDiagnosis
    attributions: list[BlockAttribution]
    diagnosis: Diagnosis
    directive: str  # Architect에게 전달할 pre-digested 진단 지시문
    # v2.1 확장 진단
    curve: object | None = None        # CurveDiagnosis
    horizon: object | None = None      # HorizonDiagnosis
    feature: object | None = None      # FeatureDiagnosis
    disagreement: object | None = None # DisagreementDiagnosis

    def to_dict(self) -> dict:
        d = {
            "residual": asdict(self.residual),
            "attributions": [asdict(a) for a in self.attributions],
            "diagnosis": asdict(self.diagnosis),
            "directive": self.directive,
        }
        if self.curve:
            d["curve"] = asdict(self.curve) if hasattr(self.curve, '__dataclass_fields__') else {}
        if self.horizon:
            d["horizon"] = asdict(self.horizon) if hasattr(self.horizon, '__dataclass_fields__') else {}
        if self.disagreement:
            d["disagreement"] = asdict(self.disagreement) if hasattr(self.disagreement, '__dataclass_fields__') else {}
        return d


@dataclass
class RoundRecord:
    """한 라운드의 기록."""
    round_num: int
    config: dict
    config_hash: str
    metrics: dict
    verdict: str
    diagnosis: PostTrainDiagnosis | None = None

    def compact_summary(self) -> str:
        """1줄 요약."""
        recipe = self.config.get("_recipe_name", "custom")
        mixer = self.config.get("temporal_mixer", {})
        mixer_name = mixer.get("type", "?") if isinstance(mixer, dict) else str(mixer)
        enc = self.config.get("encoder", {})
        enc_name = enc.get("type", "?") if isinstance(enc, dict) else str(enc)
        norm_mse = self.metrics.get("norm_MSE", "?")
        mae = self.metrics.get("MAE", "?")

        summary = f"R{self.round_num}: {recipe}({enc_name}+{mixer_name}) -> MAE={mae}, norm_MSE={norm_mse}"

        if self.diagnosis:
            # 블록 기여도 중 ineffective만
            ineffective = [a for a in self.diagnosis.attributions if a.verdict == "ineffective"]
            if ineffective:
                names = ", ".join(f"{a.block_name}" for a in ineffective)
                summary += f" [{names} 무효]"
            # 잔차 핵심 이슈
            if self.diagnosis.residual.has_autocorrelation:
                summary += " [잔차패턴남음]"
            if self.diagnosis.residual.has_heteroscedasticity:
                summary += " [이분산성]"

        summary += f" -> {self.verdict}"
        return summary


# ══════════════════════════════════════════════════════════════
#  Diversity Ledger — 과거 시도 기록 + 유사도 체크
# ══════════════════════════════════════════════════════════════

class DiversityLedger:
    """과거 시도한 블록 조합을 기록하고 중복을 방지한다."""

    def __init__(self):
        self.rounds: list[RoundRecord] = []
        self._hashes: set[str] = set()

    def add_round(self, record: RoundRecord):
        self.rounds.append(record)
        self._hashes.add(record.config_hash)

    @staticmethod
    def config_hash(config: dict) -> str:
        """블록 조합의 fingerprint. HP는 무시하고 블록 타입만 해싱."""
        key_parts = []
        for slot in ["normalizer", "decomposition", "encoder", "temporal_mixer", "channel_mixer", "loss"]:
            val = config.get(slot)
            if val is None:
                key_parts.append(f"{slot}=null")
            elif isinstance(val, dict):
                key_parts.append(f"{slot}={val.get('type', '?')}")
            else:
                key_parts.append(f"{slot}={val}")
        key = "|".join(key_parts)
        return hashlib.md5(key.encode()).hexdigest()[:8]

    def is_duplicate(self, config: dict) -> tuple[bool, int | None]:
        """동일 블록 조합이 이미 시도되었는지 확인."""
        h = self.config_hash(config)
        if h in self._hashes:
            for r in self.rounds:
                if r.config_hash == h:
                    return True, r.round_num
        return False, None

    def is_similar(self, config: dict, threshold: float = 0.8) -> tuple[bool, int | None]:
        """유사한 블록 조합이 이미 시도되었는지 확인.

        블록 5개 슬롯 중 threshold 비율 이상 동일하면 유사로 판단.
        """
        slots = ["normalizer", "decomposition", "encoder", "temporal_mixer", "channel_mixer", "loss"]
        for r in self.rounds:
            same = 0
            for slot in slots:
                new_val = _get_block_type(config, slot)
                old_val = _get_block_type(r.config, slot)
                if new_val == old_val:
                    same += 1
            if same / len(slots) >= threshold:
                return True, r.round_num
        return False, None

    def get_tried_blocks(self, slot: str) -> list[str]:
        """특정 슬롯에서 이미 시도한 블록 목록."""
        tried = set()
        for r in self.rounds:
            val = _get_block_type(r.config, slot)
            if val:
                tried.add(val)
        return sorted(tried)

    def best_round(self) -> RoundRecord | None:
        """가장 좋은 성능의 라운드."""
        best = None
        for r in self.rounds:
            norm_mse = r.metrics.get("norm_MSE")
            if norm_mse is not None:
                if best is None or norm_mse < best.metrics.get("norm_MSE", float("inf")):
                    best = r
        return best


# ══════════════════════════════════════════════════════════════
#  Residual Analyzer — 잔차 통계 검정
# ══════════════════════════════════════════════════════════════

class ResidualAnalyzer:
    """Rule-based 잔차 분석. 모든 검정이 rule-based/통계적."""

    @staticmethod
    def analyze(residuals: np.ndarray, pred_len: int,
                n_segments: int = 4) -> ResidualDiagnosis:
        """잔차 벡터를 받아 진단."""
        if len(residuals) == 0:
            return ResidualDiagnosis()

        # Ljung-Box (자기상관 검정)
        has_ac, lb_p = ResidualAnalyzer._ljung_box(residuals, pred_len)

        # 이분산성 (간이 Breusch-Pagan)
        has_het, bp_p = ResidualAnalyzer._heteroscedasticity(residuals, n_segments)

        # 구간별 MAE
        segment_mae = ResidualAnalyzer._segment_mae(residuals, n_segments)
        overall_mae = float(np.abs(residuals).mean())

        worst_seg = ""
        worst_mae = 0.0
        for seg, mae in segment_mae.items():
            if mae > worst_mae:
                worst_mae = mae
                worst_seg = seg

        # 잔차 분포
        skew = float(_safe_skew(residuals))
        kurtosis = float(_safe_kurtosis(residuals))

        return ResidualDiagnosis(
            has_autocorrelation=has_ac, lb_pvalue=lb_p,
            has_heteroscedasticity=has_het, bp_pvalue=bp_p,
            segment_mae=segment_mae, worst_segment=worst_seg,
            worst_segment_mae=worst_mae, overall_mae=overall_mae,
            residual_skew=skew, residual_kurtosis=kurtosis,
        )

    @staticmethod
    def _ljung_box(residuals: np.ndarray, pred_len: int) -> tuple[bool, float]:
        """간이 Ljung-Box 검정."""
        n = len(residuals)
        max_lag = min(pred_len, n // 5, 20)
        if max_lag < 1 or n < 20:
            return False, 1.0

        mean = residuals.mean()
        var = ((residuals - mean) ** 2).sum()
        if var == 0:
            return False, 1.0

        q_stat = 0.0
        for k in range(1, max_lag + 1):
            acf_k = ((residuals[:n - k] - mean) * (residuals[k:] - mean)).sum() / var
            q_stat += (acf_k ** 2) / (n - k)
        q_stat *= n * (n + 2)

        # chi-squared 근사: df=max_lag, p<0.05면 자기상관 있음
        # 간이 판정: Q > 1.5 * max_lag이면 유의
        threshold = 1.5 * max_lag
        p_approx = max(0.001, 1.0 - q_stat / (2 * max_lag + q_stat))
        return q_stat > threshold, round(p_approx, 4)

    @staticmethod
    def _heteroscedasticity(residuals: np.ndarray,
                            n_segments: int) -> tuple[bool, float]:
        """간이 이분산성 검정: 구간별 분산 비교."""
        n = len(residuals)
        if n < n_segments * 10:
            return False, 1.0

        seg_size = n // n_segments
        variances = []
        for i in range(n_segments):
            seg = residuals[i * seg_size:(i + 1) * seg_size]
            variances.append(np.var(seg))

        if min(variances) == 0:
            return False, 1.0

        # Bartlett-like: max/min variance ratio
        ratio = max(variances) / min(variances)
        # ratio > 3이면 이분산성 의심
        p_approx = max(0.001, 1.0 / ratio)
        return ratio > 3.0, round(p_approx, 4)

    @staticmethod
    def _segment_mae(residuals: np.ndarray, n_segments: int) -> dict[str, float]:
        """시간축 구간별 MAE."""
        n = len(residuals)
        seg_size = max(1, n // n_segments)
        result = {}
        labels = ["초반", "중반", "후반", "말미"]
        for i in range(min(n_segments, 4)):
            seg = residuals[i * seg_size:(i + 1) * seg_size]
            if len(seg) > 0:
                label = labels[i] if i < len(labels) else f"seg{i}"
                result[label] = round(float(np.abs(seg).mean()), 4)
        return result


# ══════════════════════════════════════════════════════════════
#  Distribution Fit Checker — 분포 적합도 검증
# ══════════════════════════════════════════════════════════════

class DistributionFitChecker:
    """잔차에 대해 선택된 분포의 적합도를 검증한다 (KS test).

    p < 0.05이면 선택 분포와 잔차 불일치 → distribution_mismatch.
    """

    @staticmethod
    def check(residuals: np.ndarray, distribution: str) -> float:
        """KS test p-value를 반환. 높을수록 적합."""
        if len(residuals) < 20:
            return 1.0

        # 표준화
        mean = residuals.mean()
        std = residuals.std()
        if std == 0:
            return 1.0
        z = (residuals - mean) / std

        try:
            from scipy.stats import kstest, norm, t as t_dist, lognorm

            if distribution == "gaussian":
                _, p = kstest(z, "norm")
            elif distribution == "student_t":
                # df 추정: excess kurtosis로 근사
                kurt = float(((z ** 4).mean()) - 3)
                df_est = max(3, 6 / kurt + 4) if kurt > 0 else 30
                _, p = kstest(z, t_dist(df=df_est).cdf)
            elif distribution == "log_normal":
                # 양수 잔차만 사용 (log-normal은 양수)
                pos = residuals[residuals > 0]
                if len(pos) < 20:
                    return 1.0
                log_z = (np.log(pos) - np.log(pos).mean()) / (np.log(pos).std() + 1e-8)
                _, p = kstest(log_z, "norm")
            elif distribution == "mixture_gaussian":
                # mixture는 KS test 어려움 → 간이: bimodality 체크
                # Hartigan's dip test 대신 간이 kurtosis 기반
                kurt = float(((z ** 4).mean()) - 3)
                # bimodal이면 kurtosis가 음수 경향
                p = 0.5 if kurt < -0.5 else 0.1  # 간이 판정
            else:
                p = 1.0

            return round(float(p), 4)

        except ImportError:
            # scipy 없으면 간이 판정: |skew| + |kurtosis|로 추정
            skew = abs(float(((z ** 3).mean())))
            kurt = abs(float(((z ** 4).mean()) - 3))
            if distribution == "gaussian":
                return 0.5 if (skew < 0.5 and kurt < 2) else 0.01
            return 0.5  # 다른 분포는 scipy 없으면 pass


# ══════════════════════════════════════════════════════════════
#  Block Attribution — 블록별 기여도 추정
# ══════════════════════════════════════════════════════════════

class BlockAttributor:
    """블록별 기여도를 추정한다.

    완전한 ablation은 비용이 높으므로, 다음 휴리스틱을 사용:
    - Encoder: FourierEmbedding이면 dominant ACF 대비 기여도 추정
    - TemporalMixer: capacity 대비 norm_MSE 개선 비율
    - ChannelMixer: 변수간 상관 대비 기여도
    - Normalizer: val-test 갭으로 기여도 추정
    """

    @staticmethod
    def attribute(config: dict, metrics: dict,
                  profile: dict, ledger: DiversityLedger) -> list[BlockAttribution]:
        """현재 config + metrics + 이전 라운드 비교로 블록 기여도 추정."""
        attrs = []

        # ── Encoder ──
        enc_type = _get_block_type(config, "encoder")
        if enc_type == "FourierEmbedding":
            max_acf = profile.get("max_acf_at_known_periods", 0)
            if max_acf < 0.4:
                attrs.append(BlockAttribution(
                    block_name=enc_type, slot="encoder",
                    contribution=0.05, verdict="ineffective",
                ))
            elif max_acf < 0.7:
                attrs.append(BlockAttribution(
                    block_name=enc_type, slot="encoder",
                    contribution=0.3, verdict="marginal",
                ))
            else:
                attrs.append(BlockAttribution(
                    block_name=enc_type, slot="encoder",
                    contribution=0.7, verdict="effective",
                ))
        elif enc_type == "PatchEmbedding":
            n_rows = profile.get("n_rows", 0)
            if n_rows < 3000:
                attrs.append(BlockAttribution(
                    block_name=enc_type, slot="encoder",
                    contribution=0.1, verdict="marginal",
                ))
            else:
                attrs.append(BlockAttribution(
                    block_name=enc_type, slot="encoder",
                    contribution=0.6, verdict="effective",
                ))

        # ── TemporalMixer: 이전 라운드 대비 개선으로 추정 ──
        mixer_type = _get_block_type(config, "temporal_mixer")
        if mixer_type and ledger.rounds:
            best = ledger.best_round()
            norm_mse = metrics.get("norm_MSE", 1.0)
            if best and best.metrics.get("norm_MSE"):
                delta = best.metrics["norm_MSE"] - norm_mse
                if delta > 0.02:
                    attrs.append(BlockAttribution(
                        block_name=mixer_type, slot="temporal_mixer",
                        contribution=0.8, verdict="effective",
                    ))
                elif delta > -0.01:
                    attrs.append(BlockAttribution(
                        block_name=mixer_type, slot="temporal_mixer",
                        contribution=0.3, verdict="marginal",
                    ))
                else:
                    attrs.append(BlockAttribution(
                        block_name=mixer_type, slot="temporal_mixer",
                        contribution=0.05, verdict="ineffective",
                    ))

        # ── ChannelMixer ──
        ch_type = _get_block_type(config, "channel_mixer")
        if ch_type and ch_type != "null":
            high_corr = profile.get("high_cross_corr_pairs", 0)
            if high_corr < 3:
                attrs.append(BlockAttribution(
                    block_name=ch_type, slot="channel_mixer",
                    contribution=0.05, verdict="ineffective",
                ))
            else:
                attrs.append(BlockAttribution(
                    block_name=ch_type, slot="channel_mixer",
                    contribution=0.5, verdict="effective",
                ))

        # ── Normalizer ──
        norm_type = _get_block_type(config, "normalizer")
        if norm_type and norm_type != "null":
            val_mae = metrics.get("val_MAE", 0)
            test_mae = metrics.get("MAE", 0)
            if val_mae > 0 and test_mae > 0:
                gap_ratio = test_mae / val_mae
                if gap_ratio < 1.5:
                    attrs.append(BlockAttribution(
                        block_name=norm_type, slot="normalizer",
                        contribution=0.7, verdict="effective",
                    ))
                else:
                    attrs.append(BlockAttribution(
                        block_name=norm_type, slot="normalizer",
                        contribution=0.2, verdict="marginal",
                    ))

        return attrs


# ══════════════════════════════════════════════════════════════
#  PreTrainHook — 사전 검증
# ══════════════════════════════════════════════════════════════

class PreTrainHook:
    """Trainer 실행 전 블록 조합을 사전 검증한다."""

    @staticmethod
    def check(config: dict, profile: dict,
              ledger: DiversityLedger) -> PreTrainVerdict:
        """ALLOW/DENY/WARN 반환."""

        # ── Check 1: 완전 동일 조합 → DENY ──
        is_dup, dup_round = ledger.is_duplicate(config)
        if is_dup:
            return PreTrainVerdict(
                exit_code=HookResult.DENY,
                reason=f"Round {dup_round}과 동일한 블록 조합. 다른 조합 시도 필요.",
                similar_round=dup_round,
            )

        # ── Check 2: 유사 조합 (80% 이상 동일) → WARN ──
        is_sim, sim_round = ledger.is_similar(config, threshold=0.8)
        if is_sim:
            return PreTrainVerdict(
                exit_code=HookResult.WARN,
                reason=f"Round {sim_round}과 유사한 조합 (80%+ 동일). HP만 다름.",
                similar_round=sim_round,
            )

        # ── Check 3: 데이터 크기 대비 모델 복잡도 ──
        n_rows = profile.get("n_rows", 0)
        mixer_type = _get_block_type(config, "temporal_mixer")

        high_capacity = {"AttentionMix", "PatchAttentionMix", "InvertedAttentionMix"}
        if mixer_type in high_capacity and n_rows < 5000:
            return PreTrainVerdict(
                exit_code=HookResult.WARN,
                reason=f"데이터 {n_rows}행에 {mixer_type}는 과적합 위험. "
                       f"n_rows >= 10000 권장.",
            )

        medium_capacity = {"GatedMLPMix", "ConvMix", "PatchMLPMix"}
        if mixer_type in medium_capacity and n_rows < 2000:
            return PreTrainVerdict(
                exit_code=HookResult.WARN,
                reason=f"데이터 {n_rows}행에 {mixer_type}는 과적합 위험.",
            )

        return PreTrainVerdict(exit_code=HookResult.ALLOW, reason="통과")


# ══════════════════════════════════════════════════════════════
#  PostTrainHook — 학습 후 진단
# ══════════════════════════════════════════════════════════════

class PostTrainHook:
    """Trainer 실행 후 잔차 분석 + 블록 기여도 + 진단 지시문 생성."""

    @staticmethod
    def diagnose(config: dict, metrics: dict, profile: dict,
                 ledger: DiversityLedger,
                 residuals: np.ndarray | None = None,
                 pred_len: int = 96,
                 prev_norm_mse: float | None = None,
                 best_epoch: int | None = None,
                 max_epochs: int = 100,
                 val_test_ratio: float = 1.0,
                 distribution: str | None = None,
                 train_loss_history: list[float] | None = None,
                 val_loss_history: list[float] | None = None,
                 val_mae_by_step: list[float] | None = None,
                 val_predictions: np.ndarray | None = None) -> PostTrainDiagnosis:
        """종합 진단 (v2.1 확장)."""
        from cballm.diagnostics import (
            CurveAnalyzer, HorizonDiagnoser, DisagreementDiagnoser,
        )

        # 잔차 분석
        if residuals is not None and len(residuals) > 0:
            residual_diag = ResidualAnalyzer.analyze(residuals, pred_len)
        else:
            residual_diag = ResidualDiagnosis()

        # 분포 적합도 검증
        dist_fit_pvalue = None
        if residuals is not None and len(residuals) > 20 and distribution:
            dist_fit_pvalue = DistributionFitChecker.check(residuals, distribution)

        # 블록 기여도
        attributions = BlockAttributor.attribute(config, metrics, profile, ledger)

        # ── 4종 확장 진단 ──

        # 1) CurveAnalyzer
        curve_diag = None
        if train_loss_history and val_loss_history:
            curve_diag = CurveAnalyzer.analyze(
                train_loss_history, val_loss_history,
                best_epoch=best_epoch or 0,
                max_epochs=max_epochs,
            )

        # 2) HorizonDiagnoser
        horizon_diag = None
        if val_mae_by_step and len(val_mae_by_step) >= 4:
            horizon_diag = HorizonDiagnoser.analyze(val_mae_by_step)

        # 3) DisagreementDiagnoser (라운드 2 이후)
        disagree_diag = None
        if ledger.rounds and val_predictions is not None:
            prev_round = ledger.rounds[-1]
            prev_preds = getattr(prev_round, '_val_predictions', None)
            if prev_preds is not None:
                disagree_diag = DisagreementDiagnoser.analyze(prev_preds, val_predictions)

        # ── 구조화된 Diagnosis 생성 ──
        diagnosis = PostTrainHook._classify(
            residual_diag, attributions, metrics,
            prev_norm_mse=prev_norm_mse,
            best_epoch=best_epoch,
            max_epochs=max_epochs,
            val_test_ratio=val_test_ratio,
            dist_fit_pvalue=dist_fit_pvalue,
        )

        # curve.pattern에 의한 failure_mode 오버라이드
        if curve_diag:
            if curve_diag.pattern == "lr_instability":
                diagnosis.failure_mode = "training_failure"
                diagnosis.suggested_direction = "fix_training"
            elif curve_diag.pattern == "overfitting" and diagnosis.failure_mode not in ("training_failure",):
                diagnosis.failure_mode = "overfitting_spike"
                diagnosis.suggested_direction = "reduce_capacity"
            elif curve_diag.pattern == "still_improving":
                if "max_epochs 증가" not in (diagnosis.suggested_direction or ""):
                    diagnosis.suggested_direction += " + add_epochs" if diagnosis.suggested_direction != "maintain" else "add_epochs"

        # ── directive 생성 (우선순위 필터링) ──
        directive = diagnosis.to_prompt_directive()

        # 우선순위에 따라 확장 진단 추가 (최대 2줄)
        extra_lines = PostTrainHook._build_extra_directive(
            diagnosis.failure_mode, curve_diag, horizon_diag, None, disagree_diag,
        )
        if extra_lines:
            directive += "\n" + "\n".join(extra_lines)

        # _build_directive 상세 보충
        detail = PostTrainHook._build_directive(
            residual_diag, attributions, metrics, config,
        )
        if detail and detail != "진단 이상 없음":
            existing = set(directive.split("\n"))
            for line in detail.split("\n"):
                if line and line not in existing:
                    directive += "\n" + line
                    break  # 1줄만 보충

        return PostTrainDiagnosis(
            residual=residual_diag,
            attributions=attributions,
            diagnosis=diagnosis,
            directive=directive,
            curve=curve_diag,
            horizon=horizon_diag,
            disagreement=disagree_diag,
        )

    @staticmethod
    def _build_extra_directive(failure_mode: str,
                                curve, horizon, feature, disagreement) -> list[str]:
        """우선순위 필터링 후 확장 진단을 최대 2줄로."""
        lines: list[str] = []

        # 우선순위:
        # training_failure → curve만
        # overfitting → curve + 방향만
        # underfitting → 전부
        # plateau → feature 특히 중요
        # acceptable → horizon + disagreement

        if failure_mode == "training_failure":
            if curve:
                lines.append(curve.summary())
            return lines[:2]

        if failure_mode == "overfitting_spike":
            if curve:
                lines.append(curve.summary())
            return lines[:2]

        # underfitting, plateau, acceptable → 모두 표시 (최대 2줄)
        if curve and curve.pattern != "healthy":
            lines.append(curve.summary())
        if horizon and horizon.pattern != "uniform":
            lines.append(horizon.summary())
        if disagreement and disagreement.stability < 0.8:
            lines.append(disagreement.summary())

        return lines[:2]

    @staticmethod
    def _classify(residual: ResidualDiagnosis,
                  attributions: list[BlockAttribution],
                  metrics: dict,
                  prev_norm_mse: float | None = None,
                  best_epoch: int | None = None,
                  dist_fit_pvalue: float | None = None,
                  max_epochs: int = 100,
                  val_test_ratio: float = 1.0) -> Diagnosis:
        """Rule-based Diagnosis 판정."""

        norm_mse = metrics.get("norm_MSE", None)

        # ── failure_mode ─��
        invalid_threshold = max(1, int(max_epochs * 0.05))
        if best_epoch is not None and best_epoch < invalid_threshold:
            failure_mode = "training_failure"
        elif val_test_ratio > 2.0 and best_epoch is not None and best_epoch < max_epochs / 3:
            failure_mode = "overfitting_spike"
        elif prev_norm_mse is not None and norm_mse is not None:
            improvement = (prev_norm_mse - norm_mse) / prev_norm_mse if prev_norm_mse > 0 else 0
            if improvement < 0.01 and norm_mse < 0.3:
                # 이미 양호한데 정체 → 진짜 plateau
                failure_mode = "plateau"
            elif norm_mse > 0.5 and residual.has_autocorrelation:
                # 아직 나쁨 (개선 정체여도 underfitting이 우선)
                failure_mode = "underfitting_global"
            elif improvement < 0.01 and norm_mse >= 0.3:
                # 나쁜데 정체 → underfitting (plateau가 아님)
                failure_mode = "underfitting_global"
            elif residual.worst_segment_mae > residual.overall_mae * 2.0:
                failure_mode = "underfitting_tail"
            else:
                failure_mode = "acceptable"
        elif norm_mse is not None and norm_mse > 0.5 and residual.has_autocorrelation:
            failure_mode = "underfitting_global"
        elif residual.worst_segment_mae > residual.overall_mae * 2.0 and residual.overall_mae > 0:
            failure_mode = "underfitting_tail"
        else:
            failure_mode = "acceptable"

        # ── residual_pattern ──
        if residual.has_autocorrelation and residual.has_heteroscedasticity:
            residual_pattern = "autocorrelated"  # 자기상관이 더 근본 원인
        elif residual.has_autocorrelation:
            residual_pattern = "autocorrelated"
        elif residual.has_heteroscedasticity:
            residual_pattern = "heteroscedastic"
        elif abs(residual.residual_skew) > 1.0:
            residual_pattern = "biased"
        elif dist_fit_pvalue is not None and dist_fit_pvalue < 0.05:
            residual_pattern = "distribution_mismatch"
        else:
            residual_pattern = "clean"

        # ── underperforming_regime ──
        # segment key: ResidualAnalyzer가 ["초반", "중반", "후반", "말미"] 또는 "seg{i}" 생성
        EARLY_KEYS = {"초반", "seg0"}
        MID_KEYS = {"중반", "seg1"}
        LATE_KEYS = {"후반", "말미", "seg2", "seg3"}

        seg = residual.segment_mae
        overall = residual.overall_mae
        if overall > 0 and seg:
            worst_key = max(seg, key=seg.get) if seg else ""
            worst_ratio = seg.get(worst_key, 0) / overall if overall > 0 else 1.0
            if worst_ratio > 2.0:
                if worst_key in LATE_KEYS:
                    regime = "high_volatility"
                elif worst_key in EARLY_KEYS:
                    regime = "low_demand"
                elif worst_key in MID_KEYS:
                    regime = "transition"
                else:
                    regime = "none"
            else:
                regime = "none"
        else:
            regime = "none"

        # ── suggested_direction ──
        direction_map = {
            "underfitting_global": "increase_capacity",
            "underfitting_tail": "add_asymmetric_loss",
            "overfitting_spike": "reduce_capacity",
            "plateau": "change_recipe",
            "training_failure": "fix_training",
        }
        direction = direction_map.get(failure_mode, "maintain")

        # regime이 high_volatility면 direction 오버라이드
        if regime == "high_volatility" and failure_mode != "training_failure":
            direction = "add_volatility_block"

        # 분포 불일치면 direction 오버라이드
        if residual_pattern == "distribution_mismatch":
            direction = "change_distribution"

        # ── block_blacklist ──
        blacklist = [a.block_name for a in attributions if a.verdict == "ineffective"]

        return Diagnosis(
            failure_mode=failure_mode,
            residual_pattern=residual_pattern,
            underperforming_regime=regime,
            suggested_direction=direction,
            block_blacklist=blacklist,
        )

    @staticmethod
    def _build_directive(residual: ResidualDiagnosis,
                         attributions: list[BlockAttribution],
                         metrics: dict, config: dict) -> str:
        """Layer 2 해석 수준의 진단 지시문.

        Architect에게 "왜 실패했는지 + 무엇을 바꿔야 하는지"를 전달.
        """
        parts = []

        # 잔차 진단
        residual_summary = residual.summary()
        if residual_summary != "잔차 정상":
            parts.append(f"[잔차] {residual_summary}")

        # 비효과적 블록
        ineffective = [a for a in attributions if a.verdict == "ineffective"]
        marginal = [a for a in attributions if a.verdict == "marginal"]
        if ineffective:
            blocks = ", ".join(f"{a.slot}={a.block_name}" for a in ineffective)
            parts.append(f"[무효블록] {blocks} -> 교체 또는 제거 권장")
        if marginal:
            blocks = ", ".join(f"{a.slot}={a.block_name}" for a in marginal)
            parts.append(f"[한계블록] {blocks} -> HP 조정으로 개선 시도")

        # 구체적 액션 제안
        actions = []
        if residual.has_autocorrelation:
            mixer = _get_block_type(config, "temporal_mixer")
            actions.append(f"잔차에 패턴 남음 -> {mixer}의 capacity 상향 또는 ConvMix 시도")
        if residual.has_heteroscedasticity:
            actions.append("이분산성 -> AsymmetricLoss 또는 구간별 가중치 권장")
        if abs(residual.residual_skew) > 1.5:
            if residual.residual_skew > 0:
                actions.append("과소예측 편향 -> under_weight 상향")
            else:
                actions.append("과대예측 편향 -> Positivity constraint 확인")
        if residual.worst_segment and residual.overall_mae > 0:
            ratio = residual.worst_segment_mae / residual.overall_mae
            if ratio > 2.0:
                actions.append(
                    f"{residual.worst_segment} 구간 MAE {ratio:.1f}x -> "
                    "해당 구간 특성에 맞는 블록 필요"
                )

        if actions:
            parts.append("[액션] " + "; ".join(actions))

        return "\n".join(parts) if parts else "진단 이상 없음"


# ══════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════

def _get_block_type(config: dict, slot: str) -> str | None:
    """config에서 특정 슬롯의 블록 타입을 추출."""
    val = config.get(slot)
    if val is None:
        return None
    if isinstance(val, dict):
        return val.get("type")
    return str(val)


def _safe_skew(arr: np.ndarray) -> float:
    if len(arr) < 3:
        return 0.0
    m = arr.mean()
    s = arr.std()
    if s == 0:
        return 0.0
    return float(((arr - m) ** 3).mean() / s ** 3)


def _safe_kurtosis(arr: np.ndarray) -> float:
    if len(arr) < 4:
        return 0.0
    m = arr.mean()
    s = arr.std()
    if s == 0:
        return 0.0
    return float(((arr - m) ** 4).mean() / s ** 4 - 3.0)
