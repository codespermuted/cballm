"""KG Matcher — rule-based 프로파일 → KG 쿼리 → 후보 레시피/조합 생성.

Scout의 데이터 프로파일을 받아:
1. Level 0: topology 결정 (encoder-only)
2. Level 1: 필요한 slot 결정
3. use_conditions 자동 체크 → 블록 pre-filtering
4. compatibility_rules 검증 → 유효 조합만
5. 후보 레시피 3~5개 생성 + 각각의 근거
"""
from __future__ import annotations

import re
from typing import Any

from cballm.ontology.kg_engine import (
    load_catalog,
    load_compatibility,
    get_compatible_mixers,
    filter_blocks_by_conditions,
    get_recommendations,
    detect_forecasting_setting,
    get_hp_preset,
    get_block_capacity,
    check_compatibility,
)
from cballm.recipes.registry import find_recipes_for_profile, load_verified_recipes


class KGMatcher:
    """Rule-based KG 매칭. LLM 불필요."""

    name = "kg_matcher"
    description = "프로파일 → KG 쿼리 → 유효 조합 필터링 → 후보 레시피 생성"

    def __init__(self, cwd: str = "", rules: str = ""):
        self.cwd = cwd
        self.catalog = load_catalog()
        self.compat = load_compatibility()

    def run(self, task: str) -> dict:
        """Scout 프로파일을 받아 후보 레시피/조합을 생성."""
        try:
            profile = self._parse_profile(task)
            result = self._match(profile)
            result_text = self._format_result(result, profile)
        except Exception as e:
            import traceback
            result_text = f"ERROR: {e}\n{traceback.format_exc()}"

        return {
            "worker": self.name,
            "response": result_text,
            "code": None,
            "execution_result": result_text,
        }

    def _match(self, profile: dict) -> dict:
        """프로파일 기반 매칭 실행."""
        result: dict[str, Any] = {}

        # ── Level 0: Forecasting Setting ──
        n_features = profile.get("n_features", 1)
        target_col = profile.get("target_col")
        all_cols = profile.get("all_cols", [])
        setting = detect_forecasting_setting(n_features, target_col, all_cols)
        result["forecasting_setting"] = setting

        # output_dim 결정
        if setting == "M":
            result["output_dim"] = n_features
        else:
            result["output_dim"] = 1

        # ── Level 1: 유효 블록 필터링 ──
        valid_encoders = filter_blocks_by_conditions("encoder", profile, self.catalog)
        valid_temporal = filter_blocks_by_conditions("temporal_mixer", profile, self.catalog)
        valid_channel = filter_blocks_by_conditions("channel_mixer", profile, self.catalog)
        valid_loss = filter_blocks_by_conditions("loss", profile, self.catalog)

        # normalizer 결정
        is_stationary = profile.get("is_stationary", False)
        outlier_ratio = profile.get("outlier_ratio", 0.0)
        if outlier_ratio > 0.05:
            result["normalizer"] = "RobustScaler"
        elif not is_stationary:
            result["normalizer"] = "RevIN"
        else:
            result["normalizer"] = None  # DatasetNorm 자동 적용

        result["valid_blocks"] = {
            "encoder": valid_encoders,
            "temporal_mixer": valid_temporal,
            "channel_mixer": valid_channel,
            "loss": valid_loss,
        }

        # ── Level 1.5: 호환성 체크 + 유효 조합 ──
        valid_combinations = []
        for enc in valid_encoders:
            compatible = get_compatible_mixers(enc, self.catalog, self.compat)
            for mixer in valid_temporal:
                if mixer in compatible:
                    valid_combinations.append((enc, mixer))

        result["valid_combinations"] = valid_combinations

        # ── 추천 ──
        recommendations = get_recommendations(profile)
        result["recommendations"] = recommendations

        # ── 후보 레시피 ──
        recipes = find_recipes_for_profile(profile)
        result["candidate_recipes"] = recipes[:5]

        # ── 각 레시피에 HP preset 적용 ──
        for recipe in result["candidate_recipes"]:
            blocks = recipe.get("blocks", {})
            mixer_name = blocks.get("temporal_mixer", "LinearMix")
            capacity = get_block_capacity(mixer_name, self.catalog)
            recipe["capacity"] = capacity
            recipe["hp_preset"] = get_hp_preset(capacity)

        # ── 분포 추천 (v2.1) ──
        valid_dists, recommended_dist = self._recommend_distribution(profile)
        result["valid_distributions"] = valid_dists
        result["recommended_distribution"] = recommended_dist

        # ── Block-first: 슬롯별 추천 (v3) ──
        result["slot_recommendations"] = self._build_slot_recommendations(
            profile, result, valid_dists, recommended_dist,
        )

        # 레시피는 reference로만
        result["reference_recipes"] = result.pop("candidate_recipes", [])

        return result

    def _format_result(self, result: dict, profile: dict) -> str:
        """매칭 결과를 슬롯별 추천 형태로 포맷 (v3 block-first)."""
        lines = [
            "=== KG MATCHER REPORT (block-first) ===",
            "",
            f"FORECASTING_SETTING={result['forecasting_setting']}",
            f"OUTPUT_DIM={result['output_dim']}",
            "",
        ]

        # 복잡도
        slots = result.get("slot_recommendations", {})
        cx = slots.get("_complexity", {})
        lines.append(f"COMPLEXITY={{score={cx.get('score','?')}, level={cx.get('level','?')}}}")
        lines.append("")
        lines.append("--- SLOT RECOMMENDATIONS ---")
        for slot in ["normalizer", "decomposer", "encoder", "temporal_mixer",
                      "channel_mixer", "head", "loss", "constraint"]:
            rec = slots.get(slot, {})
            if not rec:
                continue
            recommended = rec.get("recommended", "?")
            options = rec.get("options", [])
            confidence = rec.get("confidence", "?")
            reason = rec.get("reason", "")
            lines.append(f"SLOT_{slot.upper()}={{recommended={recommended}, "
                         f"options={options}, confidence={confidence}, "
                         f"reason=\"{reason}\"}}")

        # HP scale
        scale = slots.get("_data_scale", {})
        if scale:
            lines.append(f"\nDATA_SCALE={{d_model={scale.get('d_model',64)}, "
                         f"n_layers={scale.get('n_layers',2)}, "
                         f"n_heads={scale.get('n_heads',4)}, "
                         f"scale={scale.get('scale','?')}}}")

        # 분포 추천
        lines.append(f"VALID_DISTRIBUTIONS={result.get('valid_distributions', ['gaussian'])}")
        lines.append(f"RECOMMENDED_DISTRIBUTION={result.get('recommended_distribution', 'gaussian')}")

        # 참고 레시피 (reference only)
        refs = result.get("reference_recipes", [])
        if refs:
            ref_names = [r.get("name", "?") for r in refs[:3]]
            lines.append(f"\nREFERENCE_RECIPES={ref_names} (참고용, 최종 조합은 슬롯별 결정)")

        # 호환성
        combos = result.get("valid_combinations", [])
        if combos:
            lines.append(f"VALID_COMBOS={combos[:5]}")

        lines.append("\n=== END KG MATCHER REPORT ===")
        return "\n".join(lines)

    def _parse_profile(self, task: str) -> dict:
        """Scout 프로파일 텍스트에서 구조화된 프로파일 추출."""
        profile: dict[str, Any] = {}

        # 기본 정보
        profile["n_rows"] = self._extract_int(task, r"Rows:\s*(\d+)") or \
                            self._extract_int(task, r"Shape:\s*\((\d+),") or 0
        profile["n_features"] = self._extract_int(task, r"Columns:\s*(\d+)") or \
                                self._extract_int(task, r"Shape:\s*\(\d+,\s*(\d+)\)") or 1
        profile["seq_len"] = self._extract_int(task, r"seq_len\s*=\s*(\d+)") or 96

        # 컬럼 목록 추출 (Columns: A, B, C 형태)
        cols_match = re.search(r'Columns:\s*(.+)', task)
        if cols_match:
            profile["all_cols"] = [c.strip() for c in cols_match.group(1).split(",")]
        else:
            profile["all_cols"] = []

        # target 정보
        profile["target_col"] = self._extract_str(task, "TARGET_COL")
        profile["target_min"] = self._extract_float(task, r"Min:\s*([0-9.-]+)") or 0.0
        profile["target_skew"] = self._extract_float(task, r"Skew:\s*([0-9.-]+)") or 0.0

        # stationarity
        if "stationary" in task.lower():
            profile["is_stationary"] = "stationary: yes" in task.lower() or \
                                        "non-stationary" not in task.lower()
        else:
            profile["is_stationary"] = False

        # seasonality
        profile["has_strong_seasonality"] = bool(
            re.search(r'ACF[>=\s]*0\.[7-9]', task) or "strong" in task.lower()
        )

        # max ACF
        acf_values = re.findall(r'ACF=([0-9.]+)', task)
        profile["max_acf_at_known_periods"] = max(
            (float(v) for v in acf_values), default=0.0
        )

        # outlier
        outlier_match = re.search(r'Outlier[^:]*:\s*([0-9.]+)%?', task, re.IGNORECASE)
        if outlier_match:
            val = float(outlier_match.group(1))
            profile["outlier_ratio"] = val / 100 if val > 1 else val
        else:
            profile["outlier_ratio"] = 0.0

        # cross correlation
        high_corr_match = re.search(r'high.corr.*?(\d+)\s*pairs?', task, re.IGNORECASE)
        if high_corr_match:
            profile["high_cross_corr_pairs"] = int(high_corr_match.group(1))
        else:
            profile["high_cross_corr_pairs"] = 0

        # extreme ratio
        extreme_match = re.search(r'extreme.ratio[:\s]*([0-9.]+)', task, re.IGNORECASE)
        profile["extreme_ratio"] = float(extreme_match.group(1)) if extreme_match else 1.0

        # 기타
        profile["can_be_negative"] = profile["target_min"] < 0
        profile["has_local_patterns"] = bool(
            re.search(r'(spike|change.point|sudden)', task, re.IGNORECASE)
        )
        profile["complex_temporal_pattern"] = bool(
            re.search(r'(complex|non-linear|nonlinear|multiple.regimes?)', task, re.IGNORECASE)
        )

        # 분포 통계량 (v2.1)
        profile["target_kurtosis"] = self._extract_float(task, r'kurtosis=([0-9.\-]+)') or 0.0
        profile["tail_index"] = self._extract_float(task, r'tail_index=([0-9.]+)') or 0.0
        profile["jarque_bera_pvalue"] = self._extract_float(task, r'jarque_bera_p=([0-9.]+)') or 1.0
        profile["segment_variance_ratio"] = self._extract_float(task, r'segment_variance_ratio=([0-9.]+)') or 1.0

        # 다변량 통계 (v2.1)
        profile["pairwise_corr_mean"] = self._extract_float(task, r'pairwise_corr_mean=([0-9.]+)') or 0.0
        profile["n_high_corr_pairs"] = self._extract_int(task, r'n_high_corr_pairs=(\d+)') or 0
        # filter_blocks_by_conditions에서 사용하는 키 이름과 동기화
        profile["high_cross_corr_pairs"] = profile["n_high_corr_pairs"]
        profile["granger_significant_count"] = self._extract_int(task, r'granger_significant=(\d+)') or 0

        return profile

    def _build_slot_recommendations(self, profile: dict, result: dict,
                                     valid_dists: list, rec_dist: str) -> dict:
        """슬롯별 추천 — complexity score 기반 (v3.1).

        simple data → DLinear 등가 조합 추천
        complex data → 커스텀 조합 추천
        """
        from cballm.ontology.kg_engine import compute_data_scale, get_compatible_mixers

        n_rows = profile.get("n_rows", 0)
        n_features = profile.get("n_features", 1)
        pred_len = profile.get("pred_len", 96) if profile.get("pred_len") else 96
        is_stat = profile.get("is_stationary", False)
        max_acf = profile.get("max_acf_at_known_periods", 0)
        outlier = profile.get("outlier_ratio", 0)
        seg_var = profile.get("segment_variance_ratio", 1.0)
        n_high_corr = profile.get("n_high_corr_pairs", 0)
        skew = abs(profile.get("target_skew", 0))
        can_neg = profile.get("can_be_negative", True)
        kurtosis = profile.get("target_kurtosis", 0)
        tail_idx = profile.get("tail_index", 0)
        regime_stable = profile.get("regime_stable", True)

        data_scale = compute_data_scale(n_rows, n_features, pred_len)
        valid_enc = result["valid_blocks"]["encoder"]
        valid_mix = result["valid_blocks"]["temporal_mixer"]
        valid_ch = result["valid_blocks"]["channel_mixer"]

        # ── Complexity Score (0~1) ──
        complexity = 0.0
        # 비정상
        if not is_stat:
            complexity += 0.05
        # 이분산성: seg_var 절대값이 높으면 regime과 무관하게 기여
        if seg_var > 10.0:
            complexity += 0.2
        elif seg_var > 5.0:
            complexity += 0.1
        # regime 불안정은 추가 기여
        if not regime_stable:
            complexity += 0.15
        # heavy tail (kurtosis 또는 tail_index)
        if kurtosis > 6.0 or tail_idx > 0.03:
            complexity += 0.15
        # 다변량 (많은 변수 + 상관 높음)
        if n_features > 10 and n_high_corr >= 10:
            complexity += 0.15
        elif n_features > 10:
            complexity += 0.05
        # 대형 데이터 (복잡한 모델을 학습할 데이터가 충분)
        if n_rows > 50000:
            complexity += 0.1
        # 다중 주기 (ACF가 두 주기 이상에서 높으면)
        # → 단순히 ACF > 0.7이면 올리지 않음 (단일 주기는 Linear으로 충분)
        if not regime_stable:
            complexity += 0.15

        complexity = min(1.0, complexity)

        # ── Complexity Level ──
        if complexity < 0.2:
            level = "simple"
        elif complexity < 0.5:
            level = "moderate"
        else:
            level = "complex"

        slots = {}
        slots["_complexity"] = {"score": round(complexity, 2), "level": level}

        # ── Q1: Normalizer (complexity 무관 — 데이터 특성만) ──
        if outlier > 0.05:
            slots["normalizer"] = {"recommended": "RobustScaler", "options": ["RobustScaler", "RevIN", "None"],
                                    "confidence": "high", "reason": f"outlier={outlier:.2f}"}
        elif not is_stat:
            slots["normalizer"] = {"recommended": "RevIN", "options": ["RevIN", "None"],
                                    "confidence": "high", "reason": "비정상"}
        else:
            # 정상이어도 RevIN 기본 적용 (해가 안 되고, 분포 shift 방어)
            slots["normalizer"] = {"recommended": "RevIN", "options": ["RevIN", "None"],
                                    "confidence": "medium", "reason": "정상이지만 RevIN 안전"}

        # ── Q2: Decomposer ──
        # simple: None. 비정상이어도 RevIN이 처리.
        # moderate+: trend가 강하고 변동성 높으면 MovingAvgDecomp
        if level == "simple":
            slots["decomposer"] = {"recommended": "None", "options": ["None", "MovingAvgDecomp"],
                                    "confidence": "high", "reason": "단순 데이터, 분해 불필요"}
        elif not is_stat and not regime_stable:
            slots["decomposer"] = {"recommended": "MovingAvgDecomp", "options": ["MovingAvgDecomp", "None"],
                                    "confidence": "high", "reason": "비정상+불안정regime"}
        else:
            slots["decomposer"] = {"recommended": "None", "options": ["None", "MovingAvgDecomp"],
                                    "confidence": "medium", "reason": "RevIN으로 충분"}

        # ── Q3: Encoder ──
        # simple: LinearProjection. ACF가 높아도 단순 주기는 Linear이 처리.
        # complex: ACF > 0.8 AND 다중주기 → Fourier. 대형+패턴 → Patch.
        if level == "simple":
            enc_rec = "LinearProjection"
            enc_reason = "단순 데이터, Linear 충분"
            enc_conf = "high"
        elif max_acf > 0.8 and not regime_stable and "FourierEmbedding" in valid_enc:
            enc_rec = "FourierEmbedding"
            enc_reason = f"비안정+강주기(ACF={max_acf:.2f})"
            enc_conf = "medium"
        elif n_rows >= 10000 and level == "complex" and "PatchEmbedding" in valid_enc:
            enc_rec = "PatchEmbedding"
            enc_reason = f"대규모+복잡(rows={n_rows})"
            enc_conf = "medium"
        else:
            enc_rec = "LinearProjection"
            enc_reason = "기본"
            enc_conf = "baseline"
        slots["encoder"] = {"recommended": enc_rec, "options": valid_enc,
                             "confidence": enc_conf, "reason": enc_reason}

        # ── Q4: Temporal Mixer ──
        compatible = get_compatible_mixers(enc_rec, self.catalog, self.compat)
        valid_compat = [m for m in valid_mix if m in compatible]
        if not valid_compat:
            valid_compat = ["LinearMix"]

        if level == "simple":
            mix_rec = "LinearMix"
            mix_reason = "단순 데이터, Linear baseline"
            mix_conf = "high"
        elif not regime_stable and seg_var > 5.0 and "ConvMix" in valid_compat:
            mix_rec = "ConvMix"
            mix_reason = f"불안정regime+변동성(var={seg_var:.1f})"
            mix_conf = "high"
        elif n_rows >= 10000 and enc_rec == "PatchEmbedding" and "PatchAttentionMix" in valid_compat:
            mix_rec = "PatchAttentionMix"
            mix_reason = "대규모+Patch"
            mix_conf = "medium"
        elif n_rows >= 30000 and "MambaMix" in valid_compat:
            mix_rec = "MambaMix"
            mix_reason = f"대규모({n_rows}행), SSM 효율적"
            mix_conf = "medium"
        elif level == "moderate" and n_rows >= 5000 and "MLPMix" in valid_compat:
            mix_rec = "MLPMix"
            mix_reason = "중간 복잡도, MLP 시도"
            mix_conf = "medium"
        else:
            mix_rec = "LinearMix"
            mix_reason = "기본"
            mix_conf = "baseline"
        slots["temporal_mixer"] = {"recommended": mix_rec, "options": valid_compat,
                                    "confidence": mix_conf, "reason": mix_reason}

        # ── Q5: Channel Mixer ──
        # simple/moderate: None. complex + 다변량 + 상관 높음 → Channel mixer
        if level == "complex" and n_features > 10 and n_high_corr >= 10 and "InvertedAttentionMix" in valid_ch:
            ch_rec = "InvertedAttentionMix"
            ch_reason = f"복잡+다변량({n_features})+높은상관({n_high_corr})"
            ch_conf = "medium"
        elif n_high_corr >= 5 and n_features > 5 and "FeatureMLPMix" in valid_ch:
            ch_rec = "FeatureMLPMix"
            ch_reason = f"변수상관({n_high_corr}쌍)"
            ch_conf = "medium"
        else:
            ch_rec = "None"
            ch_reason = "CI 모드 또는 상관 약함"
            ch_conf = "high"
        slots["channel_mixer"] = {"recommended": ch_rec, "options": valid_ch + ["None"],
                                   "confidence": ch_conf, "reason": ch_reason}

        # ── Q6: Head ──
        if tail_idx > 0.03 or (kurtosis > 5.0 and tail_idx > 0.02):
            slots["head"] = {"recommended": "DistributionalHead", "options": ["LinearHead", "DistributionalHead"],
                              "confidence": "medium", "reason": f"tail_idx={tail_idx:.3f}, kurt={kurtosis:.1f}"}
        else:
            slots["head"] = {"recommended": "LinearHead", "options": ["LinearHead", "DistributionalHead"],
                              "confidence": "high", "reason": "기본"}

        # ── Q7: Loss ──
        extreme_ratio = profile.get("extreme_ratio", 1.0)
        if seg_var > 5.0 or extreme_ratio > 1.5:
            loss_rec = "Asymmetric"
            loss_reason = f"seg_var={seg_var:.1f}, 극단구간 보호"
        elif skew > 2.0:
            loss_rec, loss_reason = "Huber", f"skew={skew:.2f}"
        else:
            loss_rec, loss_reason = "MAE", "기본"
        slots["loss"] = {"recommended": loss_rec,
                          "options": result["valid_blocks"]["loss"],
                          "confidence": "medium", "reason": loss_reason}

        # ── Q8: Constraint ──
        constraints = []
        if not can_neg and profile.get("target_min", 0) >= 0:
            constraints.append("Positivity")
        if level == "complex" and seg_var > 5.0:
            constraints.append("VolatilityGate")
        slots["constraint"] = {
            "recommended": constraints if constraints else ["None"],
            "options": ["None", "Positivity", "VolatilityGate"],
            "confidence": "medium",
            "reason": f"level={level}, seg_var={seg_var:.1f}",
        }

        slots["_data_scale"] = data_scale

        return slots

    @staticmethod
    def _recommend_distribution(profile: dict) -> tuple[list[str], str]:
        """프로파일 기반 분포 추천. rule-based."""
        kurtosis = profile.get("target_kurtosis", 0)
        tail_idx = profile.get("tail_index", 0)
        jb_p = profile.get("jarque_bera_pvalue", 1.0)
        skew = abs(profile.get("target_skew", 0))
        can_neg = profile.get("can_be_negative", True)
        n_changes = profile.get("n_regime_changes", 0) if not profile.get("regime_stable", True) else 0

        valid = ["gaussian"]  # 항상 포함
        recommended = "gaussian"

        # Student-t: heavy tail
        if kurtosis > 5 and tail_idx > 0.02:
            valid.append("student_t")
            recommended = "student_t"

        # Log-Normal: 양수 + right skew
        if not can_neg and skew > 0.5:
            valid.append("log_normal")
            if recommended == "gaussian":
                recommended = "log_normal"

        # Mixture: multi-modal
        if n_changes >= 2:
            valid.append("mixture_gaussian")
            if recommended == "gaussian" and n_changes >= 3:
                recommended = "mixture_gaussian"

        # 정규성 기각 못함 → gaussian 강화
        if jb_p > 0.05:
            recommended = "gaussian"

        return valid, recommended

    @staticmethod
    def _extract_int(text: str, pattern: str) -> int | None:
        m = re.search(pattern, text)
        return int(m.group(1)) if m else None

    @staticmethod
    def _extract_float(text: str, pattern: str) -> float | None:
        m = re.search(pattern, text)
        return float(m.group(1)) if m else None

    @staticmethod
    def _extract_str(text: str, field: str) -> str | None:
        m = re.search(rf"{field}\s*=\s*['\"]([^'\"]+)['\"]", text)
        if m:
            return m.group(1)
        m = re.search(rf"{field}\s*=\s*(\S+)", text)
        return m.group(1) if m else None
