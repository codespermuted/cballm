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

        return result

    def _format_result(self, result: dict, profile: dict) -> str:
        """매칭 결과를 텍스트 리포트로 포맷."""
        lines = [
            "=== KG MATCHER REPORT ===",
            "",
            f"Forecasting Setting: {result['forecasting_setting']}",
            f"Output Dim: {result['output_dim']}",
            f"Normalizer: {result.get('normalizer', 'DatasetNorm (default)')}",
            "",
        ]

        # 유효 블록
        lines.append("Valid Blocks:")
        for slot, blocks in result["valid_blocks"].items():
            lines.append(f"  {slot}: {blocks}")

        # 유효 조합
        lines.append(f"\nValid Encoder-Mixer Combinations ({len(result['valid_combinations'])}):")
        for enc, mixer in result["valid_combinations"]:
            lines.append(f"  {enc} → {mixer}")

        # 추천
        recs = result.get("recommendations", {})
        if any(recs.values()):
            lines.append("\nRecommendations:")
            for slot, items in recs.items():
                if items:
                    for item in items:
                        lines.append(f"  {slot}: {item}")

        # 후보 레시피
        lines.append(f"\n--- Candidate Recipes ({len(result['candidate_recipes'])}) ---")
        for i, recipe in enumerate(result["candidate_recipes"], 1):
            name = recipe.get("name", "Unknown")
            blocks = recipe.get("blocks", {})
            capacity = recipe.get("capacity", "?")
            hp = recipe.get("hp_preset", {})
            benchmark = recipe.get("benchmark", {})
            best_for = recipe.get("best_for", [])

            lines.append(f"\n[{i}] {name} (capacity={capacity})")
            lines.append(f"    Blocks: {' → '.join(str(v) for v in blocks.values())}")
            lines.append(f"    HP: lr={hp.get('lr')}, epochs={hp.get('epochs')}, patience={hp.get('patience')}")
            if benchmark:
                bm_str = ", ".join(f"{k}={v}" for k, v in benchmark.items())
                lines.append(f"    Benchmark: {bm_str}")
            if best_for:
                lines.append(f"    Best for: {', '.join(best_for)}")

        # Architect용 구조화 데이터
        lines.append("\n--- FOR ARCHITECT ---")
        lines.append(f"FORECASTING_SETTING={result['forecasting_setting']}")
        lines.append(f"OUTPUT_DIM={result['output_dim']}")
        lines.append(f"NORMALIZER={result.get('normalizer', 'null')}")
        lines.append(f"VALID_ENCODERS={result['valid_blocks']['encoder']}")
        lines.append(f"VALID_TEMPORAL_MIXERS={result['valid_blocks']['temporal_mixer']}")
        lines.append(f"VALID_CHANNEL_MIXERS={result['valid_blocks']['channel_mixer']}")
        lines.append(f"VALID_LOSSES={result['valid_blocks']['loss']}")

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

        return profile

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
