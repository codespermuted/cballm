"""KG Engine — YAML 기반 블록 카탈로그 쿼리 엔진.

블록 호환성 검증, 조건 매칭, 추천 규칙 실행을 담당한다.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


_DIR = Path(__file__).parent


def _load(name: str) -> dict:
    with open(_DIR / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_catalog() -> dict:
    return _load("block_catalog.yaml")


def load_compatibility() -> dict:
    return _load("compatibility.yaml")


def load_recommendations() -> dict:
    return _load("recommendations.yaml")


# ── 호환성 검증 ──────────────────────────────────────────────


def get_compatible_mixers(encoder: str, catalog: dict | None = None,
                          compat: dict | None = None) -> list[str]:
    """encoder에 대해 호환되는 temporal_mixer 목록 반환."""
    if compat is None:
        compat = load_compatibility()
    for rule in compat.get("shape_rules", []):
        enc = rule["encoder"]
        if isinstance(enc, list):
            if encoder in enc:
                return rule["compatible_mixers"]
        elif enc == encoder:
            return rule["compatible_mixers"]
    # fallback — 모든 non-patch mixer
    if catalog is None:
        catalog = load_catalog()
    return [m for m in catalog.get("temporal_mixer", {})
            if not m.startswith("Patch")]


def check_compatibility(config: dict) -> list[dict]:
    """config의 호환성을 검사하고, 위반 사항 목록을 반환한다.

    Returns:
        list of {"level": "error"|"warn", "message": str}
    """
    compat = load_compatibility()
    issues: list[dict] = []

    encoder = config.get("encoder", "LinearProjection")
    mixer = config.get("temporal_mixer", "LinearMix")
    normalizer = config.get("normalizer")
    channel_mixer = config.get("channel_mixer")

    # shape 호환성
    compatible = get_compatible_mixers(encoder, compat=compat)
    if mixer not in compatible:
        issues.append({
            "level": "error",
            "message": f"{encoder}와 {mixer}는 shape 비호환. "
                       f"호환 mixer: {compatible}"
        })

    # normalizer 충돌
    if normalizer in ("RevIN", "RobustScaler"):
        if not config.get("skip_dataset_norm", True):
            issues.append({
                "level": "warn",
                "message": f"{normalizer} 사용 시 skip_dataset_norm=true 필요"
            })

    # capacity 경고
    n_rows = config.get("n_rows", 0)
    for rule in compat.get("capacity_rules", []):
        if mixer in rule["block"] or (channel_mixer and channel_mixer in rule["block"]):
            cond = rule["condition"]
            if "< 10000" in cond and n_rows < 10000:
                issues.append({"level": "warn", "message": rule["message"]})
            elif "< 5000" in cond and n_rows < 5000:
                issues.append({"level": "warn", "message": rule["message"]})

    return issues


# ── 블록 필터링 ──────────────────────────────────────────────


def filter_blocks_by_conditions(slot: str, profile: dict,
                                catalog: dict | None = None) -> list[str]:
    """데이터 프로파일 기반으로 특정 slot에서 유효한 블록만 필터링.

    Args:
        slot: "encoder", "temporal_mixer", "channel_mixer", "loss" 등
        profile: Scout의 데이터 프로파일 dict
    """
    if catalog is None:
        catalog = load_catalog()
    slot_blocks = catalog.get(slot, {})
    valid = []

    n_rows = profile.get("n_rows", 0)
    n_features = profile.get("n_features", 1)
    seq_len = profile.get("seq_len", 96)
    is_stationary = profile.get("is_stationary", False)
    has_strong_seasonality = profile.get("has_strong_seasonality", False)
    outlier_ratio = profile.get("outlier_ratio", 0.0)
    high_cross_corr_pairs = profile.get("high_cross_corr_pairs", 0)
    target_skew = profile.get("target_skew", 0.0)
    target_min = profile.get("target_min", 0.0)
    extreme_ratio = profile.get("extreme_ratio", 1.0)

    for name, spec in slot_blocks.items():
        if name == "None":
            valid.append(name)
            continue

        conditions = spec.get("use_conditions", [])
        min_data = spec.get("min_data", 0)

        if n_rows < min_data:
            continue

        passes = True
        for cond in conditions:
            cond_str = str(cond)
            if "is_stationary == false" in cond_str and is_stationary:
                passes = False
            elif "has_strong_seasonality == true" in cond_str and not has_strong_seasonality:
                passes = False
            elif "seq_len >= 96" in cond_str and seq_len < 96:
                passes = False
            elif "n_rows >= " in cond_str:
                threshold = int(cond_str.split(">=")[1].strip())
                if n_rows < threshold:
                    passes = False
            elif "n_features > " in cond_str:
                threshold = int(cond_str.split(">")[1].strip())
                if n_features <= threshold:
                    passes = False
            elif "outlier_ratio > " in cond_str:
                threshold = float(cond_str.split(">")[1].strip())
                if outlier_ratio <= threshold:
                    passes = False
            elif "high_cross_corr_pairs >= " in cond_str:
                threshold = int(cond_str.split(">=")[1].strip())
                if high_cross_corr_pairs < threshold:
                    passes = False
            elif "target_skew > " in cond_str:
                threshold = float(cond_str.split(">")[1].strip())
                if target_skew <= threshold:
                    passes = False
            elif "extreme_ratio > " in cond_str:
                threshold = float(cond_str.split(">")[1].strip())
                if extreme_ratio <= threshold:
                    passes = False
            elif "default" in cond_str:
                pass  # always included

        if passes:
            valid.append(name)

    return valid


# ── 추천 생성 ────────────────────────────────────────────────


def get_recommendations(profile: dict) -> dict[str, list[dict]]:
    """프로파일 기반 슬롯별 추천 블록 목록 반환."""
    recs = load_recommendations()
    result: dict[str, list[dict]] = {
        "encoder": [],
        "temporal_mixer": [],
        "channel_mixer": [],
        "loss": [],
        "constraint": [],
    }

    n_rows = profile.get("n_rows", 0)
    n_features = profile.get("n_features", 1)
    max_acf = profile.get("max_acf_at_known_periods", 0.0)
    seq_len = profile.get("seq_len", 96)
    high_cross_corr_pairs = profile.get("high_cross_corr_pairs", 0)
    target_skew = profile.get("target_skew", 0.0)
    target_min = profile.get("target_min", 0.0)
    extreme_ratio = profile.get("extreme_ratio", 1.0)
    can_be_negative = profile.get("can_be_negative", True)
    has_local_patterns = profile.get("has_local_patterns", False)
    complex_temporal = profile.get("complex_temporal_pattern", False)

    # encoder
    for r in recs.get("encoder_recommendations", []):
        if "max_acf" in r["condition"] and max_acf > 0.7:
            result["encoder"].append({"block": r["recommends"],
                                      "hint": r.get("param_hint", "")})
        elif "seq_len >= 96" in r["condition"] and seq_len >= 96:
            result["encoder"].append({"block": r["recommends"],
                                      "hint": r.get("param_hint", "")})

    # temporal_mixer
    for r in recs.get("temporal_mixer_recommendations", []):
        if r["condition"] == "always":
            result["temporal_mixer"].append({"block": r["recommends"],
                                             "priority": r.get("priority", "")})
        elif "n_rows >= 10000" in r["condition"] and n_rows >= 10000 and complex_temporal:
            result["temporal_mixer"].append({"block": r["recommends"],
                                             "requires": r.get("requires", "")})
        elif "has_local_patterns" in r["condition"] and has_local_patterns:
            result["temporal_mixer"].append({"block": r["recommends"]})

    # channel_mixer
    for r in recs.get("channel_mixer_recommendations", []):
        if "high_cross_corr_pairs >= 5" in r["condition"] and high_cross_corr_pairs >= 5 and n_features > 10:
            result["channel_mixer"].append({"block": r["recommends"]})
        elif "high_cross_corr_pairs >= 3" in r["condition"] and high_cross_corr_pairs >= 3 and n_features <= 10:
            result["channel_mixer"].append({"block": r["recommends"]})

    # loss
    for r in recs.get("loss_recommendations", []):
        if "extreme_ratio" in r["condition"] and extreme_ratio > 2.0:
            result["loss"].append({"block": r["recommends"],
                                   "hint": r.get("param_hint", "")})
        elif "target_skew" in r["condition"] and target_skew > 1.5:
            result["loss"].append({"block": r["recommends"]})

    # constraint
    for r in recs.get("constraint_recommendations", []):
        if not can_be_negative and target_min >= 0:
            result["constraint"].append({"block": r["recommends"]})

    return result


def detect_forecasting_setting(n_features: int, target_col: str | None,
                               all_cols: list[str] | None = None) -> str:
    """Forecasting setting 자동 감지: S, MS, M."""
    if n_features == 1:
        return "S"
    if target_col and all_cols and len(all_cols) > 1:
        return "MS"
    return "M"


def get_hp_preset(capacity: str) -> dict:
    """Capacity 기반 HP preset 반환."""
    presets = {
        "minimal": {"lr": 1e-3, "epochs": 50, "patience": 10, "weight_decay": 1e-4},
        "low": {"lr": 5e-4, "epochs": 100, "patience": 15, "weight_decay": 1e-5},
        "medium": {"lr": 1e-4, "epochs": 100, "patience": 15, "weight_decay": 1e-5},
        "high": {"lr": 1e-4, "epochs": 200, "patience": 20, "weight_decay": 1e-5},
    }
    return presets.get(capacity, presets["minimal"])


def compute_data_scale(n_rows: int, n_features: int, pred_len: int) -> dict:
    """데이터 특성 기반 아키텍처 규모 자동 결정.

    effective_size = n_rows * n_features / pred_len

    Returns:
        {"d_model": int, "n_layers": int, "n_heads": int, "scale": str}
    """
    # n_rows가 모델 규모의 주요 결정 요인, n_features/pred_len은 보정
    effective = n_rows / max(pred_len, 1) * min(n_features, 20)

    if effective < 200:
        return {"d_model": 32, "n_layers": 1, "n_heads": 2, "scale": "tiny"}
    elif effective < 2000:
        return {"d_model": 64, "n_layers": 2, "n_heads": 4, "scale": "small"}
    elif effective < 20000:
        return {"d_model": 128, "n_layers": 3, "n_heads": 8, "scale": "medium"}
    else:
        return {"d_model": 256, "n_layers": 6, "n_heads": 8, "scale": "large"}


def resolve_hp(recipe: dict, data_scale: dict) -> dict:
    """HP 결정 우선순위 통합.

    우선순위: constraints.min/max > recipe.recommended > data_scale > llm_answer
    이 함수는 recipe 기본값과 data_scale을 병합하여 recommended + min/max를 반환.
    Architect가 LLM 답변을 이 범위 안에서 clamp.

    Returns:
        {"d_model": {"recommended": int, "min": int, "max": int},
         "n_layers": {"recommended": int, "min": int, "max": int},
         "n_heads": {"recommended": int, "min": int, "max": int}}
    """
    constraints = recipe.get("hp_constraints", {})
    recipe_d = recipe.get("d_model", 64)

    # d_model: recipe 기본값과 data_scale 중 큰 쪽 사용
    data_d = data_scale.get("d_model", 64)
    recommended_d = max(recipe_d, data_d)
    d_min = constraints.get("d_model", {}).get("min", 16)
    d_max = constraints.get("d_model", {}).get("max", 256)
    recommended_d = max(d_min, min(d_max, recommended_d))

    # n_layers/n_heads: data_scale 기반, recipe에 명시 없으면 data_scale 사용
    data_layers = data_scale.get("n_layers", 2)
    data_heads = data_scale.get("n_heads", 4)

    return {
        "d_model": {"recommended": recommended_d, "min": d_min, "max": d_max},
        "n_layers": {"recommended": data_layers, "min": 1, "max": 6},
        "n_heads": {"recommended": data_heads, "min": 2, "max": 16},
    }


def get_block_capacity(block_name: str, catalog: dict | None = None) -> str:
    """블록의 capacity 반환."""
    if catalog is None:
        catalog = load_catalog()
    for slot in ("encoder", "temporal_mixer", "channel_mixer"):
        blocks = catalog.get(slot, {})
        if block_name in blocks:
            return blocks[block_name].get("capacity", "minimal")
    return "minimal"
