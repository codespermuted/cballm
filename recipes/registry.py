"""Recipe Registry — 검증된/커스텀 레시피 로드, 검색, 등록, 성능 비교."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

_DIR = Path(__file__).parent
_VERIFIED_DIR = _DIR / "verified"
_CUSTOM_DIR = _DIR / "custom"


def load_verified_recipes() -> dict[str, dict]:
    """verified/ 디렉토리의 모든 레시피 로드."""
    recipes = {}
    for f in sorted(_VERIFIED_DIR.glob("*.yaml")):
        with open(f, encoding="utf-8") as fh:
            r = yaml.safe_load(fh)
            recipes[r["name"]] = r
    return recipes


def load_custom_recipes() -> dict[str, dict]:
    """custom/ 디렉토리의 모든 레시피 로드."""
    recipes = {}
    for f in sorted(_CUSTOM_DIR.glob("*.yaml")):
        with open(f, encoding="utf-8") as fh:
            r = yaml.safe_load(fh)
            recipes[r["name"]] = r
    return recipes


def load_all_recipes() -> dict[str, dict]:
    """모든 레시피 로드 (verified 우선)."""
    all_r = load_verified_recipes()
    all_r.update(load_custom_recipes())
    return all_r


def find_recipes_for_profile(profile: dict) -> list[dict]:
    """프로파일에 맞는 레시피 후보 반환 (적합도순 정렬).

    Args:
        profile: Scout의 데이터 프로파일 dict
            n_rows, n_features, has_strong_seasonality,
            high_cross_corr_pairs, target_min, ...
    """
    recipes = load_verified_recipes()
    scored: list[tuple[float, str, dict]] = []

    n_rows = profile.get("n_rows", 0)
    n_features = profile.get("n_features", 1)
    high_cross_corr = profile.get("high_cross_corr_pairs", 0)

    for name, recipe in recipes.items():
        score = 0.0
        best_for = recipe.get("best_for", [])

        # DLinear은 항상 baseline으로 포함
        if name == "DLinear":
            score += 10.0

        # 데이터 크기에 따른 매칭
        if n_rows < 5000:
            if "small_data" in best_for or "baseline" in best_for:
                score += 5.0
        elif n_rows >= 10000:
            if "large_data" in best_for:
                score += 5.0

        # feature 수에 따른 매칭
        if n_features > 10 and high_cross_corr >= 5:
            if "many_features" in best_for or "high_cross_correlation" in best_for:
                score += 8.0

        # local pattern
        if profile.get("has_local_patterns", False):
            if "local_patterns" in best_for:
                score += 3.0

        scored.append((score, name, recipe))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [{"name": s[1], "score": s[0], **s[2]} for s in scored]


def register_custom_recipe(recipe: dict) -> Path:
    """커스텀 레시피를 custom/ 디렉토리에 저장."""
    _CUSTOM_DIR.mkdir(parents=True, exist_ok=True)
    name = recipe["name"]
    path = _CUSTOM_DIR / f"{name}.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(recipe, f, allow_unicode=True, default_flow_style=False)
    return path


def compare_recipes(results: list[dict]) -> str:
    """여러 레시피 결과를 비교 테이블로 출력."""
    if not results:
        return "No results to compare."

    lines = ["Recipe Comparison:"]
    lines.append(f"{'Recipe':<20} {'MSE':>8} {'MAE':>8} {'Capacity':>10}")
    lines.append("-" * 50)

    for r in results:
        name = r.get("recipe_name", r.get("name", "?"))
        mse = r.get("test_mse", r.get("mse", "N/A"))
        mae = r.get("test_mae", r.get("mae", "N/A"))
        cap = r.get("capacity", "?")
        mse_s = f"{mse:.4f}" if isinstance(mse, float) else str(mse)
        mae_s = f"{mae:.4f}" if isinstance(mae, float) else str(mae)
        lines.append(f"{name:<20} {mse_s:>8} {mae_s:>8} {cap:>10}")

    return "\n".join(lines)
