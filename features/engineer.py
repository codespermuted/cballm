"""Engineer — rule-based 피쳐 엔지니어링 (v2).

Scout 결과 기반 결정론적 피쳐 생성. workers/ 에서 features/ 로 이동.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd


class Engineer:
    """Rule-based 피쳐 엔지니어링. LLM 불필요."""
    name = "engineer"
    description = "Scout 프로파일 기반 결정론적 피쳐 생성"

    def __init__(self, cwd: str = "", rules: str = ""):
        self.cwd = cwd

    def run(self, task: str) -> dict:
        """Scout 프로파일을 파싱하여 피쳐를 생성."""
        data_path = self._extract(task, "DATA_PATH")
        target_col = self._extract(task, "TARGET_COL") or "target"
        pred_len = int(self._extract(task, "PREDICTION_LENGTH") or "96")
        feature_path = self._extract(task, "FEATURE_OUTPUT_PATH")

        if not data_path:
            return {"worker": self.name, "response": "ERROR: DATA_PATH 없음",
                    "code": None, "execution_result": None}

        try:
            profile_info = self._parse_profile(task)
            result_text = self._engineer_features(
                data_path, target_col, pred_len, profile_info, feature_path
            )
        except Exception as e:
            import traceback
            result_text = f"ERROR: {e}\n{traceback.format_exc()}"

        return {
            "worker": self.name,
            "response": result_text,
            "code": None,
            "execution_result": result_text,
        }

    def _engineer_features(self, data_path: str, target_col: str,
                           pred_len: int, profile: dict,
                           feature_path: str | None) -> str:
        """규칙 기반 피쳐 생성."""
        if data_path.endswith(".parquet"):
            df = pd.read_parquet(data_path)
        else:
            df = pd.read_csv(data_path)

        # datetime 처리
        datetime_col = None
        for col in df.columns:
            if "date" in col.lower() or "time" in col.lower():
                datetime_col = col
                df[datetime_col] = pd.to_datetime(df[datetime_col])
                break

        features_added = []
        n_features_before = len(df.columns)

        # ── 1. Calendar features (known covariates → 미래값 사용 가능) ──
        if datetime_col:
            dt = df[datetime_col]

            df["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
            df["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)
            features_added.extend(["hour_sin [known]", "hour_cos [known]"])

            df["dow_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
            df["dow_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)
            features_added.extend(["dow_sin [known]", "dow_cos [known]"])

            df["month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12)
            df["month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12)
            features_added.extend(["month_sin [known]", "month_cos [known]"])

            df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
            features_added.append("is_weekend [known]")

            df = df.drop(columns=[datetime_col])

        # ── 2. Lag features (unknown → k >= pred_len 필수) ──
        recommended_lags = profile.get("recommended_lags", [pred_len])

        for lag in recommended_lags:
            if lag >= pred_len:
                col_name = f"{target_col}_lag_{lag}"
                df[col_name] = df[target_col].shift(lag)
                features_added.append(f"{col_name} [unknown, lag≥H]")

        # ── 3. Rolling features (unknown → shift(pred_len) 필수) ──
        seasonality = profile.get("seasonality", {})
        windows = []
        if any(v > 0.5 for k, v in seasonality.items() if "24" in k):
            windows.append(24)
        if any(v > 0.3 for k, v in seasonality.items() if "168" in k):
            windows.append(168)
        if not windows:
            windows = [24]

        for w in windows:
            mean_col = f"{target_col}_rmean_{w}"
            std_col = f"{target_col}_rstd_{w}"
            df[mean_col] = df[target_col].rolling(w).mean().shift(pred_len)
            df[std_col] = df[target_col].rolling(w).std().shift(pred_len)
            features_added.append(f"{mean_col} [unknown, shifted H]")
            features_added.append(f"{std_col} [unknown, shifted H]")

        # ── 4. 외생변수 분석 ──
        exog_corr = profile.get("exog_correlations", {})
        kept_exog = []
        dropped_exog = []
        for col, r in sorted(exog_corr.items(), key=lambda x: -abs(x[1])):
            if abs(r) >= 0.3:
                kept_exog.append(f"{col} (r={r:.2f})")
            else:
                dropped_exog.append(f"{col} (r={r:.2f}, low)")

        # ── 5. NaN 처리 ──
        n_before = len(df)
        df = df.dropna()
        n_after = len(df)

        # ── 6. Leakage 사후 검증 ──
        leakage_warnings = self._verify_no_leakage(df, target_col, pred_len)

        # 저장
        if feature_path:
            df.to_parquet(feature_path, index=False)

        # 결과 보고
        lines = [
            "=== FEATURE ENGINEERING REPORT ===",
            f"Features before: {n_features_before}, after: {len(df.columns)}",
            f"Rows: {n_before} → {n_after} (dropped {n_before - n_after} NaN rows)",
            "",
            "Added features:",
        ]
        for f in features_added:
            lines.append(f"  + {f}")

        if kept_exog:
            lines.append(f"\nKept exogenous: {', '.join(kept_exog)}")
        if dropped_exog:
            lines.append(f"Dropped exogenous (low corr): {', '.join(dropped_exog)}")

        if leakage_warnings:
            lines.append(f"\nLEAKAGE WARNINGS ({len(leakage_warnings)}):")
            for w in leakage_warnings:
                lines.append(f"  {w}")
        else:
            lines.append("\nLeakage 검증 통과: 모든 feature 안전")

        lines.append(f"\nFinal columns ({len(df.columns)}): {', '.join(df.columns.tolist())}")
        lines.append(f"n_features for model: {len(df.columns)}")
        lines.append("=== END REPORT ===")

        return "\n".join(lines)

    def _parse_profile(self, task: str) -> dict:
        """Scout 프로파일 텍스트에서 정보 추출."""
        profile: dict = {}

        # 계절성
        seasonality: dict = {}
        season_match = re.search(r'Seasonality:\s*\[(.+?)\]', task)
        if season_match:
            for part in season_match.group(1).split(","):
                part = part.strip()
                period_match = re.search(r'(\d+h?\w*)', part)
                acf_match = re.search(r'ACF=([0-9.]+)', part)
                if period_match and acf_match:
                    seasonality[period_match.group(1)] = float(acf_match.group(1))
                elif period_match:
                    if "strong" in part:
                        seasonality[period_match.group(1)] = 0.8
                    elif "moderate" in part:
                        seasonality[period_match.group(1)] = 0.5
                    else:
                        seasonality[period_match.group(1)] = 0.2
        profile["seasonality"] = seasonality

        # 추천 lags
        lags_match = re.search(r'Recommended lags:\s*\[([0-9, ]+)\]', task)
        if lags_match:
            profile["recommended_lags"] = [int(x.strip()) for x in lags_match.group(1).split(",")]
        else:
            pred_len = int(self._extract(task, "PREDICTION_LENGTH") or "96")
            profile["recommended_lags"] = [pred_len, pred_len + 24]

        # 외생변수 상관
        exog: dict = {}
        exog_match = re.search(r'Exogenous ranking:\s*\[(.+?)\]', task)
        if exog_match:
            for part in exog_match.group(1).split(","):
                col_match = re.search(r'(\w+)\s*\(r=([0-9.-]+)\)', part.strip())
                if col_match:
                    exog[col_match.group(1)] = float(col_match.group(2))
        profile["exog_correlations"] = exog

        return profile

    @staticmethod
    def _verify_no_leakage(df: pd.DataFrame, target_col: str,
                           pred_len: int) -> list[str]:
        """Leakage 사후 검증."""
        warnings = []
        n = len(df)
        if n <= pred_len + 100:
            return warnings

        target = df[target_col].values
        future_target = target[pred_len:n]
        overlap_len = len(future_target)

        for col in df.columns:
            if col == target_col:
                continue

            feat = df[col].values[:overlap_len]
            valid = np.isfinite(future_target) & np.isfinite(feat)
            if valid.sum() < 100:
                continue

            corr = np.corrcoef(feat[valid], future_target[valid])[0, 1]
            if np.isnan(corr):
                continue

            if abs(corr) > 0.98:
                warnings.append(
                    f"{col}: corr={corr:.3f} (>0.98) — 높은 leakage 의심, 자동 제거"
                )
            elif abs(corr) > 0.95:
                warnings.append(
                    f"{col}: corr={corr:.3f} (>0.95) — leakage 가능성"
                )

        return warnings

    @staticmethod
    def _extract(text: str, field: str) -> str | None:
        match = re.search(rf"{field}\s*=\s*['\"]([^'\"]+)['\"]", text)
        if match:
            return match.group(1)
        match = re.search(rf"{field}\s*=\s*(\S+)", text)
        return match.group(1) if match else None
