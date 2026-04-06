"""Scout — rule-based 데이터 프로파일링. LLM 없이 결정론적 분석."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass
class DataProfile:
    """구조화된 데이터 프로파일 — Architect 입력용."""
    # 기본 정보
    n_rows: int
    n_cols: int
    columns: list[str]
    target_col: str
    freq_hours: float | None  # 추정 frequency (hours)

    # 타겟 통계
    target_mean: float
    target_std: float
    target_min: float
    target_max: float
    target_skew: float
    target_can_be_negative: bool

    # 결측
    missing_pct: dict[str, float]  # col → 결측률
    total_missing_pct: float

    # 정상성
    is_stationary: bool
    adf_pvalue: float

    # 계절성 (ACF 기반)
    seasonality: dict[str, float]  # {"24h": acf_val, "168h": acf_val, ...}
    dominant_period: int | None  # 가장 강한 주기 (lag 수)

    # 외생변수 상관
    exog_correlations: dict[str, float]  # col → pearson r with target
    top_exog: list[str]  # 상관 높은 순

    # Regime
    regime_stable: bool
    n_regime_changes: int

    # 분포 특성 (v2.1)
    target_kurtosis: float
    tail_index: float            # |z| > 3인 비율
    jarque_bera_pvalue: float    # 정규성 검정 p-value
    segment_variance_ratio: float  # 구간별 분산 비 (max/min) — VolatilityGate 추천 기준

    # 다변량 통계 (v2.1)
    pairwise_corr_mean: float       # 변수 간 평균 |상관|
    pairwise_corr_max: float        # 최대 상관 쌍의 |r|
    n_high_corr_pairs: int          # |corr| > 0.7인 쌍 수
    granger_significant_count: int  # target에 Granger 유의한 변수 수

    # 추천
    recommended_lags: list[int]

    def to_architect_text(self) -> str:
        """Architect가 읽을 구조화된 프로파일 텍스트."""
        lines = [
            "=== DATA PROFILE ===",
            f"Shape: ({self.n_rows}, {self.n_cols}), Target: {self.target_col}",
            f"Frequency: {'~' + str(self.freq_hours) + 'h' if self.freq_hours else 'unknown'}",
            f"Columns: {', '.join(self.columns)}",
            "",
            f"Target stats: mean={self.target_mean:.2f}, std={self.target_std:.2f}, "
            f"min={self.target_min:.2f}, max={self.target_max:.2f}, skew={self.target_skew:.2f}",
            f"Can be negative: {self.target_can_be_negative}",
            f"Missing: {self.total_missing_pct:.1f}% overall",
        ]

        # 정상성
        status = "stationary" if self.is_stationary else "non-stationary (differencing needed)"
        lines.append(f"Stationarity: {status} (ADF p={self.adf_pvalue:.4f})")

        # 계절성
        season_parts = []
        for period, strength in self.seasonality.items():
            level = "strong" if strength > 0.6 else "moderate" if strength > 0.3 else "weak"
            season_parts.append(f"{period}: {level} (ACF={strength:.2f})")
        lines.append(f"Seasonality: [{', '.join(season_parts)}]")

        if self.dominant_period:
            lines.append(f"Dominant period: {self.dominant_period} steps")

        # 외생변수
        if self.exog_correlations:
            exog_parts = [f"{c} (r={v:.2f})" for c, v in
                         sorted(self.exog_correlations.items(), key=lambda x: -abs(x[1]))[:5]]
            lines.append(f"Exogenous ranking: [{', '.join(exog_parts)}]")

        # 다변량 통계
        lines.append(
            f"Multivariate: pairwise_corr_mean={self.pairwise_corr_mean:.3f}, "
            f"pairwise_corr_max={self.pairwise_corr_max:.3f}, "
            f"n_high_corr_pairs={self.n_high_corr_pairs}, "
            f"granger_significant={self.granger_significant_count}"
        )

        # Regime
        regime_status = "stable" if self.regime_stable else f"unstable ({self.n_regime_changes} changes)"
        lines.append(f"Regime: {regime_status}")

        # 분포 특성
        lines.append(
            f"Distribution: kurtosis={self.target_kurtosis:.2f}, "
            f"tail_index={self.tail_index:.4f}, "
            f"jarque_bera_p={self.jarque_bera_pvalue:.4f}, "
            f"segment_variance_ratio={self.segment_variance_ratio:.2f}"
        )
        if self.jarque_bera_pvalue > 0.05:
            lines.append("  -> 정규성 기각 못함 (gaussian 적합)")
        elif self.target_kurtosis > 5 and self.tail_index > 0.02:
            lines.append("  -> heavy tail 감지 (student_t 권장)")
        elif not self.target_can_be_negative and self.target_skew > 0.5:
            lines.append("  -> 양수 + right skew (log_normal 권장)")

        lines.append(f"Recommended lags: {self.recommended_lags}")
        lines.append("=== END PROFILE ===")

        return "\n".join(lines)


class Scout:
    """Rule-based 데이터 프로파일링. LLM 불필요."""
    name = "scout"
    description = "결정론적 데이터 프로파일링 + 통계 진단"

    def __init__(self, cwd: str = "", rules: str = ""):
        self.cwd = cwd

    def run(self, task: str) -> dict:
        """데이터를 로드하고 프로파일링을 실행."""
        import re

        data_path = self._extract(task, "DATA_PATH")
        target_col = self._extract(task, "TARGET_COL") or "target"
        pred_len = int(self._extract(task, "PREDICTION_LENGTH") or "96")

        if not data_path:
            return {"worker": self.name, "response": "ERROR: DATA_PATH 없음",
                    "code": None, "execution_result": None}

        try:
            profile = self._profile(data_path, target_col, pred_len)
            response = profile.to_architect_text()
        except Exception as e:
            import traceback
            response = f"ERROR: {e}\n{traceback.format_exc()}"

        return {
            "worker": self.name,
            "response": response,
            "code": None,
            "execution_result": response,
        }

    def _profile(self, data_path: str, target_col: str, pred_len: int) -> DataProfile:
        """데이터 프로파일링 실행."""
        df = pd.read_csv(data_path)

        # datetime 감지 + frequency 추정
        datetime_col = None
        freq_hours = None
        for col in df.columns:
            if "date" in col.lower() or "time" in col.lower():
                datetime_col = col
                break

        if datetime_col:
            df[datetime_col] = pd.to_datetime(df[datetime_col])
            if len(df) > 1:
                diffs = df[datetime_col].diff().dropna()
                median_diff = diffs.median()
                freq_hours = median_diff.total_seconds() / 3600
            df = df.drop(columns=[datetime_col])

        cols = df.columns.tolist()
        target = df[target_col].values.astype(float)

        # 결측
        missing = {c: round(df[c].isna().mean() * 100, 2) for c in cols if df[c].isna().any()}
        total_missing = round(df.isna().mean().mean() * 100, 2)

        # 타겟 통계
        target_clean = target[~np.isnan(target)]

        # 정상성 (ADF)
        is_stationary, adf_p = self._adf_test(target_clean)

        # 계절성 (ACF)
        seasonality, dominant = self._seasonality(target_clean, freq_hours)

        # 외생변수 상관
        exog_corr = {}
        for c in cols:
            if c != target_col and df[c].dtype in [np.float64, np.int64, float, int]:
                try:
                    r = df[target_col].corr(df[c])
                    if not np.isnan(r):
                        exog_corr[c] = round(r, 3)
                except Exception:
                    pass

        top_exog = sorted(exog_corr, key=lambda c: -abs(exog_corr[c]))[:5]

        # Regime detection (rolling std 변화)
        regime_stable, n_changes = self._regime_check(target_clean, freq_hours)

        # 분포 특성 (v2.1)
        kurtosis = round(float(pd.Series(target_clean).kurtosis()), 4)
        tail_index = self._tail_index(target_clean)
        jb_pvalue = self._jarque_bera(target_clean)
        seg_var_ratio = self._segment_variance_ratio(target_clean)

        # 다변량 통계 (v2.1)
        pw_mean, pw_max, n_high, granger_n = self._multivariate_stats(df, cols, target_col)

        # 추천 lags
        recommended_lags = self._recommend_lags(seasonality, pred_len)

        return DataProfile(
            n_rows=len(df), n_cols=len(cols), columns=cols,
            target_col=target_col, freq_hours=freq_hours,
            target_mean=round(float(np.nanmean(target)), 4),
            target_std=round(float(np.nanstd(target)), 4),
            target_min=round(float(np.nanmin(target)), 4),
            target_max=round(float(np.nanmax(target)), 4),
            target_skew=round(float(pd.Series(target_clean).skew()), 4),
            target_can_be_negative=bool(np.nanmin(target) < 0),
            missing_pct=missing, total_missing_pct=total_missing,
            is_stationary=is_stationary, adf_pvalue=round(adf_p, 6),
            seasonality=seasonality, dominant_period=dominant,
            exog_correlations=exog_corr, top_exog=top_exog,
            regime_stable=regime_stable, n_regime_changes=n_changes,
            target_kurtosis=kurtosis,
            tail_index=tail_index,
            jarque_bera_pvalue=jb_pvalue,
            segment_variance_ratio=seg_var_ratio,
            pairwise_corr_mean=pw_mean,
            pairwise_corr_max=pw_max,
            n_high_corr_pairs=n_high,
            granger_significant_count=granger_n,
            recommended_lags=recommended_lags,
        )

    @staticmethod
    def _adf_test(target: np.ndarray) -> tuple[bool, float]:
        try:
            from statsmodels.tsa.stattools import adfuller
            result = adfuller(target[:5000], autolag="AIC")  # 속도 위해 최대 5000개
            return bool(result[1] < 0.05), float(result[1])
        except ImportError:
            # statsmodels 없으면 간이 판정
            diff = np.diff(target)
            ratio = np.std(diff) / (np.std(target) + 1e-8)
            return ratio > 0.5, 0.05 if ratio > 0.5 else 0.5

    @staticmethod
    def _seasonality(target: np.ndarray, freq_hours: float | None) -> tuple[dict, int | None]:
        """ACF 기반 계절성 감지."""
        result = {}
        max_lag = min(len(target) // 3, 500)

        # ACF 직접 계산 (statsmodels 없어도 동작)
        n = len(target)
        mean = target.mean()
        var = ((target - mean) ** 2).sum()

        def acf_at_lag(k):
            if k >= n or var == 0:
                return 0
            return ((target[:n-k] - mean) * (target[k:] - mean)).sum() / var

        # 주요 주기 후보
        check_lags = {}
        if freq_hours and freq_hours > 0:
            daily = int(round(24 / freq_hours))
            weekly = int(round(168 / freq_hours))
            if daily > 0 and daily < max_lag:
                check_lags["24h"] = daily
            if weekly > 0 and weekly < max_lag:
                check_lags["168h"] = weekly
        else:
            for lag in [24, 48, 96, 168, 336]:
                if lag < max_lag:
                    check_lags[f"{lag}steps"] = lag

        dominant = None
        max_acf = 0
        for label, lag in check_lags.items():
            val = round(acf_at_lag(lag), 3)
            result[label] = val
            if abs(val) > max_acf:
                max_acf = abs(val)
                dominant = lag

        return result, dominant

    @staticmethod
    def _regime_check(target: np.ndarray, freq_hours: float | None) -> tuple[bool, int]:
        """Rolling mean/std의 큰 수준 변화로 regime 감지.

        단순 threshold crossing이 아닌, 장기 평균의 의미 있는 수준 변화만 카운트.
        """
        window = 168 if freq_hours and freq_hours <= 1 else min(100, len(target) // 10)
        if len(target) < window * 4:
            return True, 0

        ts = pd.Series(target)
        rolling_mean = ts.rolling(window).mean().dropna().values

        # 전체를 N개 구간으로 나누고, 구간 간 평균 차이가 전체 std 대비 큰 경우 regime change
        n_segments = min(8, len(rolling_mean) // window)
        if n_segments < 2:
            return True, 0

        segment_size = len(rolling_mean) // n_segments
        segment_means = [rolling_mean[i*segment_size:(i+1)*segment_size].mean()
                        for i in range(n_segments)]

        overall_std = np.std(target)
        changes = 0
        for i in range(1, len(segment_means)):
            # 전체 std의 1배 이상 수준 변화만 regime change로 인정
            # (계절적 변동은 이 이하)
            if abs(segment_means[i] - segment_means[i-1]) > overall_std:
                changes += 1

        return changes <= 2, changes

    @staticmethod
    def _recommend_lags(seasonality: dict, pred_len: int) -> list[int]:
        """계절성 기반 추천 lags (prediction_length 이상만)."""
        lags = set()
        for label, acf_val in seasonality.items():
            if acf_val > 0.2:  # 의미 있는 상관만
                if "24h" in label:
                    for mult in [1, 2, 7]:
                        lag = 24 * mult
                        if lag >= pred_len:
                            lags.add(lag)
                elif "168h" in label:
                    if 168 >= pred_len:
                        lags.add(168)

        if not lags:
            lags.add(pred_len)

        return sorted(lags)

    @staticmethod
    def _multivariate_stats(df: pd.DataFrame, cols: list[str],
                            target_col: str) -> tuple[float, float, int, int]:
        """다변량 통계 계산.

        Returns: (pairwise_corr_mean, pairwise_corr_max, n_high_corr_pairs, granger_count)
        """
        numeric_cols = [c for c in cols if df[c].dtype in [np.float64, np.int64, float, int]]
        n = len(numeric_cols)

        if n < 2:
            return 0.0, 0.0, 0, 0

        # pairwise correlation
        corr_matrix = df[numeric_cols].corr().abs()
        # 대각선 제외
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        upper = corr_matrix.where(mask)
        pw_values = upper.values[mask]
        pw_values = pw_values[~np.isnan(pw_values)]

        pw_mean = round(float(pw_values.mean()), 4) if len(pw_values) > 0 else 0.0
        pw_max = round(float(pw_values.max()), 4) if len(pw_values) > 0 else 0.0
        n_high = int((pw_values > 0.7).sum()) if len(pw_values) > 0 else 0

        # Granger causality (변수 20개 이하일 때만)
        granger_n = 0
        if n <= 20 and target_col in numeric_cols:
            try:
                from statsmodels.tsa.stattools import grangercausalitytests
                target_series = df[target_col].values
                for c in numeric_cols:
                    if c == target_col:
                        continue
                    try:
                        data = np.column_stack([target_series[:2000], df[c].values[:2000]])
                        # NaN 제거
                        valid = ~np.isnan(data).any(axis=1)
                        data = data[valid]
                        if len(data) < 50:
                            continue
                        result = grangercausalitytests(data, maxlag=4, verbose=False)
                        # 어느 lag에서든 p < 0.05이면 유의
                        for lag, test in result.items():
                            p = test[0]["ssr_ftest"][1]
                            if p < 0.05:
                                granger_n += 1
                                break
                    except Exception:
                        pass
            except ImportError:
                pass  # statsmodels 없으면 skip

        return pw_mean, pw_max, n_high, granger_n

    @staticmethod
    def _segment_variance_ratio(target: np.ndarray, n_segments: int = 4) -> float:
        """구간별 분산 비 (max/min). VolatilityGate 추천 기준."""
        n = len(target)
        if n < n_segments * 10:
            return 1.0
        seg_size = n // n_segments
        variances = []
        for i in range(n_segments):
            seg = target[i * seg_size:(i + 1) * seg_size]
            v = np.var(seg)
            if v > 0:
                variances.append(v)
        if len(variances) < 2 or min(variances) == 0:
            return 1.0
        return round(float(max(variances) / min(variances)), 2)

    @staticmethod
    def _tail_index(target: np.ndarray) -> float:
        """극단값 빈도: |z-score| > 3인 비율. heavy tail 지표."""
        std = np.std(target)
        if std == 0:
            return 0.0
        z = np.abs((target - np.mean(target)) / std)
        return round(float((z > 3).mean()), 4)

    @staticmethod
    def _jarque_bera(target: np.ndarray) -> float:
        """Jarque-Bera 정규성 검정. p > 0.05면 정규성 기각 못함."""
        try:
            from scipy.stats import jarque_bera
            _, p = jarque_bera(target[:5000])
            return round(float(p), 6)
        except ImportError:
            # scipy 없으면 간이 판정: skew와 kurtosis로 추정
            n = len(target)
            s = pd.Series(target)
            skew = s.skew()
            kurt = s.kurtosis()
            jb = n / 6 * (skew ** 2 + kurt ** 2 / 4)
            # chi-squared df=2 근사
            p_approx = max(0.0, 1.0 - jb / (jb + 6))
            return round(p_approx, 6)

    @staticmethod
    def _extract(text: str, field: str) -> str | None:
        import re
        match = re.search(rf"{field}\s*=\s*['\"]([^'\"]+)['\"]", text)
        if match:
            return match.group(1)
        match = re.search(rf"{field}\s*=\s*(\S+)", text)
        return match.group(1) if match else None
