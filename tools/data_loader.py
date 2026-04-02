"""데이터 로더 — 다양한 포맷을 안전하게 읽고 검증한다. 절대 추측하지 않는다."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


SUPPORTED_FORMATS = {".csv", ".parquet", ".json", ".xlsx", ".xls", ".xml", ".feather", ".tsv"}


def load_data(path: str) -> pd.DataFrame:
    """파일을 로드하고 기본 검증을 수행한다. 실패 시 명확한 에러."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"파일 없음: {path}")

    ext = p.suffix.lower()
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(f"지원하지 않는 포맷: {ext}. 지원: {SUPPORTED_FORMATS}")

    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext == ".tsv":
        df = pd.read_csv(path, sep="\t")
    elif ext == ".parquet":
        df = pd.read_parquet(path)
    elif ext == ".json":
        df = pd.read_json(path)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif ext == ".xml":
        df = pd.read_xml(path)
    elif ext == ".feather":
        df = pd.read_feather(path)
    else:
        raise ValueError(f"구현되지 않은 포맷: {ext}")

    # 기본 검증
    if df.empty:
        raise ValueError(f"빈 데이터프레임: {path}")

    return df


def validate_data(df: pd.DataFrame, target_col: str, datetime_col: str = None) -> dict:
    """데이터 무결성 검증. 문제 발견 시 명확히 보고."""
    issues = []

    # 타겟 컬럼 존재 확인
    if target_col not in df.columns:
        available = ", ".join(df.columns[:20].tolist())
        raise ValueError(
            f"타겟 컬럼 '{target_col}'이 데이터에 없음.\n"
            f"사용 가능한 컬럼: {available}"
        )

    # datetime 컬럼 자동 감지
    if datetime_col is None:
        for col in df.columns:
            if df[col].dtype == "datetime64[ns]" or "date" in col.lower() or "time" in col.lower():
                datetime_col = col
                break

    if datetime_col and datetime_col in df.columns:
        df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
        nat_count = df[datetime_col].isna().sum()
        if nat_count > 0:
            issues.append(f"datetime 파싱 실패: {nat_count}건")

    # 결측 보고
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if len(missing_cols) > 0:
        for col, count in missing_cols.items():
            pct = count / len(df) * 100
            issues.append(f"결측: {col} = {count}건 ({pct:.1f}%)")

    # 타겟 결측
    target_missing = df[target_col].isna().sum()
    if target_missing > 0:
        issues.append(f"⚠️ 타겟({target_col}) 결측: {target_missing}건 — 반드시 처리 필요")

    # 중복 행
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        issues.append(f"중복 행: {dup_count}건")

    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "datetime_col": datetime_col,
        "target_col": target_col,
        "issues": issues,
        "target_stats": {
            "mean": float(df[target_col].mean()),
            "std": float(df[target_col].std()),
            "min": float(df[target_col].min()),
            "max": float(df[target_col].max()),
        },
    }
