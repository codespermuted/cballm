"""Trainer Engine — build_model(config) → 학습 → 표준 결과 JSON.

LLM 코드 생성 없이, Architect의 JSON config를 받아 결정론적으로 학습한다.
결과는 Critic이 바로 파싱할 수 있는 표준 포맷으로 출력.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .builder import build_model


# ── 표준 결과 포맷 (Critic 입력 스펙) ──

@dataclass
class FoldResult:
    """단일 fold의 학습 결과."""
    fold: int
    train_size: int
    val_size: int
    metrics: dict  # {"MAE": ..., "MSE": ..., "RMSE": ...}
    train_metrics: dict
    train_val_gap: float  # train MAE - val MAE (음수면 과적합 아님)
    fit_time_sec: float
    best_epoch: int


@dataclass
class TrainResult:
    """전체 학습 결과 — Critic 입력용 표준 포맷."""
    best_model: str
    best_metric: dict       # 최종 test 메트릭 (refit 후, 원본 스케일)
    best_metric_norm: dict  # 최종 test 메트릭 (정규화 스케일, 벤치마크 비교용)
    fold_results: list[FoldResult]
    cv_mean: dict  # fold 평균 메트릭
    cv_std: dict   # fold 표준편차
    normal_metric: dict  # 정상 구간 메트릭 (test set)
    extreme_metric: dict  # 극단 구간 메트릭 (test set, 샘플 없으면 빈 dict)
    extreme_threshold: float
    extreme_n_samples: int  # extreme 구간 샘플 수
    config_used: dict
    total_time_sec: float

    def to_json(self) -> str:
        """Critic이 파싱할 표준 JSON."""
        d = asdict(self)
        # fold_results를 간결하게
        d["fold_results"] = [asdict(f) for f in self.fold_results]
        return json.dumps(d, ensure_ascii=False, indent=2, default=str)

    def to_critic_text(self) -> str:
        """Critic rule-based 파서가 읽을 수 있는 텍스트 포맷."""
        lines = [
            f"BEST_MODEL: {self.best_model}",
            f"METRICS: {json.dumps(self.best_metric)}",
            f"METRICS_NORM: {json.dumps(self.best_metric_norm)}",
        ]
        if self.normal_metric:
            lines.append(f"NORMAL_MAE: {self.normal_metric.get('MAE', 'N/A')}")
        if self.extreme_metric:
            lines.append(f"EXTREME_MAE: {self.extreme_metric.get('MAE', 'N/A')}")
        else:
            lines.append(f"EXTREME_MAE: N/A (extreme 구간 샘플 {self.extreme_n_samples}개)")
        lines.append(f"CV_MEAN: {json.dumps(self.cv_mean)}")
        lines.append(f"CV_STD: {json.dumps(self.cv_std)}")
        for f in self.fold_results:
            lines.append(f"FOLD_{f.fold}: MAE={f.metrics.get('MAE', 'N/A'):.4f}, gap={f.train_val_gap:.4f}")
        return "\n".join(lines)


# ── 시계열 데이터셋 ──

class TimeSeriesDataset(Dataset):
    """시계열 슬라이딩 윈도우 데이터셋."""

    def __init__(self, data: np.ndarray, target_idx: int,
                 seq_len: int, pred_len: int):
        self.data = torch.FloatTensor(data)
        self.target_idx = target_idx
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_samples = len(data) - seq_len - pred_len + 1

    def __len__(self):
        return max(0, self.n_samples)

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]              # (T, features)
        y = self.data[idx + self.seq_len: idx + self.seq_len + self.pred_len,
                      self.target_idx: self.target_idx + 1]  # (H, 1)
        return x, y


# ── Temporal Split ──

def temporal_split(n_total: int, seq_len: int, pred_len: int,
                   n_folds: int = 3, test_ratio: float = 0.15):
    """Time-Aware CV split 인덱스 생성.

    Returns:
        folds: list of (train_end, val_start, val_end)
        test_start: test set 시작 인덱스
    """
    # test set: 마지막 test_ratio
    test_size = int(n_total * test_ratio)
    test_start = n_total - test_size

    # 나머지를 n_folds로 sliding window
    trainval_size = test_start
    fold_val_size = trainval_size // (n_folds + 1)
    gap = pred_len  # leakage 방지

    folds = []
    for i in range(n_folds):
        val_end = trainval_size - i * fold_val_size
        val_start = val_end - fold_val_size
        train_end = val_start - gap

        if train_end < seq_len + pred_len:
            break  # 데이터 부족

        folds.append((train_end, val_start, val_end))

    folds.reverse()  # 시간순
    return folds, test_start


# ── Standard Benchmark Splits ──

STANDARD_SPLITS = {
    # ETT datasets: 12:4:4 = 8545:2881:2881 (hourly, ~17420 rows)
    "ETTh1": (8545, 11426, 14307),  # (train_end, val_end, test_end)
    "ETTh2": (8545, 11426, 14307),
    # ETT minute-level: same ratio
    "ETTm1": (34465, 46097, 57729),
    "ETTm2": (34465, 46097, 57729),
    # Weather: 36792 rows, 7:1:2
    "weather": (25750, 29394, 36792),
    # ECL (Electricity): 26304 rows, 7:1:2
    "ECL": (18412, 21044, 26304),
    # SMP hourly: 98232 rows, last 2 months test (~1441), val = 2 months before that
    "smp_hourly": (94909, 96791, 98232),
}


def standard_split(dataset_name: str, n_total: int) -> tuple[int, int, int] | None:
    """표준 벤치마크 split 반환. (train_end, val_end, test_end) or None."""
    if dataset_name in STANDARD_SPLITS:
        train_end, val_end, test_end = STANDARD_SPLITS[dataset_name]
        # 데이터 크기가 맞는지 검증
        if test_end <= n_total:
            return train_end, val_end, test_end
    return None


# ── 학습 엔진 ──

def train_model(
    data_path: str,
    target_col: str,
    model_config: dict,
    seq_len: int = 96,
    pred_len: int = 96,
    n_folds: int = 3,
    batch_size: int = 32,
    device: str = "auto",
    extreme_percentile: float = 95,
    benchmark_mode: bool = False,
) -> TrainResult:
    """End-to-end 학습 파이프라인.

    benchmark_mode=True: 표준 벤치마크 split(고정 train/val/test)으로 1회 학습+평가.
                         CV 없이 train → val early stopping → test 보고.
    benchmark_mode=False: Time-Aware CV + refit (기본, 실무용).
    """
    total_start = time.time()

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Capacity 기반 HP 자동 조정 (v2) ──
    # v2: temporal_mixer의 capacity로 결정, v1: backbone type으로 fallback
    mixer_type = model_config.get("temporal_mixer", {}).get("type", "") if isinstance(
        model_config.get("temporal_mixer"), dict) else model_config.get("temporal_mixer", "")
    backbone_type = model_config.get("backbone", {}).get("type", "") if isinstance(
        model_config.get("backbone"), dict) else ""

    # capacity 매핑
    capacity_map = {
        # v2 temporal mixers
        "LinearMix": "minimal", "MLPMix": "low", "GatedMLPMix": "medium",
        "PatchMLPMix": "medium", "AttentionMix": "high", "PatchAttentionMix": "high",
        "ConvMix": "medium", "RecurrentMix": "medium",
        # v1 backbones
        "Linear": "minimal", "PatchMLP": "medium",
    }
    capacity = capacity_map.get(mixer_type, capacity_map.get(backbone_type, "minimal"))

    hp_presets = {
        "minimal": {"lr": 1e-3,  "epochs": 50,  "patience": 10, "wd": 1e-4},
        "low":     {"lr": 5e-4,  "epochs": 100, "patience": 15, "wd": 1e-5},
        "medium":  {"lr": 1e-4,  "epochs": 100, "patience": 15, "wd": 1e-5},
        "high":    {"lr": 1e-4,  "epochs": 200, "patience": 20, "wd": 1e-5},
    }
    preset = hp_presets.get(capacity, hp_presets["minimal"])
    lr = preset["lr"]
    epochs = preset["epochs"]
    patience = preset["patience"]
    weight_decay = preset["wd"]
    block_name = mixer_type or backbone_type or "Linear"
    print(f"HP preset ({block_name}, capacity={capacity}): lr={lr}, epochs={epochs}, patience={patience}, wd={weight_decay}")

    # ── 1. 데이터 로드 ──
    if data_path.endswith(".parquet"):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)

    # datetime 컬럼 제거 (있으면)
    datetime_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
    if datetime_cols:
        df = df.drop(columns=datetime_cols)

    # target 인덱스
    cols = df.columns.tolist()
    if target_col not in cols:
        raise ValueError(f"Target '{target_col}' not in columns: {cols}")
    target_idx = cols.index(target_col)

    # NaN 처리
    df = df.ffill().bfill()

    # ── 전처리 (Architect Decision Protocol 결과) ──
    preprocessing = model_config.pop("preprocessing", {})
    if preprocessing.get("log_transform") and df[target_col].min() > 0:
        print("  전처리: log transform 적용")
        df[target_col] = np.log1p(df[target_col])
    if preprocessing.get("differencing"):
        print("  전처리: 1차 차분 적용")
        df[target_col] = df[target_col].diff()
        df = df.iloc[1:]  # 첫 행 NaN 제거

    data_values = df.values.astype(np.float32)
    n_features = data_values.shape[1]
    n_total = len(data_values)

    print(f"데이터: {n_total}행, {n_features}피쳐, 타겟: {target_col} (idx={target_idx})")

    # ── 2. Split 결정 (정규화보다 먼저 — train 범위를 알아야 함) ──
    dataset_name = Path(data_path).stem
    std_split = standard_split(dataset_name, n_total) if benchmark_mode else None

    test_end = n_total  # 기본: 데이터 끝까지

    if std_split:
        train_end, val_end, test_end = std_split
        print(f"📏 Standard benchmark split: train[:{train_end}] val[{train_end}:{val_end}] test[{val_end}:{test_end}]")
        folds = [(train_end, train_end, val_end)]
        test_start = val_end
    else:
        if benchmark_mode:
            print(f"⚠️ '{dataset_name}'의 표준 split 없음, temporal CV로 fallback")
        folds, test_start = temporal_split(n_total, seq_len, pred_len, n_folds)
        train_end = folds[0][0] if folds else int(n_total * 0.7)
    print(f"CV folds: {len(folds)}, test_start: {test_start}")

    # ── 3. 정규화 — DatasetNorm은 항상 적용 (벤치마크 비교 기준) ──
    # RevIN/RobustScaler는 DatasetNorm 위에서 instance-level 정규화로 동작.
    # 논문 표준: DatasetNorm(train mean/std) + RevIN(instance) 모두 적용.
    train_data = data_values[:train_end]
    means = train_data.mean(axis=0)
    stds = train_data.std(axis=0) + 1e-8
    data_norm = (data_values - means) / stds

    normalizer_cfg = model_config.get("normalizer")
    norm_type = ""
    if normalizer_cfg:
        norm_type = normalizer_cfg if isinstance(normalizer_cfg, str) \
            else normalizer_cfg.get("type", "") if isinstance(normalizer_cfg, dict) else ""
    if norm_type:
        print(f"정규화: DatasetNorm(train[:{train_end}]) + {norm_type}(instance)")
    else:
        print(f"정규화: DatasetNorm(train[:{train_end}])")

    # 극단값 threshold: train 분포 기준
    target_values_train = data_values[:train_end, target_idx]
    lower_thresh = np.percentile(target_values_train, 100 - extreme_percentile)
    upper_thresh = np.percentile(target_values_train, extreme_percentile)
    print(f"Extreme thresholds: lower={lower_thresh:.4f}, upper={upper_thresh:.4f}")

    # ── 3. CV 학습 ──
    fold_results = []
    best_fold_mae = float("inf")
    best_fold_state = None

    for fold_idx, (train_end, val_start, val_end) in enumerate(folds):
        print(f"\n--- Fold {fold_idx+1}/{len(folds)} ---")
        print(f"  Train: [0:{train_end}], Val: [{val_start}:{val_end}]")

        train_ds = TimeSeriesDataset(data_norm[:train_end], target_idx, seq_len, pred_len)
        val_ds = TimeSeriesDataset(data_norm[val_start:val_end], target_idx, seq_len, pred_len)

        if len(train_ds) == 0 or len(val_ds) == 0:
            print(f"  ⚠️ 데이터 부족, fold 스킵")
            continue

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        # 모델 빌드
        model, loss_fn = build_model(
            _deep_copy_config(model_config),
            seq_len=seq_len, pred_len=pred_len,
            n_features=n_features, target_idx=target_idx,
        )
        model = model.to(device)
        loss_fn = loss_fn.to(device)

        # Nadam 옵티마이저
        optimizer = torch.optim.NAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # 학습
        fold_start = time.time()
        best_val_mae = float("inf")
        best_epoch = 0
        wait = 0

        for epoch in range(epochs):
            # Train
            model.train()
            train_losses = []
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss.item())
            scheduler.step()

            # Validate
            model.eval()
            val_preds, val_targets = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    pred = model(xb)
                    val_preds.append(pred.cpu())
                    val_targets.append(yb.cpu())

            val_preds = torch.cat(val_preds)
            val_targets = torch.cat(val_targets)

            # 원본 스케일로 역변환
            val_preds_orig = val_preds * stds[target_idx] + means[target_idx]
            val_targets_orig = val_targets * stds[target_idx] + means[target_idx]

            val_mae = (val_preds_orig - val_targets_orig).abs().mean().item()

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        fit_time = time.time() - fold_start
        train_mae = np.mean(train_losses) * stds[target_idx]  # 대략적 역변환

        # 정규화 스케일 메트릭도 계산 (벤치마크 비교용)
        val_norm_mae = (val_preds - val_targets).abs().mean().item()
        val_norm_mse = ((val_preds - val_targets) ** 2).mean().item()

        val_mse = ((val_preds_orig - val_targets_orig) ** 2).mean().item()
        val_rmse = val_mse ** 0.5

        fold_result = FoldResult(
            fold=fold_idx + 1,
            train_size=len(train_ds),
            val_size=len(val_ds),
            metrics={"MAE": round(best_val_mae, 4), "MSE": round(val_mse, 4), "RMSE": round(val_rmse, 4),
                     "norm_MAE": round(val_norm_mae, 4), "norm_MSE": round(val_norm_mse, 4)},
            train_metrics={"MAE": round(float(train_mae), 4)},
            train_val_gap=round(float(train_mae) - best_val_mae, 4),
            fit_time_sec=round(fit_time, 1),
            best_epoch=best_epoch,
        )
        fold_results.append(fold_result)
        print(f"  MAE: {best_val_mae:.4f}, norm_MSE: {val_norm_mse:.4f} (epoch {best_epoch}, {fit_time:.1f}s)")

        if best_val_mae < best_fold_mae:
            best_fold_mae = best_val_mae
            best_fold_state = best_state

    # ── 4. CV 통계 ──
    if not fold_results:
        return TrainResult(
            best_model="FAILED", best_metric={}, best_metric_norm={},
            fold_results=[], cv_mean={}, cv_std={},
            normal_metric={}, extreme_metric={},
            extreme_threshold=float(upper_thresh), extreme_n_samples=0,
            config_used=model_config, total_time_sec=time.time() - total_start,
        )

    cv_maes = [f.metrics["MAE"] for f in fold_results]
    cv_mses = [f.metrics["MSE"] for f in fold_results]
    cv_mean = {"MAE": round(np.mean(cv_maes), 4), "MSE": round(np.mean(cv_mses), 4)}
    cv_std = {"MAE": round(np.std(cv_maes), 4), "MSE": round(np.std(cv_mses), 4)}

    print(f"\nCV Mean MAE: {cv_mean['MAE']:.4f} ± {cv_std['MAE']:.4f}")

    # ── 5. Refit (train+val → test) ──
    print(f"\n--- Refit (train+val → test) [test window: {test_start}:{test_end}] ---")
    refit_data = data_norm[:test_start]
    test_data = data_norm[test_start:test_end]

    refit_ds = TimeSeriesDataset(refit_data, target_idx, seq_len, pred_len)
    test_ds = TimeSeriesDataset(test_data, target_idx, seq_len, pred_len)

    if len(test_ds) > 0:
        refit_loader = DataLoader(refit_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        # 새 모델로 refit
        model, loss_fn = build_model(
            _deep_copy_config(model_config),
            seq_len=seq_len, pred_len=pred_len,
            n_features=n_features, target_idx=target_idx,
        )
        model = model.to(device)
        loss_fn = loss_fn.to(device)
        optimizer = torch.optim.NAdam(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_epoch_avg = int(np.mean([f.best_epoch for f in fold_results]))
        refit_epochs = max(best_epoch_avg + 5, 20)  # CV best epoch + 여유

        model.train()
        for epoch in range(refit_epochs):
            for xb, yb in refit_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        # Test 평가
        model.eval()
        test_preds, test_targets = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                test_preds.append(pred.cpu())
                test_targets.append(yb.cpu())

        test_preds = torch.cat(test_preds)
        test_targets = torch.cat(test_targets)
        test_preds_orig = test_preds * stds[target_idx] + means[target_idx]
        test_targets_orig = test_targets * stds[target_idx] + means[target_idx]

        # 원본 스케일 메트릭
        test_mae = (test_preds_orig - test_targets_orig).abs().mean().item()
        test_mse = ((test_preds_orig - test_targets_orig) ** 2).mean().item()

        # 정규화 스케일 메트릭 (벤치마크 비교용)
        test_norm_mae = (test_preds - test_targets).abs().mean().item()
        test_norm_mse = ((test_preds - test_targets) ** 2).mean().item()

        refit_metric = {
            "MAE": round(test_mae, 4), "MSE": round(test_mse, 4), "RMSE": round(test_mse**0.5, 4),
            "norm_MAE": round(test_norm_mae, 4), "norm_MSE": round(test_norm_mse, 4),
        }
        refit_metric_norm = {"MAE": round(test_norm_mae, 4), "MSE": round(test_norm_mse, 4)}

        # Normal/Extreme on test — 상위/하위 percentile 기준
        extreme_mask_t = _make_extreme_mask(test_targets_orig, lower_thresh, upper_thresh)
        normal_mask_t = ~extreme_mask_t
        extreme_n = int(extreme_mask_t.sum().item())

        test_normal_mae = float("nan")
        test_extreme_mae = float("nan")
        test_normal_metric = {}
        test_extreme_metric = {}

        if normal_mask_t.any():
            test_normal_mae = (test_preds_orig[normal_mask_t] - test_targets_orig[normal_mask_t]).abs().mean().item()
            test_normal_metric = {"MAE": round(test_normal_mae, 4)}

        if extreme_mask_t.any():
            test_extreme_mae = (test_preds_orig[extreme_mask_t] - test_targets_orig[extreme_mask_t]).abs().mean().item()
            test_extreme_metric = {"MAE": round(test_extreme_mae, 4)}

        print(f"  Test MAE: {test_mae:.4f}, norm_MSE: {test_norm_mse:.4f} (벤치마크 기준)")
        print(f"  Normal MAE: {test_normal_mae:.4f}, Extreme MAE: {test_extreme_mae:.4f} ({extreme_n} samples)")
    else:
        refit_metric = {"MAE": 0, "MSE": 0}
        refit_metric_norm = {"MAE": 0, "MSE": 0}
        test_normal_metric = {}
        test_extreme_metric = {}
        extreme_n = 0

    total_time = time.time() - total_start

    # 모델 이름: config에서 backbone type
    model_name = model_config.get("backbone", {}).get("type", "Unknown")
    regime = model_config.get("regime")
    if regime:
        model_name = f"{regime.get('type', 'Regime')}({model_name})"

    return TrainResult(
        best_model=model_name,
        best_metric=refit_metric,
        best_metric_norm=refit_metric_norm,
        fold_results=fold_results,
        cv_mean=cv_mean,
        cv_std=cv_std,
        normal_metric=test_normal_metric,
        extreme_metric=test_extreme_metric,
        extreme_threshold=round(float(upper_thresh), 4),
        extreme_n_samples=extreme_n,
        config_used=model_config,
        total_time_sec=round(total_time, 1),
    )


def _make_extreme_mask(targets: torch.Tensor, lower: float,
                       upper: float) -> torch.Tensor:
    """극단 구간 마스크. 샘플 단위 (B,) 반환.

    targets: (B, H, 1) 원본 스케일 타겟.
    한 샘플의 H개 예측 중 하나라도 극단이면 극단 샘플로 분류.
    """
    # (B, H, 1) → 각 스텝이 극단인지
    extreme_steps = (targets < lower) | (targets > upper)  # (B, H, 1)
    # 샘플 단위: H개 중 하나라도 극단이면 True
    extreme_samples = extreme_steps.squeeze(-1).any(dim=-1)  # (B,)
    return extreme_samples


def _deep_copy_config(config: dict) -> dict:
    """config dict를 깊은 복사 (build_model이 pop으로 소비하므로)."""
    import copy
    return copy.deepcopy(config)
