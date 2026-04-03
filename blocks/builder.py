"""Builder — Architect의 JSON config를 받아 모델을 조립한다.

LLM이 코드를 생성할 필요 없이, JSON만으로 모델이 만들어진다.

예시 config:
{
    "encoder": {"type": "Fourier", "n_harmonics": 3},
    "backbone": {"type": "PatchMLP", "patch_len": 16, "hidden_dim": 256},
    "regime": {"type": "SoftGate", "n_regimes": 2},
    "constraint": [{"type": "Positivity"}],
    "loss": {"type": "MAE"}
}
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .encoder import ENCODER_REGISTRY
from .backbone import BACKBONE_REGISTRY
from .regime import REGIME_REGISTRY
from .constraint import CONSTRAINT_REGISTRY
from .loss import LOSS_REGISTRY


class ForecastModel(nn.Module):
    """슬롯 기반 시계열 예측 모델.

    Input → [Encoder] → [Backbone or RegimeGate(Backbones)] → [Constraints] → Output
    """

    def __init__(self, encoder: nn.Module, backbone: nn.Module,
                 constraints: list[nn.Module] | None = None):
        super().__init__()
        self.encoder = encoder
        self.backbone = backbone
        self.constraints = nn.ModuleList(constraints or [])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, raw_features) → (B, H, 1)"""
        h = self.encoder(x)       # (B, T, d_model)
        pred = self.backbone(h)   # (B, H, 1)
        for c in self.constraints:
            pred = c(pred)        # (B, H, 1)
        # 확장점: encoder에 reverse가 있으면 출력 역변환 (RevIN 등)
        if hasattr(self.encoder, 'reverse'):
            pred = self.encoder.reverse(pred)
        return pred


def build_model(config: dict, seq_len: int, pred_len: int,
                n_features: int, d_model: int = 64) -> tuple[ForecastModel, nn.Module]:
    """JSON config → (model, loss_fn) 튜플.

    Args:
        config: Architect가 생성한 모델 설계 JSON
        seq_len: 입력 시퀀스 길이
        pred_len: 예측 길이
        n_features: 입력 피쳐 수
        d_model: 내부 표현 차원

    Returns:
        (model, loss_fn) 튜플
    """
    # ── Encoder ──
    enc_cfg = config.get("encoder", {"type": "Linear"})
    enc_type = enc_cfg.pop("type", "Linear")
    enc_cls = ENCODER_REGISTRY[enc_type]

    # encoder-specific args
    enc_kwargs: dict[str, Any] = {"n_features": n_features, "d_model": d_model}
    enc_kwargs.update(enc_cfg)
    encoder = enc_cls(**enc_kwargs)

    # ── Backbone (or RegimeGate wrapping Backbones) ──
    bb_cfg = config.get("backbone", {"type": "MLP"})
    regime_cfg = config.get("regime", None)

    if regime_cfg:
        # Regime gate wrapping N identical backbones
        n_regimes = regime_cfg.get("n_regimes", 2)
        regime_type = regime_cfg.get("type", "SoftGate")

        backbones = []
        for _ in range(n_regimes):
            bb = _build_backbone(bb_cfg.copy(), seq_len, pred_len, d_model)
            backbones.append(bb)

        regime_cls = REGIME_REGISTRY[regime_type]
        backbone = regime_cls(backbones=backbones, seq_len=seq_len, d_model=d_model)
    else:
        backbone = _build_backbone(bb_cfg, seq_len, pred_len, d_model)

    # ── Constraints ──
    constraints = []
    for c_cfg in config.get("constraint", []):
        c_cfg = c_cfg.copy()
        c_type = c_cfg.pop("type")
        c_cls = CONSTRAINT_REGISTRY[c_type]
        constraints.append(c_cls(**c_cfg))

    # ── Loss ──
    loss_cfg = config.get("loss", {"type": "MAE"})
    loss_cfg = loss_cfg.copy()
    loss_type = loss_cfg.pop("type", "MAE")
    loss_cls = LOSS_REGISTRY[loss_type]
    loss_fn = loss_cls(**loss_cfg)

    model = ForecastModel(encoder, backbone, constraints)
    return model, loss_fn


def _build_backbone(cfg: dict, seq_len: int, pred_len: int, d_model: int):
    """단일 backbone 빌드."""
    cfg = cfg.copy()
    bb_type = cfg.pop("type", "MLP")
    bb_cls = BACKBONE_REGISTRY[bb_type]
    return bb_cls(seq_len=seq_len, pred_len=pred_len, d_model=d_model, **cfg)


def list_available_blocks() -> dict[str, list[str]]:
    """사용 가능한 블록 목록. Architect 프롬프트에 주입용."""
    return {
        "encoder": list(ENCODER_REGISTRY.keys()),
        "backbone": list(BACKBONE_REGISTRY.keys()),
        "regime": list(REGIME_REGISTRY.keys()),
        "constraint": list(CONSTRAINT_REGISTRY.keys()),
        "loss": list(LOSS_REGISTRY.keys()),
    }
