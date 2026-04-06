"""Builder — Architect/KG Matcher의 config를 받아 모델을 조립한다 (v2.1).

v2.1 파이프라인:
  Input → [Normalizer?] → [Decomposer?] →
    (decomposer 있으면) seasonal → Encoder → TemporalMixer → merge
                        trend   → Encoder → trend_mixer  ↗
    (decomposer 없으면) → [Encoder] → [TemporalMixer]
  → [ChannelMixer?] → [Head] → [Constraint?] → Output

예시 config (decomposer 미사용 — 기존 동작):
{
    "normalizer": {"type": "RevIN", "affine": true},
    "encoder": {"type": "LinearProjection"},
    "temporal_mixer": {"type": "LinearMix"},
    "head": {"type": "LinearHead", "output_dim": 1},
    "loss": {"type": "MAE"}
}

예시 config (decomposer 사용 — DLinear 분해):
{
    "normalizer": {"type": "RevIN"},
    "decomposer": {"type": "MovingAvgDecomp", "kernel_size": 25},
    "encoder": {"type": "LinearProjection"},
    "temporal_mixer": {"type": "MLPMix"},
    "trend_mixer": {"type": "LinearMix"},  // 없으면 LinearMix 자동
    "head": {"type": "LinearHead", "output_dim": 1},
    "loss": {"type": "MAE"}
}
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .normalizer import NORMALIZER_REGISTRY
from .encoder import ENCODER_REGISTRY
from .temporal_mixer import TEMPORAL_MIXER_REGISTRY, PATCH_MIXERS
from .channel_mixer import CHANNEL_MIXER_REGISTRY
from .head import HEAD_REGISTRY
from .constraint import CONSTRAINT_REGISTRY
from .loss import LOSS_REGISTRY
from .decomposition import DECOMPOSITION_REGISTRY

# v1 호환용
from .backbone import BACKBONE_REGISTRY
from .regime import REGIME_REGISTRY


class ForecastModel(nn.Module):
    """KG 온톨로지 기반 슬롯 조립 모델 (v2.1).

    decomposer가 없으면:
      Input → [Norm] → Enc → Mix → [ChMix] → Head → [Constr] → Output

    decomposer가 있으면:
      Input → [Norm] → Decomp → seasonal → Enc → Mix_s ──┐
                               → trend    → Enc → Mix_t ──┤ 합산
      → [ChMix] → Head → [Constr] → Output
    """

    def __init__(self, encoder: nn.Module,
                 temporal_mixer: nn.Module,
                 head: nn.Module,
                 normalizer: nn.Module | None = None,
                 decomposer: nn.Module | None = None,
                 trend_encoder: nn.Module | None = None,
                 trend_mixer: nn.Module | None = None,
                 channel_mixer: nn.Module | None = None,
                 constraints: list[nn.Module] | None = None,
                 target_idx: int = 0,
                 n_features: int = 1):
        super().__init__()
        self.normalizer = normalizer
        self.decomposer = decomposer
        self.encoder = encoder
        self.temporal_mixer = temporal_mixer
        self.trend_encoder = trend_encoder
        self.trend_mixer = trend_mixer
        self.channel_mixer = channel_mixer
        self.head = head
        self.constraints = nn.ModuleList(constraints or [])
        self.target_idx = target_idx
        self.n_features = n_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, C) → (B, H, output_dim)"""
        # Normalizer
        if self.normalizer is not None:
            x = self.normalizer(x)

        # Decomposition path
        if self.decomposer is not None:
            seasonal, trend = self.decomposer(x)

            # Seasonal path
            h_s = self.encoder(seasonal)
            h_s = self.temporal_mixer(h_s)

            # Trend path
            enc_t = self.trend_encoder if self.trend_encoder is not None else self.encoder
            mix_t = self.trend_mixer if self.trend_mixer is not None else self.temporal_mixer
            h_t = enc_t(trend)
            h_t = mix_t(h_t)

            # Merge: 합산
            h = h_s + h_t
        else:
            # Standard path (기존 동작)
            h = self.encoder(x)
            h = self.temporal_mixer(h)

        # ChannelMixer
        if self.channel_mixer is not None:
            h = self.channel_mixer(h)

        # Head
        pred = self.head(h)

        # DistributionalHead: 학습 시 dict 반환 → mean 추출
        dist_params = None
        if isinstance(pred, dict):
            dist_params = pred
            pred = pred["mean"] if "mean" in pred else pred.get("mu", h)

        # Constraints
        for c in self.constraints:
            pred = c(pred)

        # Normalizer reverse
        if self.normalizer is not None and hasattr(self.normalizer, "reverse"):
            if pred.shape[-1] < self.n_features:
                pred = self.normalizer.reverse(pred, target_idx=self.target_idx)
            else:
                pred = self.normalizer.reverse(pred)

        # DistributionalHead: reverse된 mean을 다시 dict에 넣어서 반환
        if dist_params is not None:
            if "mean" in dist_params:
                dist_params["mean"] = pred
            elif "mu" in dist_params:
                dist_params["mu"] = pred
            return dist_params

        return pred


def build_model(config: dict, seq_len: int, pred_len: int,
                n_features: int, d_model: int = 64,
                output_dim: int = 1,
                target_idx: int = 0) -> tuple[ForecastModel, nn.Module]:
    """JSON config → (model, loss_fn) 튜플 (v2.1).

    Args:
        config: 모델 설계 JSON (v2/v2.1 또는 v1 포맷)
        seq_len: 입력 시퀀스 길이
        pred_len: 예측 길이
        n_features: 입력 피쳐 수
        d_model: 내부 표현 차원
        output_dim: 출력 차원 (S/MS=1, M=C)
        target_idx: 타겟 변수의 인덱스 (RevIN reverse 시 사용)
    """
    # v1 포맷 감지 및 변환
    if "backbone" in config and "temporal_mixer" not in config:
        config = _convert_v1_to_v2(config)

    # ── Normalizer (optional) ──
    normalizer = None
    norm_cfg = config.get("normalizer")
    if norm_cfg and isinstance(norm_cfg, dict):
        norm_cfg = norm_cfg.copy()
        norm_type = norm_cfg.pop("type", "RevIN")
        if norm_type in NORMALIZER_REGISTRY:
            norm_cls = NORMALIZER_REGISTRY[norm_type]
            normalizer = norm_cls(n_features=n_features, **norm_cfg)
    elif norm_cfg and isinstance(norm_cfg, str) and norm_cfg in NORMALIZER_REGISTRY:
        normalizer = NORMALIZER_REGISTRY[norm_cfg](n_features=n_features)

    # ── Decomposer (optional, v2.1) ──
    decomposer = None
    decomp_cfg = config.get("decomposer")
    if decomp_cfg and isinstance(decomp_cfg, dict):
        decomp_cfg = decomp_cfg.copy()
        decomp_type = decomp_cfg.pop("type", "MovingAvgDecomp")
        if decomp_type in DECOMPOSITION_REGISTRY:
            decomp_cls = DECOMPOSITION_REGISTRY[decomp_type]
            decomposer = decomp_cls(**decomp_cfg)

    # ── Encoder ──
    enc_cfg = config.get("encoder", {"type": "LinearProjection"})
    if isinstance(enc_cfg, str):
        enc_cfg = {"type": enc_cfg}
    enc_cfg = enc_cfg.copy()
    enc_type = enc_cfg.pop("type", "LinearProjection")
    if "d_model" in enc_cfg:
        d_model = enc_cfg.pop("d_model")
    enc_cls = ENCODER_REGISTRY[enc_type]
    encoder = enc_cls(n_features=n_features, d_model=d_model, **enc_cfg)

    # Patch 인코더인 경우 n_patches 계산
    is_patch_encoder = getattr(encoder, "output_is_patch", False)
    n_patches = None
    if is_patch_encoder:
        patch_len = getattr(encoder, "patch_len", 16)
        stride = getattr(encoder, "stride", 8)
        n_patches = (seq_len - patch_len) // stride + 1

    # ── TemporalMixer ──
    mix_cfg = config.get("temporal_mixer", {"type": "LinearMix"})
    if isinstance(mix_cfg, str):
        mix_cfg = {"type": mix_cfg}
    mix_cfg = mix_cfg.copy()
    mix_type = mix_cfg.pop("type", "LinearMix")
    mix_cls = TEMPORAL_MIXER_REGISTRY[mix_type]
    if "d_model" in mix_cfg:
        mix_cfg.pop("d_model")
    if mix_type in PATCH_MIXERS and n_patches is not None:
        temporal_mixer = mix_cls(n_patches=n_patches, pred_len=pred_len,
                                 d_model=d_model, **mix_cfg)
    else:
        temporal_mixer = mix_cls(seq_len=seq_len, pred_len=pred_len,
                                 d_model=d_model, **mix_cfg)

    # ── Trend Mixer (optional, decomposer 사용 시) ──
    trend_encoder = None
    trend_mixer = None
    if decomposer is not None:
        # Trend encoder: 메인 encoder와 동일 아키텍처, 별도 파라미터
        trend_encoder = enc_cls(n_features=n_features, d_model=d_model)

        # Trend mixer: config에 "trend_mixer"가 있으면 사용, 없으면 LinearMix
        tmix_cfg = config.get("trend_mixer", {"type": "LinearMix"})
        if isinstance(tmix_cfg, str):
            tmix_cfg = {"type": tmix_cfg}
        tmix_cfg = tmix_cfg.copy()
        tmix_type = tmix_cfg.pop("type", "LinearMix")
        tmix_cfg.pop("d_model", None)
        tmix_cls = TEMPORAL_MIXER_REGISTRY[tmix_type]
        trend_mixer = tmix_cls(seq_len=seq_len, pred_len=pred_len,
                               d_model=d_model, **tmix_cfg)

    # ── ChannelMixer (optional) ──
    channel_mixer = None
    ch_cfg = config.get("channel_mixer")
    if ch_cfg and isinstance(ch_cfg, dict):
        ch_cfg = ch_cfg.copy()
        ch_type = ch_cfg.pop("type")
        ch_cls = CHANNEL_MIXER_REGISTRY[ch_type]
        ch_cfg.pop("d_model", None)
        channel_mixer = ch_cls(d_model=d_model, **ch_cfg)
    elif ch_cfg and isinstance(ch_cfg, str) and ch_cfg in CHANNEL_MIXER_REGISTRY:
        channel_mixer = CHANNEL_MIXER_REGISTRY[ch_cfg](d_model=d_model)

    # ── Head ──
    head_cfg = config.get("head", {"type": "LinearHead"})
    if isinstance(head_cfg, str):
        head_cfg = {"type": head_cfg}
    head_cfg = head_cfg.copy()
    head_type = head_cfg.pop("type", "LinearHead")
    head_cls = HEAD_REGISTRY[head_type]
    head_cfg.pop("d_model", None)
    head_cfg.pop("output_dim", None)
    head_kwargs: dict[str, Any] = {"d_model": d_model, "output_dim": output_dim}
    if head_type == "FlattenLinearHead":
        head_kwargs["pred_len"] = pred_len
    head_kwargs.update(head_cfg)
    head = head_cls(**head_kwargs)

    # ── Constraints ──
    constraints = []
    for c_cfg in config.get("constraint", []):
        c_cfg = c_cfg.copy()
        c_type = c_cfg.pop("type")
        c_cls = CONSTRAINT_REGISTRY[c_type]
        constraints.append(c_cls(**c_cfg))

    # ── Loss ──
    loss_cfg = config.get("loss", {"type": "MAE"})
    if isinstance(loss_cfg, str):
        loss_cfg = {"type": loss_cfg}
    loss_cfg = loss_cfg.copy()
    loss_type = loss_cfg.pop("type", "MAE")
    loss_cls = LOSS_REGISTRY[loss_type]
    loss_fn = loss_cls(**loss_cfg)

    model = ForecastModel(
        encoder=encoder,
        temporal_mixer=temporal_mixer,
        head=head,
        normalizer=normalizer,
        decomposer=decomposer,
        trend_encoder=trend_encoder,
        trend_mixer=trend_mixer,
        channel_mixer=channel_mixer,
        constraints=constraints,
        target_idx=target_idx,
        n_features=n_features,
    )
    return model, loss_fn


def _convert_v1_to_v2(config: dict) -> dict:
    """v1 config 포맷을 v2로 변환."""
    v2 = {}
    enc = config.get("encoder", {"type": "Linear"})
    if isinstance(enc, dict):
        enc_type = enc.get("type", "Linear")
        v1_to_v2_enc = {"Linear": "LinearProjection", "Fourier": "FourierEmbedding"}
        enc["type"] = v1_to_v2_enc.get(enc_type, enc_type)
    v2["encoder"] = enc

    bb = config.get("backbone", {"type": "Linear"})
    if isinstance(bb, dict):
        bb_type = bb.get("type", "Linear")
        v1_to_v2_bb = {"Linear": "LinearMix", "PatchMLP": "PatchMLPMix", "MLP": "MLPMix"}
        bb["type"] = v1_to_v2_bb.get(bb_type, bb_type)
    v2["temporal_mixer"] = bb

    v2["normalizer"] = config.get("normalizer")
    v2["head"] = config.get("head", {"type": "LinearHead"})
    v2["channel_mixer"] = config.get("channel_mixer")
    v2["constraint"] = config.get("constraint", [])
    v2["loss"] = config.get("loss", {"type": "MAE"})
    v2["regime"] = config.get("regime")

    for key in ("preprocessing", "input_design", "training"):
        if key in config:
            v2[key] = config[key]
    return v2


def list_available_blocks() -> dict[str, list[str]]:
    """사용 가능한 블록 목록."""
    return {
        "normalizer": list(NORMALIZER_REGISTRY.keys()),
        "encoder": list(ENCODER_REGISTRY.keys()),
        "decomposition": list(DECOMPOSITION_REGISTRY.keys()),
        "temporal_mixer": list(TEMPORAL_MIXER_REGISTRY.keys()),
        "channel_mixer": list(CHANNEL_MIXER_REGISTRY.keys()),
        "head": list(HEAD_REGISTRY.keys()),
        "constraint": list(CONSTRAINT_REGISTRY.keys()),
        "loss": list(LOSS_REGISTRY.keys()),
    }
