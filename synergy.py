"""SynergyChecker — 블록 조합 호환성 검증 + 자동 보정.

좋은 조합(synergy)은 유지하고, 나쁜 조합(antagonism)은 자동 보정.
CandidateGenerator 직후에 적용.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field


@dataclass
class SynergyResult:
    """검증 결과."""
    original: dict
    corrected: dict
    applied_rules: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def was_corrected(self) -> bool:
        return len(self.applied_rules) > 0


def _get_type(cfg, slot: str) -> str:
    """config에서 특정 슬롯의 블록 타입 추출."""
    val = cfg.get(slot)
    if val is None:
        return "None"
    if isinstance(val, dict):
        return val.get("type", "None")
    return str(val)


class SynergyChecker:
    """블록 조합 호환성 검증 + 자동 보정."""

    @staticmethod
    def validate(config: dict, profile: dict | None = None) -> SynergyResult:
        """config를 검증하고 필요시 보정."""
        corrected = copy.deepcopy(config)
        rules: list[str] = []
        warns: list[str] = []

        norm = _get_type(corrected, "normalizer")
        decomp = _get_type(corrected, "decomposer")
        enc = _get_type(corrected, "encoder")
        mix = _get_type(corrected, "temporal_mixer")
        head = _get_type(corrected, "head")
        loss = _get_type(corrected, "loss")
        ch_mix = _get_type(corrected, "channel_mixer")

        n_rows = (profile or {}).get("n_rows", 10000)
        is_stat = (profile or {}).get("is_stationary", False)

        # ── Antagonism (자동 보정) ──

        # RevIN + MovingAvgDecomp → RevIN 제거 (이중 mean/trend 제거)
        if norm == "RevIN" and decomp == "MovingAvgDecomp":
            corrected["normalizer"] = None
            rules.append("ANTAG-01: RevIN+MovingAvgDecomp → RevIN 제거 (이중 정규화)")

        # FourierEmb + FrequencyMix → FrequencyMix를 LinearMix로
        if enc == "FourierEmbedding" and mix == "FrequencyMix":
            corrected["temporal_mixer"] = {"type": "LinearMix"}
            rules.append("ANTAG-02: FourierEmb+FrequencyMix → LinearMix (frequency 중복)")

        # DistributionalHead + MAE/MSE → 해당 분포 NLL로
        if head == "DistributionalHead" and loss in ("MAE", "MSE"):
            dist = corrected.get("head", {}).get("distribution", "gaussian")
            nll_map = {
                "gaussian": "GaussianNLL", "student_t": "StudentTNLL",
                "log_normal": "LogNormalNLL", "mixture_gaussian": "MixtureGaussianNLL",
            }
            corrected["loss"] = {"type": nll_map.get(dist, "GaussianNLL")}
            rules.append(f"ANTAG-03: DistHead+{loss} → {nll_map.get(dist)} (분포 출력+point loss 모순)")

        # BatchInstanceNorm + RobustScaler → RobustScaler 제거
        if norm == "BatchInstanceNorm" and ch_mix == "RobustScaler":
            # 이건 channel_mixer가 아니라 normalizer 충돌
            pass
        # 둘 다 normalizer인 경우는 이미 registry에서 conflicts_with로 방지

        # PatchEmbedding + 비-Patch mixer → encoder를 LinearProjection으로
        from cballm.blocks.temporal_mixer import PATCH_MIXERS
        if enc == "PatchEmbedding" and mix not in PATCH_MIXERS:
            corrected["encoder"] = {"type": "LinearProjection",
                                     "d_model": corrected.get("encoder", {}).get("d_model", 64)}
            rules.append(f"ANTAG-04: PatchEmb+{mix} → LinearProj (shape 불일치)")

        # 비-Patch encoder + Patch mixer → mixer를 LinearMix로
        if enc != "PatchEmbedding" and mix in PATCH_MIXERS:
            corrected["temporal_mixer"] = {"type": "LinearMix"}
            rules.append(f"ANTAG-05: {enc}+{mix} → LinearMix (shape 불일치)")

        # MovingAvgDecomp + is_stationary → 분해 제거
        if decomp == "MovingAvgDecomp" and is_stat:
            corrected["decomposer"] = None
            rules.append("ANTAG-06: MovingAvgDecomp+stationary → 제거 (분해 불필요)")

        # ── Conditional (경고) ──

        # AttentionMix + 소형 데이터 → 경고
        if mix in ("AttentionMix", "PatchAttentionMix") and n_rows < 5000:
            warns.append(f"COND-01: {mix}+{n_rows}행 → 과적합 위험")
            # d_model 하향
            if isinstance(corrected.get("encoder"), dict):
                d = corrected["encoder"].get("d_model", 64)
                corrected["encoder"]["d_model"] = min(d, 32)
                rules.append("COND-01: d_model ≤ 32 (소형 데이터)")

        # DistributionalHead(mixture) + 소형 데이터 → gaussian
        if head == "DistributionalHead":
            dist = corrected.get("head", {}).get("distribution", "gaussian")
            if dist == "mixture_gaussian" and n_rows < 10000:
                corrected["head"]["distribution"] = "gaussian"
                rules.append("COND-02: mixture→gaussian (데이터 부족)")

        # ── Synergy (좋은 조합 확인 — 보정 없음, 로그만) ──
        # 이건 보정이 아니라 "이 조합이 좋다"는 확인
        # Decision Trace에 기록용

        return SynergyResult(
            original=config, corrected=corrected,
            applied_rules=rules, warnings=warns,
        )
