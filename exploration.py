"""Exploration Engine — 적응형 후보 생성 + Greedy Ensemble.

ExplorationBudget: 라운드별 후보 수 결정 (n=1~3)
CandidateGenerator: exploit/explore/wildcard 후보 구성
GreedyEnsemble: 완료 모델 풀에서 최적 앙상블 선택
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np


# ══════════════════════════════════════════════════════════════
#  ExplorationBudget
# ══════════════════════════════════════════════════════════════

@dataclass
class ExplorationBudget:
    """라운드별 탐색 후보 수 결정."""
    n_candidates: int = 3
    strategy: str = "explore"  # explore, exploit, refine

    @staticmethod
    def decide(round_num: int, prev_scores: list[float]) -> "ExplorationBudget":
        """이전 라운드 점수 기반으로 후보 수 결정.

        Round 1: n=3 (정보 없으니 넓게)
        이전 개선 > 5%: n=3 explore
        이전 개선 1-5%: n=2 exploit
        이전 개선 < 1%: n=1 refine
        """
        if round_num <= 1 or len(prev_scores) < 1:
            return ExplorationBudget(n_candidates=3, strategy="explore")

        if len(prev_scores) >= 2:
            prev = prev_scores[-2]
            curr = prev_scores[-1]
            if prev > 0:
                improvement = (curr - prev) / prev
            else:
                improvement = 0.0
        else:
            improvement = 0.1  # 첫 비교: 보수적으로 explore

        if improvement > 0.05:
            return ExplorationBudget(n_candidates=3, strategy="explore")
        elif improvement > 0.01:
            return ExplorationBudget(n_candidates=2, strategy="exploit")
        else:
            return ExplorationBudget(n_candidates=1, strategy="refine")


# ══════════════════════════════════════════════════════════════
#  CandidateGenerator
# ══════════════════════════════════════════════════════════════

@dataclass
class Candidate:
    """학습 후보."""
    config: dict
    label: str  # "exploit", "explore", "wildcard"
    rationale: str


class CandidateGenerator:
    """슬롯별 추천 기반 후보 생성."""

    @staticmethod
    def generate(slot_recs: dict, diagnosis=None, ledger=None,
                 budget: ExplorationBudget = None,
                 profile: dict = None) -> list[Candidate]:
        """n개 후보 생성.

        1. exploit: KG 슬롯 추천 그대로
        2. explore: Diagnosis 기반 처방 (해당 슬롯만 교체)
        3. wildcard: 미시도 카테고리 강제
        """
        if budget is None:
            budget = ExplorationBudget()

        candidates = []

        # ── 1. Exploit: KG 추천 그대로 ──
        exploit_config = CandidateGenerator._build_config_from_recs(slot_recs, profile)
        candidates.append(Candidate(
            config=exploit_config,
            label="exploit",
            rationale="KG 슬롯별 추천 조합",
        ))

        if budget.n_candidates < 2:
            return candidates

        # ── 2. Explore: Diagnosis 기반 슬롯 교체 ──
        explore_config = CandidateGenerator._build_explore(
            exploit_config, slot_recs, diagnosis, ledger,
        )
        if explore_config:
            candidates.append(Candidate(
                config=explore_config,
                label="explore",
                rationale="진단 기반 슬롯 교체",
            ))

        if budget.n_candidates < 3 or len(candidates) >= budget.n_candidates:
            return candidates[:budget.n_candidates]

        # ── 3. Wildcard: DiversityLedger 미시도 블록 강제 ──
        wildcard_config = CandidateGenerator._build_wildcard(
            exploit_config, slot_recs, ledger,
        )
        if wildcard_config:
            candidates.append(Candidate(
                config=wildcard_config,
                label="wildcard",
                rationale="미시도 블록 탐색",
            ))

        return candidates[:budget.n_candidates]

    @staticmethod
    def _build_config_from_recs(slot_recs: dict, profile: dict | None) -> dict:
        """슬롯 추천에서 config dict 생성."""
        from cballm.ontology.kg_engine import compute_data_scale

        config = {}
        scale = slot_recs.get("_data_scale", {})

        for slot in ["normalizer", "decomposer", "encoder", "temporal_mixer",
                      "channel_mixer", "head", "loss", "constraint"]:
            rec = slot_recs.get(slot, {})
            recommended = rec.get("recommended", "None")

            if slot == "constraint":
                if isinstance(recommended, list):
                    config[slot] = [{"type": c} for c in recommended if c != "None"]
                elif recommended == "None":
                    config[slot] = []
                else:
                    config[slot] = [{"type": recommended}]
            elif recommended == "None" or recommended is None:
                config[slot] = None
            else:
                config[slot] = {"type": recommended}

        # HP 자동 결정
        d_model = scale.get("d_model", 64)
        enc_type = config.get("encoder", {}).get("type", "") if isinstance(config.get("encoder"), dict) else ""
        mix_type = config.get("temporal_mixer", {}).get("type", "") if isinstance(config.get("temporal_mixer"), dict) else ""

        if enc_type == "PatchEmbedding":
            d_model = max(64, d_model)
        if mix_type in ("PatchAttentionMix", "AttentionMix"):
            d_model = max(64, d_model)
            n_heads = scale.get("n_heads", 4)
            if d_model % n_heads != 0:
                d_model = ((d_model + n_heads - 1) // n_heads) * n_heads

        if isinstance(config.get("encoder"), dict):
            config["encoder"]["d_model"] = d_model
            if enc_type == "PatchEmbedding":
                config["encoder"].setdefault("patch_len", 16)
                config["encoder"].setdefault("stride", 8)

        if isinstance(config.get("temporal_mixer"), dict):
            if mix_type in ("PatchAttentionMix", "AttentionMix"):
                config["temporal_mixer"].setdefault("n_heads", scale.get("n_heads", 4))
                config["temporal_mixer"].setdefault("n_layers", scale.get("n_layers", 3))
            elif mix_type in ("MLPMix", "GatedMLPMix"):
                config["temporal_mixer"].setdefault("hidden_dim", d_model * 2)
            elif mix_type == "ConvMix":
                config["temporal_mixer"].setdefault("kernel_size", 7)
                config["temporal_mixer"].setdefault("n_layers", 2)

        if isinstance(config.get("decomposer"), dict):
            config["decomposer"].setdefault("kernel_size", 25)

        if isinstance(config.get("normalizer"), dict):
            if config["normalizer"].get("type") == "RevIN":
                config["normalizer"]["affine"] = True

        if isinstance(config.get("head"), dict):
            config["head"]["output_dim"] = 1

        # training defaults
        n_rows = (profile or {}).get("n_rows", 10000)
        dominant = (profile or {}).get("dominant_period", 24)
        seq_len = max(96, (dominant or 24) * 2)
        config["input_design"] = {"seq_len": min(seq_len, max(96, n_rows // 10))}
        config["training"] = {"n_folds": 3 if n_rows < 20000 else 5}
        config["preprocessing"] = {"log_transform": False, "differencing": False}
        config["forecasting_setting"] = "MS"
        config["output_dim"] = 1
        config["_recipe_name"] = "custom"

        return config

    @staticmethod
    def _build_explore(base_config: dict, slot_recs: dict,
                       diagnosis, ledger) -> dict | None:
        """Diagnosis 기반 explore 후보. 약한 슬롯을 교체."""
        if not diagnosis or not hasattr(diagnosis, 'diagnosis'):
            # 첫 라운드: encoder를 다른 것으로
            config = copy.deepcopy(base_config)
            enc_opts = slot_recs.get("encoder", {}).get("options", [])
            current_enc = base_config.get("encoder", {}).get("type", "")
            for alt in enc_opts:
                if alt != current_enc and alt != "None":
                    config["encoder"] = {"type": alt, "d_model": base_config.get("encoder", {}).get("d_model", 64)}
                    if alt == "PatchEmbedding":
                        config["encoder"]["patch_len"] = 16
                        config["encoder"]["stride"] = 8
                        # temporal_mixer도 호환 변경
                        mix_opts = slot_recs.get("temporal_mixer", {}).get("options", [])
                        from cballm.blocks.temporal_mixer import PATCH_MIXERS
                        for pm in mix_opts:
                            if pm in PATCH_MIXERS:
                                config["temporal_mixer"] = {"type": pm, "n_heads": 4, "n_layers": 3}
                                break
                    return config
            return None

        # Diagnosis 기반: ineffective 블록을 다른 것으로
        diag = diagnosis.diagnosis
        blacklist = diag.block_blacklist
        if not blacklist:
            # blacklist 없으면 temporal_mixer를 바꿔봄
            config = copy.deepcopy(base_config)
            mix_opts = slot_recs.get("temporal_mixer", {}).get("options", [])
            current_mix = base_config.get("temporal_mixer", {}).get("type", "")
            for alt in mix_opts:
                if alt != current_mix and alt != "None":
                    config["temporal_mixer"] = {"type": alt}
                    return config
            return None

        config = copy.deepcopy(base_config)
        changed = False
        for slot in ["encoder", "temporal_mixer", "channel_mixer"]:
            current = config.get(slot, {}).get("type", "") if isinstance(config.get(slot), dict) else ""
            if current in blacklist:
                opts = slot_recs.get(slot, {}).get("options", [])
                for alt in opts:
                    if alt != current and alt not in blacklist and alt != "None":
                        config[slot] = {"type": alt}
                        changed = True
                        break
        return config if changed else None

    # complexity 한 단계 올린 mixer 매핑
    UPGRADE_MAP = {
        "LinearMix": "MLPMix",
        "MLPMix": "ConvMix",
        "ConvMix": "AttentionMix",
    }

    @staticmethod
    def _build_wildcard(base_config: dict, slot_recs: dict,
                        ledger) -> dict | None:
        """complexity를 한 단계 올린 조합. exploit과 다른 것을 보장."""
        config = copy.deepcopy(base_config)
        current_mix = config.get("temporal_mixer", {}).get("type", "") if isinstance(config.get("temporal_mixer"), dict) else ""

        # temporal_mixer를 한 단계 올림
        upgrade = CandidateGenerator.UPGRADE_MAP.get(current_mix)
        if upgrade:
            opts = slot_recs.get("temporal_mixer", {}).get("options", [])
            if upgrade in opts:
                config["temporal_mixer"] = {"type": upgrade}
                # HP 보강
                scale = slot_recs.get("_data_scale", {})
                if upgrade in ("AttentionMix",):
                    config["temporal_mixer"]["n_heads"] = scale.get("n_heads", 4)
                    config["temporal_mixer"]["n_layers"] = scale.get("n_layers", 2)
                elif upgrade in ("MLPMix", "ConvMix"):
                    d = config.get("encoder", {}).get("d_model", 64) if isinstance(config.get("encoder"), dict) else 64
                    config["temporal_mixer"]["hidden_dim"] = d * 2
                return config

        # temporal_mixer 업그레이드 불가 → 미시도 블록 시도
        if ledger:
            for slot in ["temporal_mixer", "encoder"]:
                tried = set(ledger.get_tried_blocks(slot))
                opts = slot_recs.get(slot, {}).get("options", [])
                for alt in opts:
                    if alt not in tried and alt != "None" and alt != current_mix:
                        config[slot] = {"type": alt}
                        return config

        return None


# ══════════════════════════════════════════════════════════════
#  Greedy Ensemble (Caruana 2004)
# ══════════════════════════════════════════════════════════════

@dataclass
class EnsembleResult:
    """앙상블 결과."""
    members: list[int]           # round_num 목록
    weights: list[float]
    ensemble_score: float        # CompositeScore.weighted_score
    single_best_score: float
    improvement: float           # ensemble vs single best
    prediction: np.ndarray | None = None  # (N, H)

    def summary(self) -> str:
        if len(self.members) == 1:
            return f"Single best (R{self.members[0]}), score={self.single_best_score:.4f}"
        return (f"Ensemble R{self.members} (weights={[round(w,2) for w in self.weights]}), "
                f"score={self.ensemble_score:.4f} vs single={self.single_best_score:.4f}, "
                f"improvement={self.improvement:.2%}")


class GreedyEnsemble:
    """Caruana 2004: Greedy forward selection ensemble."""

    @staticmethod
    def select(round_predictions: dict[int, np.ndarray],
               round_targets: np.ndarray,
               round_scores: dict[int, float]) -> EnsembleResult:
        """val predictions에서 최적 앙상블 선택.

        Args:
            round_predictions: {round_num: (N, H) predictions}
            round_targets: (N, H) targets
            round_scores: {round_num: CompositeScore.weighted_score}
        """
        if not round_predictions:
            return EnsembleResult(members=[], weights=[], ensemble_score=0,
                                  single_best_score=0, improvement=0)

        # single best
        best_round = max(round_scores, key=round_scores.get)
        best_score = round_scores[best_round]
        best_pred = round_predictions[best_round]

        if len(round_predictions) == 1:
            return EnsembleResult(
                members=[best_round], weights=[1.0],
                ensemble_score=best_score, single_best_score=best_score,
                improvement=0.0, prediction=best_pred,
            )

        # greedy forward selection
        members = [best_round]
        current_pred = best_pred.copy()
        current_mae = float(np.abs(current_pred - round_targets).mean())

        available = set(round_predictions.keys()) - {best_round}

        for _ in range(min(3, len(available))):  # 최대 3개 추가
            best_add = None
            best_add_mae = current_mae

            for r in available:
                # 후보 추가 시 평균 예측
                n = len(members) + 1
                trial_pred = (current_pred * len(members) + round_predictions[r]) / n
                trial_mae = float(np.abs(trial_pred - round_targets).mean())

                if trial_mae < best_add_mae:
                    best_add_mae = trial_mae
                    best_add = r

            if best_add is not None and best_add_mae < current_mae * 0.999:  # 0.1% 이상 개선
                members.append(best_add)
                n = len(members)
                current_pred = sum(round_predictions[r] for r in members) / n
                current_mae = best_add_mae
                available.discard(best_add)
            else:
                break

        # 최종 점수
        ensemble_mae = float(np.abs(current_pred - round_targets).mean())
        single_mae = float(np.abs(best_pred - round_targets).mean())

        # 앙상블이 단일보다 나쁘면 단일 채택
        if ensemble_mae >= single_mae:
            return EnsembleResult(
                members=[best_round], weights=[1.0],
                ensemble_score=best_score, single_best_score=best_score,
                improvement=0.0, prediction=best_pred,
            )

        improvement = (single_mae - ensemble_mae) / single_mae
        weights = [1.0 / len(members)] * len(members)

        return EnsembleResult(
            members=members, weights=weights,
            ensemble_score=best_score * (1 + improvement),
            single_best_score=best_score,
            improvement=improvement,
            prediction=current_pred,
        )
