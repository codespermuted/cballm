"""Architect — Block-first Decision Protocol (v3).

KG Matcher가 슬롯별 추천을 제공하고, Architect는 각 슬롯을 독립적으로 선택.
Q1: Normalizer, Q2: Decomposer, Q3: Encoder, Q4: Temporal Mixer,
Q5: Channel Mixer, Q6: Head, Q7: Loss, Q8: Constraint
각 질문에 Scout 진단 기반 추천을 KG rule이 제공.
LLM은 각 질문에 단답 (블록 이름 또는 agree).
"""
from __future__ import annotations

import json
import re

from cballm.engine import chat
from cballm.session import WorkerSession


class Architect:
    """Block-first Decision Protocol — 슬롯별 독립 블록 선택 (v3)."""
    name = "architect"
    description = "Block-first Decision Protocol"
    model_profile = "reasoning"

    SYSTEM_PROMPT = (
        "You are a time series modeling advisor. "
        "For each question, pick ONE option from the list. "
        "Answer the option name only, then optionally a SHORT comment (max 15 words). "
        "Format: ANSWER | comment (optional)"
    )

    # 슬롯 순서 (Q1-Q8)
    SLOT_ORDER = [
        "normalizer", "decomposer", "encoder", "temporal_mixer",
        "channel_mixer", "head", "loss", "constraint",
    ]

    def __init__(self, cwd: str = "", rules: str = ""):
        self.cwd = cwd
        self.session = WorkerSession(worker_name=self.name, system_prompt=self.SYSTEM_PROMPT)
        self.decisions: list[dict] = []

    def run(self, task: str) -> dict:
        """Block-first: 슬롯별 독립 블록 선택."""
        profile = self._parse_profile(task)
        kg_result = self._parse_kg_result(task)
        slot_recs = self._parse_slot_recommendations(task)
        blacklist = self._parse_blacklist(task)
        prev_config = self._parse_prev_config(task)
        retry_type = self._parse_retry_type(task)

        config = self._run_block_protocol(
            profile, kg_result, slot_recs, blacklist, prev_config, retry_type,
        )

        # 모델링 리포트
        report_lines = ["=== MODELING DECISIONS (v3 block-first) ==="]
        for d in self.decisions:
            report_lines.append(
                f"Q{d['step']}: {d['topic']} = {d['decision']} "
                f"(추천: {d['default']}, LLM: {d['answer']})"
            )
            if d.get("comment"):
                report_lines.append(f"  Insight: {d['comment']}")
        report_lines.append(f"\nConfig: {json.dumps(config, ensure_ascii=False)}")

        config_json = json.dumps(config, ensure_ascii=False)

        return {
            "worker": self.name,
            "response": config_json,
            "code": None,
            "execution_result": "\n".join(report_lines),
        }

    def _run_block_protocol(self, profile: dict, kg_result: dict,
                            slot_recs: dict, blacklist: list,
                            prev_config: dict, retry_type: str) -> dict:
        """v3 Block-first Decision Protocol.

        각 슬롯을 독립적으로 결정. KG 추천을 default로 제시.
        RETRY_HP: prev_config 유지, 특정 슬롯만 변경.
        RETRY_BLOCK: blacklist의 블록을 다른 것으로 교체.
        """
        from cballm.ontology.kg_engine import compute_data_scale, resolve_hp

        self.decisions = []
        setting = kg_result.get("forecasting_setting", "MS")
        output_dim = kg_result.get("output_dim", 1)
        data_scale = slot_recs.get("_data_scale", {})
        if not data_scale:
            n_rows = profile.get("n_rows", 0)
            n_feat = profile.get("n_features", 1)
            pred_len = profile.get("pred_len", 96)
            data_scale = compute_data_scale(n_rows, n_feat, pred_len)

        config = {}

        for i, slot in enumerate(self.SLOT_ORDER, 1):
            rec = slot_recs.get(slot, {})
            recommended = rec.get("recommended", "None")
            options = rec.get("options", [str(recommended)])
            confidence = rec.get("confidence", "medium")
            reason = rec.get("reason", "")

            # 이전 config 유지 (RETRY_HP 시)
            if retry_type == "RETRY_HP" and prev_config and slot in prev_config:
                # blacklist에 있으면 교체, 아니면 유지
                prev_val = prev_config[slot]
                prev_type = prev_val.get("type") if isinstance(prev_val, dict) else str(prev_val)
                if prev_type not in blacklist:
                    config[slot] = prev_val
                    self._record(str(i), slot, reason, str(recommended),
                                "유지", str(prev_type), "RETRY_HP: 유지")
                    continue

            # blacklist 필터
            filtered = [o for o in options if o not in blacklist]
            if not filtered:
                filtered = options  # fallback

            # confidence=high이고 옵션 1개면 LLM에게 안 물음
            if confidence == "high" and len(filtered) <= 2:
                selected = recommended if recommended not in blacklist else filtered[0]
                self._record(str(i), slot, reason, str(recommended),
                            "auto(high)", str(selected), "")
            else:
                # LLM에게 질문
                opts_str = " / ".join(filtered)
                question = f"Q{i}. {slot}? 추천: {recommended} ({reason}). 선택지: {opts_str}"
                answer, comment = self._ask(question)

                # 파싱: 답변에서 옵션 이름 매칭
                selected = recommended
                for opt in filtered:
                    if opt.lower() in answer.lower():
                        selected = opt
                        break
                # "agree" 또는 빈 답변 → recommended
                if not answer or "agree" in answer.lower():
                    selected = recommended

                self._record(str(i), slot, reason, str(recommended),
                            answer, str(selected), comment)

            # config에 반영
            if slot == "constraint":
                # constraint는 리스트
                if isinstance(selected, list):
                    config[slot] = [{"type": c} for c in selected if c != "None"]
                elif selected == "None":
                    config[slot] = []
                else:
                    config[slot] = [{"type": selected}]
            elif selected == "None" or selected is None:
                config[slot] = None
            else:
                config[slot] = {"type": selected}

        # ── HP 자동 결정 ──
        enc_type = config.get("encoder", {}).get("type", "LinearProjection") if isinstance(config.get("encoder"), dict) else "LinearProjection"
        mix_type = config.get("temporal_mixer", {}).get("type", "LinearMix") if isinstance(config.get("temporal_mixer"), dict) else "LinearMix"

        # d_model: data_scale 기반 + block hp_constraints
        d_model = data_scale.get("d_model", 64)
        # Attention 계열은 최소 64
        if mix_type in ("PatchAttentionMix", "AttentionMix"):
            d_model = max(64, d_model)
        # Patch 계열은 최소 64
        if enc_type == "PatchEmbedding":
            d_model = max(64, d_model)
        # n_heads 호환
        n_heads = data_scale.get("n_heads", 4)
        if d_model % n_heads != 0:
            d_model = ((d_model + n_heads - 1) // n_heads) * n_heads

        # encoder config 보강
        if isinstance(config.get("encoder"), dict):
            config["encoder"]["d_model"] = d_model
            if enc_type == "PatchEmbedding":
                config["encoder"].setdefault("patch_len", 16)
                config["encoder"].setdefault("stride", 8)
            elif enc_type == "FourierEmbedding":
                # n_harmonics: ACF 기반
                max_acf = profile.get("max_acf_at_known_periods", 0)
                config["encoder"].setdefault("n_harmonics", max(1, min(5, int(max_acf * 5))))

        # temporal_mixer config 보강
        if isinstance(config.get("temporal_mixer"), dict):
            if mix_type in ("PatchAttentionMix", "AttentionMix"):
                config["temporal_mixer"].setdefault("n_heads", n_heads)
                config["temporal_mixer"].setdefault("n_layers", data_scale.get("n_layers", 3))
            elif mix_type in ("MLPMix", "GatedMLPMix"):
                config["temporal_mixer"].setdefault("hidden_dim", d_model * 2)
            elif mix_type == "ConvMix":
                config["temporal_mixer"].setdefault("kernel_size", 7)
                config["temporal_mixer"].setdefault("n_layers", 2)
            elif mix_type == "FrequencyMix":
                config["temporal_mixer"].setdefault("top_k", 5)

        # decomposer config
        if isinstance(config.get("decomposer"), dict):
            dominant = profile.get("dominant_period")
            ks = dominant if dominant and dominant > 3 else 25
            if ks % 2 == 0:
                ks += 1
            config["decomposer"].setdefault("kernel_size", ks)

        # head config
        if isinstance(config.get("head"), dict):
            config["head"]["output_dim"] = output_dim

        # normalizer config
        if isinstance(config.get("normalizer"), dict):
            norm_type = config["normalizer"].get("type", "")
            if norm_type == "RevIN":
                config["normalizer"]["affine"] = True

        # ── Training strategy (rule-based, LLM 불필요) ──
        n_rows = profile.get("n_rows", 0)
        pred_len = profile.get("pred_len", 96)
        seq_len = max(96, (profile.get("dominant_period") or 24) * 2)
        seq_len = min(seq_len, max(96, n_rows // 10))

        if n_rows < 5000:
            n_folds = 2
        elif n_rows < 20000:
            n_folds = 3
        else:
            n_folds = 5

        config["input_design"] = {"seq_len": seq_len}
        config["training"] = {"n_folds": n_folds}
        config["preprocessing"] = {"log_transform": False, "differencing": False}
        config["forecasting_setting"] = setting
        config["output_dim"] = output_dim
        config["_recipe_name"] = "custom"  # block-first는 항상 custom

        # ── 호환성 자동 보정 ──
        config = self._validate_compatibility(config)

        return config

    @staticmethod
    def _validate_compatibility(config: dict) -> dict:
        """블록 조합의 호환성 검증 + 자동 보정."""
        head_type = config.get("head", {}).get("type", "") if isinstance(config.get("head"), dict) else ""
        loss_type = config.get("loss", {}).get("type", "") if isinstance(config.get("loss"), dict) else ""
        enc_type = config.get("encoder", {}).get("type", "") if isinstance(config.get("encoder"), dict) else ""
        mix_type = config.get("temporal_mixer", {}).get("type", "") if isinstance(config.get("temporal_mixer"), dict) else ""

        # DistributionalHead + MAE → NLL로 자동 보정
        if head_type == "DistributionalHead" and loss_type in ("MAE", "MSE"):
            config["loss"] = {"type": "GaussianNLL"}

        # PatchEmbedding + 비-Patch mixer → LinearProjection으로 보정
        from cballm.blocks.temporal_mixer import PATCH_MIXERS
        if enc_type == "PatchEmbedding" and mix_type not in PATCH_MIXERS:
            config["encoder"]["type"] = "LinearProjection"

        # 비-Patch encoder + Patch mixer → LinearMix로 보정
        if enc_type != "PatchEmbedding" and mix_type in PATCH_MIXERS:
            config["temporal_mixer"]["type"] = "LinearMix"

        return config

    # ── Legacy v2 (reference) ──

    def _run_protocol_v2_legacy(self, profile: dict, kg_result: dict,
                               prev_configs: list, critic: dict) -> dict:
        """v2 Decision Protocol: KG-bounded."""
        self.decisions = []

        # KG Matcher가 제공한 정보
        setting = kg_result.get("forecasting_setting", "MS")
        output_dim = kg_result.get("output_dim", 1)
        normalizer = kg_result.get("normalizer")
        candidates = kg_result.get("candidate_recipes", [])
        valid_encoders = kg_result.get("valid_encoders", ["LinearProjection"])
        valid_mixers = kg_result.get("valid_temporal_mixers", ["LinearMix"])
        valid_channel = kg_result.get("valid_channel_mixers", ["None"])
        valid_losses = kg_result.get("valid_losses", ["MAE"])

        # Step 1: 레시피 선택
        recipe = self._step1_recipe_selection(candidates, prev_configs, critic)

        # Step 2: Encoder HP
        encoder_cfg = self._step2_encoder(recipe, profile, valid_encoders)

        # Step 3: TemporalMixer HP
        mixer_cfg = self._step3_temporal_mixer(recipe, profile, valid_mixers, prev_configs)

        # Step 4: ChannelMixer + Loss
        channel_cfg = self._step4_channel_mixer(recipe, profile, valid_channel, kg_result)
        loss_cfg = self._step4b_loss(profile, critic, valid_losses)

        # Step 5: Constraint
        constraints = self._step5_constraint(profile)

        # Step 6: Training strategy
        preprocessing = self._step6a_preprocessing(profile)
        input_design = self._step6b_input_design(profile, prev_configs)
        training = self._step6c_training(profile, prev_configs)

        # Step 7: 분포 선택 (v2.1)
        dist_cfg = self._step7_distribution(profile, kg_result)
        head_cfg = dist_cfg["head"]
        # 분포 NLL loss가 선택되면 loss_cfg 오버라이드
        if dist_cfg.get("loss"):
            loss_cfg = dist_cfg["loss"]

        # Normalizer config 생성
        normalizer_cfg = None
        if normalizer:
            normalizer_cfg = {"type": normalizer}
            if normalizer == "RevIN":
                normalizer_cfg["affine"] = True

        return {
            "normalizer": normalizer_cfg,
            "encoder": encoder_cfg,
            "temporal_mixer": mixer_cfg,
            "channel_mixer": channel_cfg,
            "head": head_cfg,
            "constraint": constraints,
            "loss": loss_cfg,
            "preprocessing": preprocessing,
            "input_design": input_design,
            "training": training,
            "forecasting_setting": setting,
            "output_dim": output_dim,
            "_recipe_name": recipe.get("name", "custom"),
        }

    # ── Step 1: 레시피 선택 ──

    def _step1_recipe_selection(self, candidates: list, prev_configs: list,
                                 critic: dict) -> dict:
        if not candidates:
            recipe = {"name": "DLinear", "blocks": {
                "normalizer": "RevIN", "encoder": "LinearProjection",
                "temporal_mixer": "LinearMix", "channel_mixer": None,
                "head": "LinearHead",
            }}
            self._record("1", "레시피 선택", "후보 없음", "DLinear", "auto", "DLinear", "")
            return recipe

        # Critic의 피드백에 따라 다른 레시피 시도
        retry_type = critic.get("verdict", "")
        prev_recipe_names = [p.get("recipe_name", "") for p in prev_configs]

        if retry_type == "RETRY_RECIPE" and len(candidates) > 1:
            # 이전에 시도하지 않은 레시피 선택
            for c in candidates:
                if c.get("name") not in prev_recipe_names:
                    evidence = f"Critic: RETRY_RECIPE. 이전: {prev_recipe_names}"
                    self._record("1", "레시피 선택", evidence, c["name"],
                                "auto", c["name"], "이전과 다른 레시피")
                    return c
            # 모두 시도했으면 score 가장 높은 것
            recipe = candidates[0]
        else:
            # 후보 목록을 LLM에 보여주고 선택
            names = [c.get("name", "?") for c in candidates[:5]]
            evidence = f"후보: {names}"
            question = f"레시피 후보: {', '.join(names)}. 첫 시도는 DLinear(baseline). 어떤 레시피? 이름만."
            answer, comment = self._ask(question)

            recipe = candidates[0]  # default
            for c in candidates:
                if c.get("name", "").lower() in answer.lower():
                    recipe = c
                    break

            self._record("1", "레시피 선택", evidence, candidates[0].get("name", "?"),
                         answer, recipe.get("name", "?"), comment)

        return recipe

    # ── Step 2: Encoder ──

    def _step2_encoder(self, recipe: dict, profile: dict,
                       valid_encoders: list) -> dict:
        from cballm.ontology.kg_engine import compute_data_scale, resolve_hp

        blocks = recipe.get("blocks", {})
        enc_type = blocks.get("encoder", "LinearProjection")
        enc_config = recipe.get("encoder_config", {})

        if enc_type not in valid_encoders and valid_encoders:
            enc_type = valid_encoders[0]

        # 데이터 기반 규모 결정
        n_rows = profile.get("n_rows", 0)
        n_features = profile.get("n_features", 1)
        pred_len = profile.get("pred_len", 96)
        data_scale = compute_data_scale(n_rows, n_features, pred_len)

        # HP 우선순위 통합: constraints > recipe > data_scale > llm
        hp = resolve_hp(recipe, data_scale)
        d_model = hp["d_model"]["recommended"]
        d_min = hp["d_model"]["min"]
        d_max = hp["d_model"]["max"]

        evidence = (f"encoder={enc_type}, n_features={n_features}, "
                    f"scale={data_scale['scale']}")
        question = f"{evidence}. d_model: {d_min}~{d_max} (권장: {d_model}). 숫자."
        answer, comment = self._ask(question)
        d_model = self._parse_int(answer, default=d_model, min_val=d_min, max_val=d_max)

        # Attention 호환: d_model은 n_heads의 배수
        mixer_type = blocks.get("temporal_mixer", "")
        if mixer_type in ("PatchAttentionMix", "AttentionMix"):
            n_heads = hp["n_heads"]["recommended"]
            if d_model % n_heads != 0:
                d_model = ((d_model + n_heads - 1) // n_heads) * n_heads
                d_model = max(d_min, min(d_max, d_model))

        result = {"type": enc_type, "d_model": d_model}
        if enc_type == "PatchEmbedding":
            result["patch_len"] = enc_config.get("patch_len", 16)
            result["stride"] = enc_config.get("stride", 8)
        elif enc_type == "FourierEmbedding":
            result["n_harmonics"] = enc_config.get("n_harmonics", 3)

        self._record("2", "Encoder", evidence, f"{enc_type}(d={d_model})",
                     answer, f"{enc_type}(d={d_model})", comment)
        return result

    # ── Step 3: TemporalMixer ──

    def _step3_temporal_mixer(self, recipe: dict, profile: dict,
                               valid_mixers: list, prev_configs: list) -> dict:
        blocks = recipe.get("blocks", {})
        mix_type = blocks.get("temporal_mixer", "LinearMix")
        mix_config = recipe.get("temporal_mixer_config", {})

        if mix_type not in valid_mixers and valid_mixers:
            mix_type = valid_mixers[0]

        result = {"type": mix_type}
        result.update(mix_config)

        evidence = f"temporal_mixer={mix_type}"
        self._record("3", "TemporalMixer", evidence, mix_type,
                     "auto (recipe)", mix_type, "")
        return result

    # ── Step 4: ChannelMixer + Loss ──

    def _step4_channel_mixer(self, recipe: dict, profile: dict,
                              valid_channel: list, kg_result: dict) -> dict | None:
        blocks = recipe.get("blocks", {})
        ch_type = blocks.get("channel_mixer")

        if ch_type is None or ch_type == "null":
            self._record("4a", "ChannelMixer", "recipe=null", "null", "auto", "null", "")
            return None

        ch_config = recipe.get("channel_mixer_config", {})
        result = {"type": ch_type}
        result.update(ch_config)

        evidence = f"channel_mixer={ch_type}"
        self._record("4a", "ChannelMixer", evidence, ch_type, "auto", ch_type, "")
        return result

    def _step4b_loss(self, profile: dict, critic: dict, valid_losses: list) -> dict:
        extreme_ratio = critic.get("extreme_ratio", 1.0)
        skew = abs(profile.get("target_skew", 0))

        if extreme_ratio > 2.0 and "AsymmetricLoss" in valid_losses:
            evidence = f"extreme_ratio={extreme_ratio:.1f}"
            question = f"{evidence}. Asymmetric loss(under_weight=2.0) 제안. under_weight? (1~5)"
            answer, comment = self._ask(question)
            w = self._parse_float(answer, default=2.0, min_val=1.0, max_val=5.0)
            loss = {"type": "Asymmetric", "under_weight": w, "over_weight": 1.0}
            self._record("4b", "Loss", evidence, "Asymmetric(2.0)", answer,
                         f"Asymmetric({w})", comment)
        elif skew > 1.5 and "Huber" in valid_losses:
            evidence = f"skew={skew:.2f}"
            question = f"{evidence}. Huber 제안. 동의? MAE 선호하면 'MAE'."
            answer, comment = self._ask(question)
            if "mae" in answer.lower():
                loss = {"type": "MAE"}
                self._record("4b", "Loss", evidence, "Huber", answer, "MAE", comment)
            else:
                loss = {"type": "Huber"}
                self._record("4b", "Loss", evidence, "Huber", answer, "Huber", comment)
        else:
            loss = {"type": "MAE"}
            self._record("4b", "Loss", f"skew={skew:.2f}", "MAE", "auto", "MAE", "")

        return loss

    # ── Step 5: Constraint ──

    def _step5_constraint(self, profile: dict) -> list:
        constraints = []
        can_neg = profile.get("can_be_negative", True)
        t_min = profile.get("target_min", -999)

        if not can_neg and t_min >= 0:
            evidence = f"target min={t_min:.2f}, always non-negative"
            question = f"{evidence}. Positivity constraint 제안. 동의? (yes/no)"
            answer, comment = self._ask(question)
            if "no" not in answer.lower():
                constraints.append({"type": "Positivity"})
            self._record("5", "Constraint", evidence, "Positivity", answer,
                         "Positivity" if constraints else "없음", comment)
        else:
            self._record("5", "Constraint", f"can_negative={can_neg}", "없음",
                         "auto", "없음", "")

        return constraints

    # ── Step 6: Preprocessing + Input + Training ──

    def _step6a_preprocessing(self, profile: dict) -> dict:
        skew = abs(profile.get("target_skew", 0))
        target_min = profile.get("target_min", 0)
        is_stationary = profile.get("is_stationary", True)

        log_transform = False
        differencing = False

        if target_min > 0 and skew > 2.0:
            question = f"skew={skew:.2f}, min={target_min:.2f}. Log transform? (yes/no)"
            answer, comment = self._ask(question)
            log_transform = "no" not in answer.lower()

        if not is_stationary:
            # RevIN이 non-stationarity를 처리하므로 differencing 불필요한 경우가 많음
            question = f"Non-stationary. RevIN이 처리하므로 differencing 불요. 동의? (yes/no)"
            answer, comment = self._ask(question)
            differencing = "no" in answer.lower()  # "no"면 differencing 적용

        self._record("6a", "전처리", f"skew={skew:.2f}, stationary={is_stationary}",
                     f"log={log_transform}, diff={differencing}", "auto",
                     f"log={log_transform}, diff={differencing}", "")

        return {"log_transform": log_transform, "differencing": differencing}

    def _step6b_input_design(self, profile: dict, prev_configs: list) -> dict:
        n_rows = profile.get("n_rows", 0)
        dominant = profile.get("dominant_period", 24)
        pred_len = profile.get("pred_len", 96)

        default_seq = max(96, (dominant or 24) * 2)
        default_seq = min(default_seq, max(96, n_rows // 10))

        evidence = f"dominant={dominant}, n_rows={n_rows}, pred_len={pred_len}"
        question = f"{evidence}. seq_len={default_seq}? ({pred_len}~512)"
        answer, comment = self._ask(question)
        seq_len = self._parse_int(answer, default=default_seq, min_val=pred_len, max_val=512)

        self._record("6b", "입력 설계", evidence, f"seq_len={default_seq}",
                     answer, f"seq_len={seq_len}", comment)
        return {"seq_len": seq_len}

    def _step6c_training(self, profile: dict, prev_configs: list) -> dict:
        n_rows = profile.get("n_rows", 0)
        if n_rows < 5000:
            default_folds = 2
        elif n_rows < 20000:
            default_folds = 3
        else:
            default_folds = 5

        question = f"n_rows={n_rows}. CV fold={default_folds}? (2~5)"
        answer, comment = self._ask(question)
        n_folds = self._parse_int(answer, default=default_folds, min_val=2, max_val=5)

        self._record("6c", "학습 전략", f"n_rows={n_rows}", f"folds={default_folds}",
                     answer, f"folds={n_folds}", comment)
        return {"n_folds": n_folds}

    # ── Step 7: 분포 선택 (v2.1) ──

    def _step7_distribution(self, profile: dict, kg_result: dict) -> dict:
        """분포 선택. KG rule이 사전 필터링한 후보를 LLM에 제시."""
        # KG에서 추천 분포 추출
        valid_dists = kg_result.get("valid_distributions", ["gaussian"])
        recommended = kg_result.get("recommended_distribution", "gaussian")

        # 분포 관련 통계량 수집
        kurtosis = profile.get("target_kurtosis", 0)
        tail_idx = profile.get("tail_index", 0)
        jb_p = profile.get("jarque_bera_pvalue", 1.0)
        skew = abs(profile.get("target_skew", 0))
        can_neg = profile.get("can_be_negative", True)

        # 기본 head (분포 미사용)
        output_dim = kg_result.get("output_dim", 1)
        default_head = {"type": "LinearHead", "output_dim": output_dim}

        # 분포 사용 조건이 약하면 LinearHead로 fallback
        needs_dist = (
            jb_p < 0.05          # 정규성 기각 → 분포 모델링 가치 있음
            or kurtosis > 5      # heavy tail
            or tail_idx > 0.02   # 극단값 빈도 높음
            or (not can_neg and skew > 1.0)  # 강한 right skew
        )

        if not needs_dist:
            self._record("7", "분포 선택", f"JB_p={jb_p:.3f}, kurtosis={kurtosis:.1f}",
                         "LinearHead (분포 불필요)", "auto", "LinearHead", "")
            return {"head": default_head, "loss": None}

        # LLM에게 후보 제시
        evidence = (
            f"kurtosis={kurtosis:.1f}, tail_index={tail_idx:.3f}, "
            f"JB_p={jb_p:.3f}, skew={skew:.2f}, can_neg={can_neg}"
        )
        choices = ", ".join(valid_dists)
        question = (
            f"{evidence}. 분포 후보: {choices}. "
            f"추천: {recommended}. 분포 이름만."
        )
        answer, comment = self._ask(question)

        # 파싱
        selected = recommended
        for d in valid_dists:
            if d.lower() in answer.lower():
                selected = d
                break

        self._record("7", "분포 선택", evidence, recommended,
                     answer, selected, comment)

        # 분포 → head + loss 매핑
        dist_loss_map = {
            "gaussian": "GaussianNLL",
            "student_t": "StudentTNLL",
            "log_normal": "LogNormalNLL",
            "mixture_gaussian": "MixtureGaussianNLL",
        }

        head_cfg = {
            "type": "DistributionalHead",
            "output_dim": output_dim,
            "distribution": selected,
            "point_forecast": True,
        }
        if selected == "mixture_gaussian":
            head_cfg["n_components"] = 3

        loss_type = dist_loss_map.get(selected, "GaussianNLL")
        return {"head": head_cfg, "loss": {"type": loss_type}}

    # ── Helpers ──

    def _ask(self, question: str) -> tuple[str, str]:
        try:
            self.session.add_user(question)
            raw = chat(
                self.SYSTEM_PROMPT,
                self.session.messages,
                max_tokens=50,
                model=self.model_profile,
            )
            self.session.add_assistant(raw)
            raw = raw.strip()
            if "|" in raw:
                parts = raw.split("|", 1)
                return parts[0].strip(), parts[1].strip()
            return raw, ""
        except Exception:
            return "", ""

    def _record(self, step: str, topic: str, evidence: str,
                default: str, answer: str, decision: str, comment: str):
        self.decisions.append({
            "step": step, "topic": topic, "evidence": evidence,
            "default": default, "answer": answer, "decision": decision,
            "comment": comment,
        })

    @staticmethod
    def _parse_int(text: str, default: int, min_val: int = 0, max_val: int = 100) -> int:
        match = re.search(r'(\d+)', text)
        if match:
            return max(min_val, min(max_val, int(match.group(1))))
        return default

    @staticmethod
    def _parse_float(text: str, default: float, min_val: float = 0, max_val: float = 100) -> float:
        match = re.search(r'([0-9]+\.?[0-9]*)', text)
        if match:
            return max(min_val, min(max_val, float(match.group(1))))
        return default

    def _parse_profile(self, task: str) -> dict:
        profile: dict = {}
        m = re.search(r'\((\d+),\s*(\d+)\)', task)
        if m:
            profile["n_rows"], profile["n_cols"] = int(m.group(1)), int(m.group(2))

        profile["n_features"] = profile.get("n_cols", 1)

        seasonality: dict = {}
        s_match = re.search(r'Seasonality:\s*\[(.+?)\]', task)
        if s_match:
            for part in s_match.group(1).split(","):
                p_m = re.search(r'(\d+\w*)', part)
                a_m = re.search(r'ACF=([0-9.]+)', part)
                if p_m and a_m:
                    seasonality[p_m.group(1)] = float(a_m.group(1))
                elif p_m:
                    seasonality[p_m.group(1)] = 0.8 if "strong" in part else 0.5
        profile["seasonality"] = seasonality

        profile["is_stationary"] = "non-stationary" not in task
        profile["can_be_negative"] = "Can be negative: True" in task

        for field, key in [("target_min", "min"), ("target_max", "max"),
                           ("target_skew", "skew"), ("adf_p", "ADF p")]:
            m = re.search(rf'{key}=([0-9.\-]+)', task)
            if m:
                profile[field] = float(m.group(1))

        dom_m = re.search(r'Dominant period:\s*(\d+)', task)
        if dom_m:
            profile["dominant_period"] = int(dom_m.group(1))

        pred_m = re.search(r'PREDICTION_LENGTH\s*=\s*(\d+)', task)
        if pred_m:
            profile["pred_len"] = int(pred_m.group(1))

        # 분포 통계량 (v2.1)
        for field, key in [("target_kurtosis", "kurtosis"),
                           ("tail_index", "tail_index"),
                           ("jarque_bera_pvalue", "jarque_bera_p")]:
            m = re.search(rf'{key}=([0-9.\-]+)', task)
            if m:
                profile[field] = float(m.group(1))

        return profile

    def _parse_kg_result(self, task: str) -> dict:
        """KG Matcher 결과 파싱."""
        result: dict = {}

        for field in ["FORECASTING_SETTING", "OUTPUT_DIM", "NORMALIZER"]:
            m = re.search(rf'{field}=(\S+)', task)
            if m:
                val = m.group(1)
                if val == "null":
                    val = None
                elif val.isdigit():
                    val = int(val)
                key = field.lower()
                result[key] = val

        for field in ["VALID_ENCODERS", "VALID_TEMPORAL_MIXERS",
                      "VALID_CHANNEL_MIXERS", "VALID_LOSSES"]:
            m = re.search(rf'{field}=\[([^\]]*)\]', task)
            if m:
                items = [x.strip().strip("'\"") for x in m.group(1).split(",") if x.strip()]
                key = field.lower()
                result[key] = items

        # 후보 레시피 파싱 — registry에서 실제 blocks 로드
        from cballm.recipes.registry import load_verified_recipes
        all_recipes = load_verified_recipes()

        candidates = []
        for m in re.finditer(r'\[(\d+)\]\s+(\S+)\s+\(capacity=(\w+)\)', task):
            name = m.group(2)
            recipe_data = all_recipes.get(name, {})
            candidates.append({
                "name": name,
                "capacity": m.group(3),
                "blocks": recipe_data.get("blocks", {}),
                "d_model": recipe_data.get("d_model", 64),
                "hp_constraints": recipe_data.get("hp_constraints", {}),
                "encoder_config": recipe_data.get("encoder_config", {}),
                "temporal_mixer_config": recipe_data.get("temporal_mixer_config", {}),
                "channel_mixer_config": recipe_data.get("channel_mixer_config", {}),
            })
        result["candidate_recipes"] = candidates

        # 분포 추천 (v2.1)
        m = re.search(r'VALID_DISTRIBUTIONS=\[([^\]]*)\]', task)
        if m:
            result["valid_distributions"] = [
                x.strip().strip("'\"") for x in m.group(1).split(",") if x.strip()
            ]
        m = re.search(r'RECOMMENDED_DISTRIBUTION=(\S+)', task)
        if m:
            result["recommended_distribution"] = m.group(1)

        return result

    def _parse_prev_configs(self, task: str) -> list:
        configs = []
        for m in re.finditer(r'-\s*(.+?)\s*→\s*norm_MSE=([0-9.]+)', task):
            configs.append({"config": m.group(1), "norm_mse": float(m.group(2))})

        # TRIED_RECIPES 파싱 (brain.py에서 명시 전달)
        m = re.search(r'TRIED_RECIPES\s*=\s*\[([^\]]*)\]', task)
        if m:
            names = [x.strip().strip("'\"") for x in m.group(1).split(",") if x.strip()]
            for name in names:
                # 이미 있는지 확인
                existing = {c.get("recipe_name", "") for c in configs}
                if name not in existing:
                    configs.append({"recipe_name": name, "config": name, "norm_mse": 0})

        return configs

    def _parse_critic(self, task: str) -> dict:
        result: dict = {"extreme_ratio": 1.0, "verdict": ""}

        m = re.search(r'비율:\s*([0-9.]+)', task)
        if m:
            result["extreme_ratio"] = float(m.group(1))

        # RETRY_TYPE 우선 파싱 (brain.py에서 명시 전달)
        m = re.search(r'RETRY_TYPE\s*=\s*(\S+)', task)
        if m:
            result["verdict"] = m.group(1)
        else:
            for v in ["RETRY_HP", "RETRY_RECIPE", "RETRY_BLOCK", "DONE"]:
                if v in task:
                    result["verdict"] = v
                    break

        return result

    # ── v3 Block-first Parsers ──

    def _parse_slot_recommendations(self, task: str) -> dict:
        """KG Matcher의 SLOT_* 형태를 파싱."""
        slots = {}
        for m in re.finditer(r'SLOT_(\w+)=\{(.+?)\}', task):
            slot = m.group(1).lower()
            content = m.group(2)
            rec = {}
            # recommended 추출
            rm = re.search(r'recommended=([^,}]+)', content)
            if rm:
                val = rm.group(1).strip().strip("'\"[]")
                rec["recommended"] = val
            # options 추출
            om = re.search(r'options=\[([^\]]*)\]', content)
            if om:
                rec["options"] = [x.strip().strip("'\"") for x in om.group(1).split(",") if x.strip()]
            # confidence
            cm = re.search(r'confidence=(\w+)', content)
            if cm:
                rec["confidence"] = cm.group(1)
            # reason
            rsm = re.search(r'reason="([^"]*)"', content)
            if rsm:
                rec["reason"] = rsm.group(1)
            slots[slot] = rec

        # DATA_SCALE
        dm = re.search(r'DATA_SCALE=\{(.+?)\}', task)
        if dm:
            scale = {}
            for k in ["d_model", "n_layers", "n_heads"]:
                km = re.search(rf'{k}=(\d+)', dm.group(1))
                if km:
                    scale[k] = int(km.group(1))
            sm = re.search(r'scale=(\w+)', dm.group(1))
            if sm:
                scale["scale"] = sm.group(1)
            slots["_data_scale"] = scale

        return slots

    def _parse_blacklist(self, task: str) -> list:
        """Blacklist 파싱."""
        m = re.search(r'Blacklist:\s*(.+)', task)
        if m:
            return [x.strip() for x in m.group(1).split(",") if x.strip()]
        m = re.search(r'block_blacklist.*?\[([^\]]*)\]', task)
        if m:
            return [x.strip().strip("'\"") for x in m.group(1).split(",") if x.strip()]
        return []

    def _parse_prev_config(self, task: str) -> dict:
        """이전 라운드 config 파싱 (RETRY_HP 시 슬롯 유지용)."""
        # brain.py가 PREV_CONFIG=... 형태로 전달
        m = re.search(r'PREV_CONFIG=(\{.+\})', task, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        return {}

    def _parse_retry_type(self, task: str) -> str:
        """RETRY_TYPE 파싱."""
        m = re.search(r'RETRY_TYPE\s*=\s*(\S+)', task)
        return m.group(1) if m else ""
