"""Architect — KG-bounded Decision Protocol (v2).

KG Matcher가 제공한 유효 후보 안에서만 판단한다.
Step 1: 레시피 선택 (후보 중 매칭)
Step 2-3: HP 조정 (range 내 숫자)
Step 4: Loss 선택 (추천 + 확인)
Step 5: Constraint (추천 + 확인)
Step 6: Training strategy (fold 수)
"""
from __future__ import annotations

import json
import re

from cballm.engine import chat
from cballm.session import WorkerSession


class Architect:
    """KG-bounded Decision Protocol로 모델 설계를 결정한다."""
    name = "architect"
    description = "KG-bounded Decision Protocol 기반 모델 설계"
    model_profile = "reasoning"

    SYSTEM_PROMPT = (
        "You are a time series modeling advisor. "
        "For each question, answer the CLOSED part first (number or option name), "
        "then optionally add a SHORT comment (max 15 words) if you have an insight. "
        "Format: ANSWER | comment (optional)"
    )

    def __init__(self, cwd: str = "", rules: str = ""):
        self.cwd = cwd
        self.session = WorkerSession(worker_name=self.name, system_prompt=self.SYSTEM_PROMPT)
        self.decisions: list[dict] = []

    def run(self, task: str) -> dict:
        """KG Matcher 결과 + Scout 프로파일 기반으로 모델 설계 결정."""
        profile = self._parse_profile(task)
        kg_result = self._parse_kg_result(task)
        prev_configs = self._parse_prev_configs(task)
        critic = self._parse_critic(task)

        config = self._run_protocol(profile, kg_result, prev_configs, critic)

        # 모델링 리포트
        report_lines = ["=== MODELING DECISIONS (v2) ==="]
        for d in self.decisions:
            report_lines.append(f"Step {d['step']}: {d['topic']}")
            report_lines.append(f"  Evidence: {d['evidence']}")
            report_lines.append(f"  Default: {d['default']}")
            report_lines.append(f"  LLM: {d['answer']}")
            if d.get("comment"):
                report_lines.append(f"  Insight: {d['comment']}")
            report_lines.append(f"  -> Decision: {d['decision']}")
            report_lines.append("")
        report_lines.append(f"Config: {json.dumps(config, ensure_ascii=False)}")

        config_json = json.dumps(config, ensure_ascii=False)

        return {
            "worker": self.name,
            "response": config_json,
            "code": None,
            "execution_result": "\n".join(report_lines),
        }

    def _run_protocol(self, profile: dict, kg_result: dict,
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
            "head": {"type": "LinearHead", "output_dim": output_dim},
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
        blocks = recipe.get("blocks", {})
        enc_type = blocks.get("encoder", "LinearProjection")
        enc_config = recipe.get("encoder_config", {})

        if enc_type not in valid_encoders and valid_encoders:
            enc_type = valid_encoders[0]

        d_model = 64
        n_features = profile.get("n_features", 1)
        if n_features > 20:
            d_model = 128
        elif n_features <= 7:
            d_model = max(32, d_model)

        evidence = f"encoder={enc_type}, n_features={n_features}"
        question = f"{evidence}. d_model={d_model} 제안 (16~256). 숫자."
        answer, comment = self._ask(question)
        d_model = self._parse_int(answer, default=d_model, min_val=16, max_val=256)

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

        # 후보 레시피 파싱
        candidates = []
        for m in re.finditer(r'\[(\d+)\]\s+(\S+)\s+\(capacity=(\w+)\)', task):
            candidates.append({
                "name": m.group(2),
                "capacity": m.group(3),
                "blocks": {},
            })
        result["candidate_recipes"] = candidates

        return result

    def _parse_prev_configs(self, task: str) -> list:
        configs = []
        for m in re.finditer(r'-\s*(.+?)\s*→\s*norm_MSE=([0-9.]+)', task):
            configs.append({"config": m.group(1), "norm_mse": float(m.group(2))})
        return configs

    def _parse_critic(self, task: str) -> dict:
        result: dict = {"extreme_ratio": 1.0, "verdict": ""}

        m = re.search(r'비율:\s*([0-9.]+)', task)
        if m:
            result["extreme_ratio"] = float(m.group(1))

        for v in ["RETRY_HP", "RETRY_RECIPE", "RETRY_BLOCK", "DONE"]:
            if v in task:
                result["verdict"] = v
                break

        return result
