"""Architect — Decision Protocol. 파이프라인 전체의 결정 지점을 순차적으로 판단.

코드가 evidence를 제공하고 기본값을 제안 → LLM이 확인/조정.
Q&A 기록이 곧 모델링 리포트.
"""
from __future__ import annotations

import json
import re

from cballm.engine import chat
from cballm.session import WorkerSession


class Architect:
    """Decision Protocol로 파이프라인 전체 설정을 결정한다."""
    name = "architect"
    description = "Decision Protocol 기반 파이프라인 설계"
    model_profile = "reasoning"

    SYSTEM_PROMPT = (
        "You are a time series modeling advisor. "
        "For each question, answer the CLOSED part first (number or yes/no), "
        "then optionally add a SHORT comment (max 15 words) if you have an insight. "
        "Format: ANSWER | comment (optional)"
    )

    def __init__(self, cwd: str = "", rules: str = ""):
        self.cwd = cwd
        self.session = WorkerSession(worker_name=self.name, system_prompt=self.SYSTEM_PROMPT)
        self.decisions: list[dict] = []

    def run(self, task: str) -> dict:
        """Scout 프로파일 + Critic 피드백을 기반으로 전체 파이프라인 설정 결정."""
        profile = self._parse_profile(task)
        prev_configs = self._parse_prev_configs(task)
        critic = self._parse_critic(task)

        config = self._run_protocol(profile, prev_configs, critic)

        # 모델링 리포트 생성
        report_lines = ["=== MODELING DECISIONS ==="]
        for d in self.decisions:
            report_lines.append(f"Step {d['step']}: {d['topic']}")
            report_lines.append(f"  Evidence: {d['evidence']}")
            report_lines.append(f"  Default: {d['default']}")
            report_lines.append(f"  LLM: {d['answer']}")
            if d.get("comment"):
                report_lines.append(f"  Insight: {d['comment']}")
            report_lines.append(f"  → Decision: {d['decision']}")
            report_lines.append("")
        report_lines.append(f"Config: {json.dumps(config, ensure_ascii=False)}")

        config_json = json.dumps(config, ensure_ascii=False)

        return {
            "worker": self.name,
            "response": config_json,
            "code": None,
            "execution_result": "\n".join(report_lines),
        }

    def _run_protocol(self, profile: dict, prev_configs: list, critic: dict) -> dict:
        """6-Step Decision Protocol."""
        self.decisions = []

        preprocessing = self._step0_preprocessing(profile)
        input_design = self._step1_input_design(profile, prev_configs)
        encoder = self._step2_encoder(profile)
        backbone = self._step3_backbone(profile, prev_configs)
        constraint, loss = self._step4_constraint_loss(profile, critic)
        training = self._step5_training(profile, prev_configs)
        regime = self._step6_regime(profile)

        return {
            # 블록 config
            "encoder": encoder,
            "backbone": backbone,
            "regime": regime,
            "constraint": constraint,
            "loss": loss,
            # 파이프라인 config (Trainer가 사용)
            "preprocessing": preprocessing,
            "input_design": input_design,
            "training": training,
        }

    # ── Step 0: Preprocessing ──

    def _step0_preprocessing(self, profile: dict) -> dict:
        skew = abs(profile.get("target_skew", 0))
        is_stationary = profile.get("is_stationary", True)
        target_min = profile.get("target_min", 0)

        # Log transform 판단
        if target_min > 0 and skew > 2.0:
            default = "log transform (skew 높고 양수)"
            evidence = f"target_min={target_min:.2f}, skew={skew:.2f}"
            question = f"{evidence}. Log transform 제안. 동의? (yes/no)"
            answer, comment = self._ask(question)
            log_transform = "no" not in answer.lower()
        else:
            log_transform = False
            evidence = f"skew={skew:.2f}, min={target_min:.2f}"
            default = "없음"
            answer, comment = "auto", ""

        # Differencing 판단
        if not is_stationary:
            diff_default = "1차 차분 (non-stationary)"
            diff_evidence = f"ADF p={profile.get('adf_p', 'N/A')}"
            diff_q = f"Non-stationary ({diff_evidence}). 차분 적용? (yes/no)"
            diff_answer, diff_comment = self._ask(diff_q)
            differencing = "no" not in diff_answer.lower()
        else:
            differencing = False
            diff_evidence = "stationary"
            diff_default = "없음"
            diff_answer, diff_comment = "auto", ""

        self._record("0", "전처리",
                     f"log: {evidence}, diff: {diff_evidence}",
                     f"log={default}, diff={diff_default}",
                     f"log={answer}, diff={diff_answer}",
                     f"log_transform={log_transform}, differencing={differencing}",
                     comment or diff_comment)

        return {"log_transform": log_transform, "differencing": differencing}

    # ── Step 1: Input Design ──

    def _step1_input_design(self, profile: dict, prev_configs: list) -> dict:
        n_rows = profile.get("n_rows", 0)
        dominant = profile.get("dominant_period", 24)
        seasonality = profile.get("seasonality", {})

        # seq_len 결정: 최소 dominant period × 2, 최대 데이터의 10%
        if dominant:
            default_seq = max(96, dominant * 2)
        else:
            default_seq = 96
        default_seq = min(default_seq, n_rows // 10)

        pred_len = profile.get("pred_len", 96)
        evidence = f"dominant_period={dominant}, n_rows={n_rows}, pred_len={pred_len}"
        question = f"{evidence}. seq_len={default_seq} 제안 (보통 pred_len={pred_len} 이상). 숫자 ({pred_len}~512)."
        answer, comment = self._ask(question)
        pred_len = profile.get("pred_len", 96)
        seq_len = self._parse_int(answer, default=default_seq, min_val=pred_len, max_val=512)

        self._record("1", "입력 설계", evidence, f"seq_len={default_seq}",
                     answer, f"seq_len={seq_len}", comment)

        return {"seq_len": seq_len}

    # ── Step 2: Encoder ──

    def _step2_encoder(self, profile: dict) -> dict:
        seasonality = profile.get("seasonality", {})
        strong_periods = [(k, v) for k, v in seasonality.items() if v > 0.7]

        if strong_periods:
            n_default = len(strong_periods)
            evidence = ", ".join(f"{k} ACF={v:.2f}" for k, v in strong_periods)
            question = f"강한 주기: {evidence}. Fourier harmonics={n_default} 제안. 숫자 (1~10)."
            answer, comment = self._ask(question)
            n = self._parse_int(answer, default=n_default, min_val=1, max_val=10)
            self._record("2", "인코딩", evidence, f"Fourier({n_default})",
                        answer, f"Fourier({n})", comment)
            return {"type": "Fourier", "n_harmonics": n, "n_time_features": 0}
        else:
            max_acf = max(seasonality.values()) if seasonality else 0
            evidence = f"max ACF={max_acf:.2f}"
            self._record("2", "인코딩", evidence, "Linear", "auto", "Linear", "")
            return {"type": "Linear"}

    # ── Step 3: Backbone ──

    def _step3_backbone(self, profile: dict, prev_configs: list) -> dict:
        prev_linear_mse = None
        for p in prev_configs:
            if "Linear" in str(p.get("config", "")):
                prev_linear_mse = p.get("norm_mse")

        if prev_linear_mse is not None and prev_linear_mse < 0.15:
            evidence = f"이전 Linear norm_MSE={prev_linear_mse}"
            default = "Linear 유지 (이전 좋았음)"
            question = f"{evidence}. Linear 유지? PatchMLP 시도하려면 'PatchMLP'."
            answer, comment = self._ask(question)
        elif prev_linear_mse is not None and prev_linear_mse > 0.3:
            evidence = f"이전 Linear norm_MSE={prev_linear_mse} (부족)"
            default = "PatchMLP (Linear ceiling)"
            question = f"{evidence}. PatchMLP 시도 제안. 동의?"
            answer, comment = self._ask(question)
        else:
            n_rows = profile.get("n_rows", 0)
            regime = "stable" if profile.get("regime_stable", True) else "unstable"
            evidence = f"n_rows={n_rows}, regime={regime}"
            default = "Linear (첫 시도)"
            question = f"{evidence}. Linear backbone 제안. 동의?"
            answer, comment = self._ask(question)

        if "patch" in answer.lower():
            self._record("3", "backbone", evidence, default, answer, "PatchMLP", comment)
            return {"type": "PatchMLP", "patch_len": 16, "stride": 8, "hidden_dim": 128}
        else:
            self._record("3", "backbone", evidence, default, answer, "Linear", comment)
            return {"type": "Linear"}

    # ── Step 4: Constraint + Loss ──

    def _step4_constraint_loss(self, profile: dict, critic: dict) -> tuple[list, dict]:
        constraints = []

        # Constraint
        can_neg = profile.get("can_be_negative", True)
        t_min = profile.get("target_min", -999)
        if not can_neg and t_min >= 0:
            evidence = f"target always ≥ 0 (min={t_min:.2f})"
            question = f"{evidence}. Positivity constraint 제안. 동의? (yes/no)"
            answer, comment = self._ask(question)
            if "no" not in answer.lower():
                constraints.append({"type": "Positivity"})
            self._record("4a", "제약", evidence, "Positivity", answer,
                        "Positivity" if constraints else "없음", comment)
        else:
            self._record("4a", "제약", f"can_negative={can_neg}", "없음", "auto", "없음", "")

        # Loss
        extreme_ratio = critic.get("extreme_ratio", 1.0)
        skew = abs(profile.get("target_skew", 0))

        if extreme_ratio > 2.0:
            evidence = f"extreme/normal MAE 비율={extreme_ratio:.1f}"
            question = f"{evidence}. Asymmetric loss(under_weight=2.0) 제안. under_weight? (숫자 1~5)"
            answer, comment = self._ask(question)
            w = self._parse_float(answer, default=2.0, min_val=1.0, max_val=5.0)
            loss = {"type": "Asymmetric", "under_weight": w, "over_weight": 1.0}
            self._record("4b", "손실", evidence, f"Asymmetric(uw=2.0)", answer,
                        f"Asymmetric(uw={w})", comment)
        elif skew > 1.5:
            evidence = f"skew={skew:.2f}"
            question = f"{evidence}. Huber loss 제안. 동의? MAE 선호하면 'MAE'."
            answer, comment = self._ask(question)
            if "mae" in answer.lower():
                loss = {"type": "MAE"}
                self._record("4b", "손실", evidence, "Huber", answer, "MAE", comment)
            else:
                loss = {"type": "Huber"}
                self._record("4b", "손실", evidence, "Huber", answer, "Huber", comment)
        else:
            loss = {"type": "MAE"}
            self._record("4b", "손실", f"skew={skew:.2f}, ratio={extreme_ratio:.1f}",
                        "MAE", "auto", "MAE", "")

        return constraints, loss

    # ── Step 5: Training Strategy ──

    def _step5_training(self, profile: dict, prev_configs: list) -> dict:
        n_rows = profile.get("n_rows", 0)

        # Fold 수
        if n_rows < 5000:
            default_folds = 2
        elif n_rows < 20000:
            default_folds = 3
        else:
            default_folds = 5

        evidence = f"n_rows={n_rows}"
        question = f"{evidence}. CV fold={default_folds} 제안 (3~5가 일반적, 많으면 학습 시간 증가). 숫자 (2~5)."
        answer, comment = self._ask(question)
        n_folds = self._parse_int(answer, default=default_folds, min_val=2, max_val=5)

        self._record("5", "학습 전략", evidence, f"folds={default_folds}",
                     answer, f"folds={n_folds}", comment)

        return {"n_folds": n_folds}

    # ── Step 6: Regime ──

    def _step6_regime(self, profile: dict) -> dict | None:
        regime_stable = profile.get("regime_stable", True)
        n_changes = profile.get("n_regime_changes", 0)

        if not regime_stable and n_changes >= 3:
            evidence = f"{n_changes} regime changes"
            question = f"Regime unstable ({evidence}). SoftGate(2) 제안. 동의? (yes/no)"
            answer, comment = self._ask(question)
            if "no" in answer.lower():
                self._record("6", "regime", evidence, "SoftGate(2)", answer, "없음", comment)
                return None
            n = self._parse_int(answer, default=2, min_val=2, max_val=3)
            self._record("6", "regime", evidence, "SoftGate(2)", answer,
                        f"SoftGate({n})", comment)
            return {"type": "SoftGate", "n_regimes": n}
        else:
            self._record("6", "regime", f"stable (changes={n_changes})", "없음", "auto", "없음", "")
            return None

    # ── Helpers ──

    def _ask(self, question: str) -> tuple[str, str]:
        """LLM에 질문. (answer, comment) 반환."""
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

            # "ANSWER | comment" 파싱
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
        profile = {}
        m = re.search(r'\((\d+),\s*(\d+)\)', task)
        if m:
            profile["n_rows"], profile["n_cols"] = int(m.group(1)), int(m.group(2))

        seasonality = {}
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

        pred_m = re.search(r'예측 길이:\s*(\d+)', task)
        if pred_m:
            profile["pred_len"] = int(pred_m.group(1))
        else:
            pred_m = re.search(r'PREDICTION_LENGTH\s*=\s*(\d+)', task)
            if pred_m:
                profile["pred_len"] = int(pred_m.group(1))

        regime_m = re.search(r'Regime:\s*(stable|unstable)\s*(?:\((\d+)\s*change)?', task)
        if regime_m:
            profile["regime_stable"] = regime_m.group(1) == "stable"
            profile["n_regime_changes"] = int(regime_m.group(2)) if regime_m.group(2) else 0
        else:
            profile["regime_stable"] = True
            profile["n_regime_changes"] = 0

        return profile

    def _parse_prev_configs(self, task: str) -> list:
        configs = []
        for m in re.finditer(r'-\s*(.+?)\s*→\s*norm_MSE=([0-9.]+)', task):
            configs.append({"config": m.group(1), "norm_mse": float(m.group(2))})
        return configs

    def _parse_critic(self, task: str) -> dict:
        result = {"extreme_ratio": 1.0}
        m = re.search(r'비율:\s*([0-9.]+)', task)
        if m:
            result["extreme_ratio"] = float(m.group(1))
        return result
