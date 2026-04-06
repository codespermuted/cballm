"""Brain — 오케스트레이터 (v2.1). Hook + PromptBuilder + Compactor 통합.

v2.1 파이프라인:
  Scout → KG Matcher → [PreTrainHook → Architect → Engineer → Trainer → PostTrainHook → Critic] × N
                         ↑                                                    │
                         └──── PromptBuilder(compacted history + diagnostic) ←─┘
"""
from __future__ import annotations

import json
from pathlib import Path

from cballm.engine import chat
from cballm.session import WorkerSession
from cballm.workers.scout import Scout
from cballm.workers.kg_matcher import KGMatcher
from cballm.workers.architect import Architect
from cballm.features.engineer import Engineer
from cballm.workers.trainer import Trainer
from cballm.workers.critic import Critic
from cballm.hooks import (
    DiversityLedger, PreTrainHook, PostTrainHook,
    RoundRecord, HookResult,
)
from cballm.prompt_builder import ArchitectPromptBuilder
from cballm import display as D

MAX_ITERATIONS = 3

ORCHESTRATOR_PROMPT = """\
You are CBALLM Brain v2.1, an AI data science orchestrator.
You manage 6 specialist workers to build the best forecasting model.

Workers:
- Scout: rule-based data profiling and EDA
- KG Matcher: rule-based profile → KG query → candidate recipes/combinations
- Architect: LLM decision protocol (KG-bounded)
- Engineer: rule-based feature engineering + leakage verification
- Trainer: deterministic training execution
- Critic: rule-based evaluation + feedback routing

Hooks:
- PreTrainHook: diversity check + complexity validation (ALLOW/DENY/WARN)
- PostTrainHook: residual analysis + block attribution + diagnostic directive

Pipeline: Scout → KG → [PreHook → Architect → Engineer → Trainer → PostHook → Critic] × N

Critic feedback types:
- DONE: finalize results
- RETRY_HP: adjust hyperparameters (Architect Step 2-6)
- RETRY_RECIPE: try different recipe (Architect Step 1)
- RETRY_BLOCK: swap block via KG query
"""


def load_rules(rules_dir: str) -> dict[str, str]:
    """rules/ 디렉토리에서 모든 .md 파일을 로드."""
    rules_path = Path(rules_dir)
    if not rules_path.exists():
        return {}
    result = {}
    for f in sorted(rules_path.glob("*.md")):
        result[f.stem] = f.read_text().strip()
    return result


def extract_worker_rules(all_rules: dict[str, str], worker_name: str) -> str:
    """전체 룰에서 해당 워커에 관련된 섹션만 추출."""
    parts = []

    for name, content in all_rules.items():
        if name == "general":
            continue
        parts.append(content)

    general = all_rules.get("general", "")
    if general:
        section_map = {
            "scout": ["1. DATA PROFILING", "1.1", "1.2", "1.3"],
            "engineer": ["2. FEATURE ENGINEERING", "2.1", "2.2", "2.3", "2.4"],
            "architect": ["3. MODEL ARCHITECTURE", "3.1", "3.2", "3.3", "3.4", "3.5", "3.6", "3.7"],
            "trainer": ["4. TRAINING", "4.1", "4.2", "4.3", "4.4"],
            "critic": ["5. EVALUATION", "5.1", "5.2", "5.3", "5.4", "5.5"],
        }
        keywords = section_map.get(worker_name, [])
        if keywords:
            lines = general.split("\n")
            capturing = False
            section_lines = []
            for line in lines:
                if any(kw in line for kw in keywords):
                    capturing = True
                elif line.startswith("## ") and capturing:
                    if not any(kw in line for kw in keywords):
                        capturing = False
                if capturing:
                    section_lines.append(line)
            if section_lines:
                parts.append("\n".join(section_lines))

    return "\n\n".join(parts) if parts else ""


class Brain:
    """오케스트레이터 (v2.1).

    v2.1 파이프라인:
    Scout → KG → [PreHook → Architect → Engineer → Trainer → PostHook → Critic] × N
    """

    def __init__(self, cwd: str, rules_dir: str = "", benchmark_mode: bool = False):
        self.cwd = cwd
        self.benchmark_mode = benchmark_mode
        self.all_rules = load_rules(rules_dir) if rules_dir else {}
        self.session = WorkerSession(
            worker_name="brain",
            system_prompt=ORCHESTRATOR_PROMPT,
        )
        self.log: list[dict] = []
        self.ledger = DiversityLedger()

    def _rules_for(self, worker_name: str) -> str:
        return extract_worker_rules(self.all_rules, worker_name)

    def run_pipeline(self, data_path: str, target_col: str = "target",
                     prediction_length: int = 24, user_instructions: str = "") -> dict:
        """v2.1 파이프라인 실행."""
        D.print_banner()
        D.print_pipeline_start(data_path, target_col, prediction_length, user_instructions)

        context: dict = {
            "data_path": data_path,
            "target_col": target_col,
            "prediction_length": prediction_length,
            "user_instructions": user_instructions,
        }

        # 공통 계약
        data_stem = Path(data_path).stem
        feature_path = f"{self.cwd}/benchmark_data/{data_stem}_features.parquet"
        results_path = f"{self.cwd}/benchmark_data/{data_stem}_results.json"

        contract = (
            f"[Contract]\n"
            f"DATA_PATH = '{data_path}'\n"
            f"TARGET_COL = '{target_col}'\n"
            f"PREDICTION_LENGTH = {prediction_length}\n"
            f"FEATURE_OUTPUT_PATH = '{feature_path}'\n"
            f"RESULTS_OUTPUT_PATH = '{results_path}'\n"
        )

        # ── Step 1: Scout (rule-based) ──
        print(D.section_header("Data Profiling", "Scout"))
        scout = Scout(self.cwd)
        scout_result = scout.run(f"{contract}\n")
        self.log.append(scout_result)
        context["profile"] = scout_result.get("execution_result") or scout_result["response"]
        D.print_scout_result(context["profile"])

        # ── Step 2: KG Matcher (rule-based) ──
        print(D.section_header("Knowledge Graph", "KG"))
        kg_matcher = KGMatcher(self.cwd)
        kg_task = (
            f"{contract}\n"
            f"Scout Profile:\n{context['profile']}\n"
        )
        kg_result = kg_matcher.run(kg_task)
        self.log.append(kg_result)
        context["kg_match"] = kg_result.get("execution_result") or kg_result["response"]
        D.print_kg_result(context["kg_match"])

        # ── PromptBuilder 초기화 ──
        prompt_builder = ArchitectPromptBuilder(
            profile=context["profile"],
            kg_match=context["kg_match"],
            prediction_length=prediction_length,
        )

        # Scout profile에서 구조화 데이터 추출 (hooks용)
        scout_profile_dict = self._extract_profile_dict(context["profile"])

        # ── Iteration Loop ──
        verdict = {}
        diagnosis = None
        retry_type = ""

        for iteration in range(1, MAX_ITERATIONS + 1):
            D.print_iteration_header(iteration, MAX_ITERATIONS)

            # ── Step 3: Architect (LLM Decision Protocol) ──
            # v2.1: PromptBuilder로 구조화된 프롬프트 생성
            architect_task = prompt_builder.build(
                ledger=self.ledger,
                diagnosis=diagnosis,
                verdict=verdict if verdict else None,
                retry_type=retry_type,
            )

            # KG Matcher 결과 전달 (슬롯별 추천 포함)
            architect_task += f"\n\nKG Matcher Result:\n{context['kg_match']}\n"
            architect_task += f"PREDICTION_LENGTH = {prediction_length}\n"

            # RETRY 정보 전달
            if retry_type:
                architect_task += f"RETRY_TYPE = {retry_type}\n"

            # 이전 config 전달 (RETRY_HP 시 슬롯 유지용)
            if retry_type == "RETRY_HP" and self.ledger.rounds:
                prev = self.ledger.rounds[-1].config
                architect_task += f"PREV_CONFIG={json.dumps(prev, ensure_ascii=False)}\n"

            # blacklist 전달 (Diagnosis에서 ineffective 판정된 블록)
            if diagnosis and diagnosis.diagnosis:
                bl = diagnosis.diagnosis.block_blacklist
                if bl:
                    architect_task += f"Blacklist: {', '.join(bl)}\n"

            print(D.section_header("Architect", "LLM"))
            architect = Architect(self.cwd)
            architect_result = architect.run(architect_task)
            self.log.append(architect_result)

            config_json = architect_result["response"]
            context["config"] = config_json

            D.print_architect_decisions(
                architect_result.get("execution_result", ""),
                config_json,
            )

            # Config 파싱
            try:
                config = json.loads(config_json)
            except json.JSONDecodeError:
                config = {}

            # ── PreTrainHook: 사전 검증 ──
            pre_verdict = PreTrainHook.check(config, scout_profile_dict, self.ledger)
            if pre_verdict.exit_code == HookResult.DENY:
                print(f"  PreTrainHook: DENY — {pre_verdict.reason}")
                # 자동 대체: 미시도 레시피 중 다음 순위 선택 (LLM 재호출 없음)
                from cballm.recipes.registry import find_recipes_for_profile
                tried_names = {r.config.get("_recipe_name", "") for r in self.ledger.rounds}
                tried_names.add(config.get("_recipe_name", ""))
                candidates = find_recipes_for_profile(scout_profile_dict)
                fallback_found = False
                for cand in candidates:
                    if cand["name"] not in tried_names:
                        print(f"  -> 자동 대체: {cand['name']}")
                        # 대체 레시피로 config 재구성 (최소 설정)
                        blocks = cand.get("blocks", {})
                        fallback_d_model = cand.get("d_model", 64)
                        config = {
                            "normalizer": {"type": blocks.get("normalizer", "RevIN"), "affine": True},
                            "encoder": {"type": blocks.get("encoder", "LinearProjection"), "d_model": fallback_d_model},
                            "temporal_mixer": {"type": blocks.get("temporal_mixer", "LinearMix")},
                            "channel_mixer": None if not blocks.get("channel_mixer") else {"type": blocks["channel_mixer"]},
                            "head": {"type": "LinearHead", "output_dim": 1},
                            "constraint": [],
                            "loss": {"type": "MAE"},
                            "_recipe_name": cand["name"],
                        }
                        # encoder_config 적용
                        enc_cfg = cand.get("encoder_config", {})
                        if enc_cfg:
                            config["encoder"].update(enc_cfg)
                        # temporal_mixer_config 적용
                        mix_cfg = cand.get("temporal_mixer_config", {})
                        if mix_cfg:
                            if isinstance(config["temporal_mixer"], dict):
                                config["temporal_mixer"].update(mix_cfg)
                            else:
                                config["temporal_mixer"] = {"type": config["temporal_mixer"], **mix_cfg}
                        config_json = json.dumps(config, ensure_ascii=False)
                        fallback_found = True
                        # PreTrainHook 재검증
                        pre2 = PreTrainHook.check(config, scout_profile_dict, self.ledger)
                        if pre2.exit_code == HookResult.DENY:
                            tried_names.add(cand["name"])
                            fallback_found = False
                            continue
                        break

                if not fallback_found:
                    # 모든 레시피 소진 → DONE
                    print("  -> 모든 레시피 시도 완료")
                    if self.ledger.rounds:
                        best = self.ledger.best_round()
                        report = self._build_final_report(context, {
                            "verdict": "DONE",
                            "best_model": best.config.get("_recipe_name", "unknown") if best else "unknown",
                            "best_metric": best.metrics if best else {},
                            "analysis": "모든 레시피 시도 완료",
                        }, iteration)
                        D.print_final_report(report, {"verdict": "DONE"})
                        return report
                    verdict = {"verdict": "DONE", "best_metric": {}, "analysis": "No valid recipe"}
                    retry_type = "DONE"
                    continue

            elif pre_verdict.exit_code == HookResult.WARN:
                print(f"  PreTrainHook: WARN — {pre_verdict.reason}")

            # ── Step 4: Engineer (rule-based) ──
            engineer_task = (
                f"{contract}\n"
                f"Scout Profile:\n{context['profile']}\n"
            )
            engineer = Engineer(self.cwd)
            engineer_result = engineer.run(engineer_task)
            self.log.append(engineer_result)
            context["features"] = engineer_result.get("execution_result") or engineer_result["response"]

            # ── Step 5: Trainer (deterministic) ──
            # LLM 언로드 → GPU를 학습에 사용
            from cballm.engine import unload_local_model, _engine_type
            if _engine_type == "local":
                unload_local_model()

            # benchmark_mode: 원본 데이터만 사용 (논문 비교를 위해 feature engineering 미적용)
            if self.benchmark_mode:
                train_data_path = data_path
            else:
                train_data_path = feature_path if Path(feature_path).exists() else data_path

            trainer_contract = (
                f"DATA_PATH = '{train_data_path}'\n"
                f"TARGET_COL = '{target_col}'\n"
                f"PREDICTION_LENGTH = {prediction_length}\n"
            )
            trainer_task = (
                f"{trainer_contract}\n"
                f"Architect Config:\n{config_json}\n"
            )

            print(D.section_header("Training", "CV" if not self.benchmark_mode else "Bench"))
            trainer = Trainer(self.cwd, benchmark_mode=self.benchmark_mode)
            trainer_result = trainer.run(trainer_task)
            self.log.append(trainer_result)

            context["training_result"] = trainer_result.get("execution_result") or trainer_result["response"]

            # ── PostTrainHook: 잔차 진단 + 블록 기여도 ──
            train_metrics = self._extract_metrics_from_result(context["training_result"])
            residuals = self._extract_residuals(trainer_result)

            # 분포 정보 추출 (DistributionFitChecker용)
            head_cfg = config.get("head", {})
            distribution = head_cfg.get("distribution") if isinstance(head_cfg, dict) else None

            # Trainer 반환 확장 데이터 추출
            train_loss_history = trainer_result.get("train_loss_history", [])
            val_loss_history = trainer_result.get("val_loss_history", [])
            val_mae_by_step = trainer_result.get("val_mae_by_step", [])
            val_predictions = trainer_result.get("val_predictions")
            trainer_best_epoch = trainer_result.get("best_epoch", 0)
            trainer_max_epochs = trainer_result.get("max_epochs", 100)

            import logging
            logging.debug(
                f"PostTrainHook inputs: best_epoch={trainer_best_epoch}, "
                f"max_epochs={trainer_max_epochs}, "
                f"train_history_len={len(train_loss_history)}, "
                f"val_history_len={len(val_loss_history)}"
            )

            diagnosis = PostTrainHook.diagnose(
                config=config,
                metrics=train_metrics,
                profile=scout_profile_dict,
                ledger=self.ledger,
                residuals=residuals,
                pred_len=prediction_length,
                distribution=distribution,
                train_loss_history=train_loss_history,
                val_loss_history=val_loss_history,
                val_mae_by_step=val_mae_by_step,
                val_predictions=val_predictions,
                best_epoch=trainer_best_epoch,
                max_epochs=trainer_max_epochs,
            )

            if diagnosis.directive != "진단 이상 없음":
                print(f"  PostTrainHook: {diagnosis.directive[:100]}")

            # ── Step 6: Critic (rule-based) ──
            critic_task = (
                f"Training result:\n{context['training_result'][:2000]}\n\n"
                f"Iteration: {iteration}/{MAX_ITERATIONS}\n"
            )

            prev_mae = context.get("prev_mae")
            prev_norm_mse = context.get("prev_norm_mse")
            critic = Critic(self.cwd, prev_mae=prev_mae, prev_norm_mse=prev_norm_mse)
            critic_result = critic.run(critic_task)
            self.log.append(critic_result)

            verdict = json.loads(critic_result["response"])

            # ── INVALID_TRAINING 감지 ──
            # best_epoch < max_epochs * 0.05 → 학습 실패 판정
            is_valid_training = True
            invalid_threshold = max(1, int(trainer_max_epochs * 0.05))
            if trainer_best_epoch < invalid_threshold:
                is_valid_training = False
                recipe_name = config.get("_recipe_name", "unknown")
                print(f"  INVALID_TRAINING: best_epoch={trainer_best_epoch}/{trainer_max_epochs} "
                      f"(< {invalid_threshold}), recipe={recipe_name}")

                # 같은 recipe 연속 실패 카운트
                recipe_fail_key = f"_invalid_{recipe_name}"
                context[recipe_fail_key] = context.get(recipe_fail_key, 0) + 1

                if context[recipe_fail_key] >= 2:
                    # 2회 연속 → recipe 포기, 다음 recipe
                    print(f"  -> {recipe_name} 2회 연속 학습 실패, recipe 포기")
                    verdict["verdict"] = "RETRY_RECIPE"
                    verdict["analysis"] = f"{recipe_name} 학습 2회 연속 실패"
                else:
                    # 1회 → RETRY_HP (lr 낮추고 warmup 추가)
                    verdict["verdict"] = "RETRY_HP"
                    verdict["suggestions"] = [
                        f"lr *= 0.5, warmup += 5, d_model = recipe recommended",
                        f"best_epoch={trainer_best_epoch} < threshold={invalid_threshold}",
                    ]

            D.print_critic_verdict(
                verdict["verdict"],
                verdict.get("best_metric", {}).get("MAE", "N/A"),
                verdict.get("analysis", ""),
                verdict.get("suggestions", []),
            )

            # ── Ledger에 라운드 기록 ──
            config_hash = DiversityLedger.config_hash(config)
            round_record = RoundRecord(
                round_num=iteration,
                config=config,
                config_hash=config_hash,
                metrics=verdict.get("best_metric", {}),
                verdict=verdict.get("verdict", ""),
                diagnosis=diagnosis,
            )
            round_record._valid_training = is_valid_training
            self.ledger.add_round(round_record)

            # MAE/norm_MSE 저장 — 유효 학습만 비교 대상으로 갱신
            current_mae = verdict.get("best_metric", {}).get("MAE")
            current_norm_mse = verdict.get("best_metric", {}).get("norm_MSE")
            if is_valid_training and current_mae is not None:
                context["prev_mae"] = current_mae
            if is_valid_training and current_norm_mse is not None:
                context["prev_norm_mse"] = current_norm_mse

            # ── 분기 판단 ──
            v = verdict.get("verdict", "")
            retry_type = v

            # DONE이지만 유효 학습이 없으면 계속 시도
            if v == "DONE" and not is_valid_training and iteration < MAX_ITERATIONS:
                v = "RETRY_RECIPE"
                retry_type = "RETRY_RECIPE"

            if v == "DONE" or iteration == MAX_ITERATIONS:
                report = self._build_final_report(context, verdict, iteration)
                D.print_final_report(report, verdict)
                return report

            # RETRY_BLOCK: KG에 대체 블록 쿼리
            if v == "RETRY_BLOCK":
                kg_task_retry = (
                    f"{contract}\n"
                    f"Scout Profile:\n{context['profile']}\n"
                    f"Current config failed blocks:\n{config_json}\n"
                    f"Diagnostic: {diagnosis.directive}\n"
                    f"Critic: {verdict.get('analysis', '')}\n"
                )
                kg_retry_result = kg_matcher.run(kg_task_retry)
                context["kg_match"] = kg_retry_result.get("execution_result") or kg_retry_result["response"]
                self.log.append(kg_retry_result)
                # PromptBuilder 갱신
                prompt_builder = ArchitectPromptBuilder(
                    profile=context["profile"],
                    kg_match=context["kg_match"],
                    prediction_length=prediction_length,
                )

        report = self._build_final_report(context, verdict, MAX_ITERATIONS)
        D.print_final_report(report, verdict)
        return report

    def _build_final_report(self, context: dict, verdict: dict, iterations: int) -> dict:
        # 최고 라운드 — 유효 학습 모델 중에서만
        valid_rounds = [r for r in self.ledger.rounds if getattr(r, '_valid_training', True)]
        best = None
        for r in valid_rounds:
            nm = r.metrics.get("norm_MSE")
            if nm is not None and (best is None or nm < best.metrics.get("norm_MSE", float("inf"))):
                best = r
        if best is None:
            best = self.ledger.best_round()  # fallback
        best_round_info = None
        if best:
            best_round_info = {
                "round": best.round_num,
                "config_summary": best.compact_summary(),
            }

        # Decision Trace — "왜 이 블록인가"의 전체 경로
        decision_trace = self._build_decision_trace(context)

        # best round의 정보를 최종 메트릭으로 사용
        best_metrics = best.metrics if best else verdict.get("best_metric", {})
        best_model_name = best.config.get("_recipe_name", verdict.get("best_model", "unknown")) if best else verdict.get("best_model", "unknown")

        return {
            "status": "completed",
            "iterations": iterations,
            "best_model": best_model_name,
            "metrics": best_metrics,
            "analysis": verdict.get("analysis", ""),
            "profile_summary": context.get("profile", "")[:500],
            "features_used": context.get("features", "")[:500],
            "config_used": context.get("config", "")[:500],
            "best_round": best_round_info,
            "round_history": [r.compact_summary() for r in self.ledger.rounds],
            "decision_trace": decision_trace,
            "log": self.log,
        }

    def _build_decision_trace(self, context: dict) -> list[dict]:
        """Decision Trace — 각 라운드의 의사결정 경로를 추적.

        추적 경로:
          Scout Profile → KG Rule Match → Architect 선택 → PreHook 검증
          → Trainer 결과 → PostHook 진단 → Critic 판정

        각 단계에서 남기는 메타데이터:
          - scout:     데이터 특성 요약 (n_rows, seasonality, regime)
          - kg:        매칭된 규칙 ID, 후보 레시피 목록
          - architect: 각 step의 (evidence, default, LLM answer, decision)
          - prehook:   ALLOW/DENY/WARN + 이유
          - trainer:   메트릭 (MAE, norm_MSE, best_epoch)
          - posthook:  Diagnosis (failure_mode, residual_pattern, direction)
          - critic:    verdict + suggestions
        """
        traces = []

        for record in self.ledger.rounds:
            trace: dict = {
                "round": record.round_num,
                "blocks": {},
                "metrics": record.metrics,
                "verdict": record.verdict,
            }

            # 블록 선택 근거
            config = record.config
            for slot in ["normalizer", "encoder", "temporal_mixer",
                         "channel_mixer", "head", "loss"]:
                val = config.get(slot)
                if val is None:
                    trace["blocks"][slot] = None
                elif isinstance(val, dict):
                    trace["blocks"][slot] = val.get("type")
                else:
                    trace["blocks"][slot] = str(val)

            # PostHook 진단
            if record.diagnosis:
                diag = record.diagnosis.diagnosis
                trace["diagnosis"] = {
                    "failure_mode": diag.failure_mode,
                    "residual_pattern": diag.residual_pattern,
                    "underperforming_regime": diag.underperforming_regime,
                    "suggested_direction": diag.suggested_direction,
                    "block_blacklist": diag.block_blacklist,
                }

                # 블록별 기여도
                trace["block_attributions"] = {
                    a.slot: {"block": a.block_name, "verdict": a.verdict,
                             "contribution": a.contribution}
                    for a in record.diagnosis.attributions
                }

            traces.append(trace)

        return traces

    @staticmethod
    def _extract_profile_dict(profile_text: str) -> dict:
        """Scout 프로파일 텍스트에서 hooks용 dict 추출."""
        import re
        d: dict = {}

        m = re.search(r'Shape:\s*\((\d+),\s*(\d+)\)', profile_text)
        if m:
            d["n_rows"] = int(m.group(1))
            d["n_features"] = int(m.group(2))

        d["is_stationary"] = "non-stationary" not in profile_text

        acf_vals = re.findall(r'ACF=([0-9.]+)', profile_text)
        d["max_acf_at_known_periods"] = max((float(v) for v in acf_vals), default=0.0)

        corr_match = re.search(r'high.corr.*?(\d+)\s*pairs?', profile_text, re.IGNORECASE)
        d["high_cross_corr_pairs"] = int(corr_match.group(1)) if corr_match else 0

        return d

    @staticmethod
    def _extract_metrics_from_result(result_text: str) -> dict:
        """Trainer 결과 텍스트에서 메트릭 dict 추출."""
        import re
        metrics: dict = {}
        for name in ["MAE", "MSE", "RMSE", "norm_MSE"]:
            m = re.search(rf'{name}[:\s=]+([0-9]+\.?[0-9]*)', result_text)
            if m:
                metrics[name] = float(m.group(1))
        # val MAE
        m = re.search(r'val.MAE[:\s=]+([0-9]+\.?[0-9]*)', result_text, re.IGNORECASE)
        if m:
            metrics["val_MAE"] = float(m.group(1))
        return metrics

    @staticmethod
    def _extract_residuals(trainer_result: dict):
        """Trainer 결과에서 val 잔차 벡터 추출.

        Trainer.run()이 반환하는 response JSON에서 val_residuals를 파싱하거나,
        직접 TrainResult 객체에서 추출.

        중요: val 잔차만 사용. test 잔차는 루프에 유입 금지.
        """
        import numpy as np
        # trainer_result는 dict (worker output)
        # val_residuals는 TrainResult에 저장되어 있으나,
        # worker output에서는 JSON response만 전달됨.
        # → Trainer.run()에서 val_residuals를 별도 필드로 전달하도록 연결
        residuals = trainer_result.get("val_residuals")
        if residuals is not None:
            if isinstance(residuals, np.ndarray):
                return residuals
            # list로 온 경우
            return np.array(residuals, dtype=np.float32)
        return None
