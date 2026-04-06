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


        # ── Iteration Loop (v4: multi-candidate + direct config) ──
        from cballm.exploration import ExplorationBudget, CandidateGenerator
        from cballm.composite_score import compute_composite_score
        from cballm.synergy import SynergyChecker

        verdict = {}
        diagnosis = None
        retry_type = ""
        composite_scores: list[float] = []

        # KG slot recommendations 파싱
        slot_recs = self._parse_slot_recs(context["kg_match"])

        for iteration in range(1, MAX_ITERATIONS + 1):
            D.print_iteration_header(iteration, MAX_ITERATIONS)

            # ── 1. ExplorationBudget ──
            budget = ExplorationBudget.decide(iteration, composite_scores)
            print(f"  Exploration: n={budget.n_candidates}, strategy={budget.strategy}")

            # 2연속 score plateau → DONE
            if (len(composite_scores) >= 2 and budget.n_candidates == 1
                    and composite_scores[-1] > 0
                    and abs(composite_scores[-1] - composite_scores[-2]) / max(composite_scores[-1], 0.001) < 0.01):
                best = self.ledger.best_round()
                report = self._build_final_report(context, {
                    "verdict": "DONE", "best_metric": best.metrics if best else {},
                    "analysis": "score plateau",
                }, iteration)
                D.print_final_report(report, {"verdict": "DONE"})
                return report

            # ── 2. 후보 생성 ──
            candidates = CandidateGenerator.generate(
                slot_recs=slot_recs, diagnosis=diagnosis,
                ledger=self.ledger, budget=budget, profile=scout_profile_dict,
            )
            # 중복 제거 + SynergyChecker
            seen = set()
            checked = []
            for c in candidates:
                sr = SynergyChecker.validate(c.config, scout_profile_dict)
                if sr.applied_rules:
                    print(f"  Synergy: {', '.join(sr.applied_rules)}")
                c.config = sr.corrected
                h = DiversityLedger.config_hash(c.config)
                if h not in seen:
                    seen.add(h)
                    checked.append(c)
            candidates = checked

            for i, c in enumerate(candidates):
                enc = c.config.get("encoder", {}).get("type", "?") if isinstance(c.config.get("encoder"), dict) else "?"
                mix = c.config.get("temporal_mixer", {}).get("type", "?") if isinstance(c.config.get("temporal_mixer"), dict) else "?"
                print(f"  [{chr(97+i)}] {c.label}: {enc}+{mix}")

            # ── 3. LLM 순위 (R2+, n>1) ──
            from cballm.engine import unload_local_model, _engine_type
            if len(candidates) > 1 and iteration > 1:
                candidates = self._rank_candidates_llm(candidates)
            elif len(candidates) > 1:
                # R1: exploit first (baseline 확보)
                exploit_first = [c for c in candidates if c.label == "exploit"]
                others = [c for c in candidates if c.label != "exploit"]
                candidates = exploit_first + others
            if _engine_type == "local":
                unload_local_model()

            # ── 4. 순차 학습 (config_dict 직접 전달) ──
            train_data_path = data_path if self.benchmark_mode else (
                feature_path if Path(feature_path).exists() else data_path)
            val_sub = "A" if iteration % 2 == 1 else "B"
            round_best = None  # (mae, config, trainer_result, diag, metrics)

            for ci, cand in enumerate(candidates):
                config = cand.config

                # PreTrainHook
                pre = PreTrainHook.check(config, scout_profile_dict, self.ledger)
                if pre.exit_code == HookResult.DENY:
                    print(f"    [{chr(97+ci)}] DENY: {pre.reason}")
                    continue

                # Train — config_dict 직접 전달 (텍스트 파싱 우회)
                trainer_task = (
                    f"DATA_PATH = '{train_data_path}'\n"
                    f"TARGET_COL = '{target_col}'\n"
                    f"PREDICTION_LENGTH = {prediction_length}\n"
                )
                tag = f"val_{val_sub}" if self.benchmark_mode else "CV"
                print(f"    [{chr(97+ci)}] Training ({tag})...")
                trainer = Trainer(self.cwd, benchmark_mode=self.benchmark_mode, val_subset=val_sub)
                trainer_result = trainer.run(trainer_task, config_dict=config)
                self.log.append(trainer_result)

                # Metrics
                result_text = trainer_result.get("execution_result") or trainer_result["response"]
                metrics = self._extract_metrics_from_result(result_text)
                mae = metrics.get("MAE")
                best_epoch = trainer_result.get("best_epoch", 0)
                max_epochs = trainer_result.get("max_epochs", 50)

                # INVALID_TRAINING
                inv_th = max(1, int(max_epochs * 0.05))
                if best_epoch < inv_th:
                    print(f"    [{chr(97+ci)}] INVALID (epoch {best_epoch}/{max_epochs})")
                    continue

                # Early Abort
                if round_best and mae and round_best[0] and mae > round_best[0] * 1.5:
                    print(f"    [{chr(97+ci)}] Abort: {mae:.4f} >> {round_best[0]:.4f}")
                    continue

                # PostTrainHook
                residuals = self._extract_residuals(trainer_result)
                head_cfg = config.get("head", {})
                dist = head_cfg.get("distribution") if isinstance(head_cfg, dict) else None
                cand_diag = PostTrainHook.diagnose(
                    config=config, metrics=metrics, profile=scout_profile_dict,
                    ledger=self.ledger, residuals=residuals, pred_len=prediction_length,
                    distribution=dist,
                    train_loss_history=trainer_result.get("train_loss_history", []),
                    val_loss_history=trainer_result.get("val_loss_history", []),
                    val_mae_by_step=trainer_result.get("val_mae_by_step", []),
                    val_predictions=trainer_result.get("val_predictions"),
                    best_epoch=best_epoch, max_epochs=max_epochs,
                )

                norm_mse = metrics.get("norm_MSE", "?")
                print(f"    [{chr(97+ci)}] MAE={mae}, nMSE={norm_mse}")

                if round_best is None or (mae is not None and mae < round_best[0]):
                    round_best = (mae, config, trainer_result, cand_diag, metrics)

            # ── 5. 전부 실패 ──
            if round_best is None:
                print(f"  No valid candidate")
                verdict = {"verdict": "RETRY_RECIPE", "best_metric": {}}
                retry_type = "RETRY_RECIPE"
                diagnosis = None
                continue

            # ── 6. Best → Critic + CompositeScore ──
            best_mae, config, trainer_result, diagnosis, train_metrics = round_best
            config_json = json.dumps(config, ensure_ascii=False)
            context["config"] = config_json
            context["training_result"] = trainer_result.get("execution_result") or trainer_result["response"]

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

            D.print_critic_verdict(
                verdict["verdict"],
                verdict.get("best_metric", {}).get("MAE", "N/A"),
                verdict.get("analysis", ""),
                verdict.get("suggestions", []),
            )

            # CompositeScore
            current_mae = verdict.get("best_metric", {}).get("MAE")
            current_norm_mse = verdict.get("best_metric", {}).get("norm_MSE")
            baseline = context.get("naive_mae", (current_mae or 1.0) * 1.5)
            val_steps = trainer_result.get("val_mae_by_step", [])
            d_stab = diagnosis.disagreement.stability if diagnosis and diagnosis.disagreement else 1.0
            composite = compute_composite_score(
                val_mae=current_mae or 0, baseline_mae=baseline,
                val_mae_by_step=val_steps or None,
                residual_diagnosis=diagnosis.residual if diagnosis else None,
                disagreement_stability=d_stab,
            )
            composite_scores.append(composite.weighted_score())
            print(f"  CompositeScore: {composite.summary()}")

            # Ledger
            ch = DiversityLedger.config_hash(config)
            record = RoundRecord(
                round_num=iteration, config=config, config_hash=ch,
                metrics=verdict.get("best_metric", {}),
                verdict=verdict.get("verdict", ""), diagnosis=diagnosis,
            )
            record._valid_training = True
            record._composite_score = composite
            self.ledger.add_round(record)

            if current_mae is not None:
                context["prev_mae"] = current_mae
            if current_norm_mse is not None:
                context["prev_norm_mse"] = current_norm_mse

            # ── 7. 분기 ──
            v = verdict.get("verdict", "")
            retry_type = v
            if v == "DONE" or iteration == MAX_ITERATIONS:
                report = self._build_final_report(context, verdict, iteration)
                D.print_final_report(report, verdict)
                return report

        report = self._build_final_report(context, verdict, MAX_ITERATIONS)
        D.print_final_report(report, verdict)
        return report


    def _rank_candidates_llm(self, candidates) -> list:
        from cballm.engine import chat
        prompt = "Rank these time series model configs (best first):\n"
        for i, c in enumerate(candidates):
            enc = c.config.get("encoder", {}).get("type", "?") if isinstance(c.config.get("encoder"), dict) else "?"
            mix = c.config.get("temporal_mixer", {}).get("type", "?") if isinstance(c.config.get("temporal_mixer"), dict) else "?"
            prompt += f"({chr(97+i)}) {enc}+{mix} [{c.label}]\n"
        prompt += "Answer: ranking like 'a > b > c'"
        try:
            answer = chat("Rank configs. Letter ranking only.", [{"role": "user", "content": prompt}], max_tokens=20)
            print(f"  LLM rank: {answer.strip()}")
            order = []
            for ch in answer.lower():
                if ch.isalpha() and ord(ch) - ord('a') < len(candidates):
                    idx = ord(ch) - ord('a')
                    if idx not in order:
                        order.append(idx)
            for i in range(len(candidates)):
                if i not in order:
                    order.append(i)
            return [candidates[i] for i in order]
        except Exception:
            return candidates

    @staticmethod
    def _parse_slot_recs(kg_text: str) -> dict:
        import re
        slots = {}
        for m in re.finditer(r'SLOT_(\w+)=\{(.+?)\}', kg_text):
            slot = m.group(1).lower()
            content = m.group(2)
            rec = {}
            rm = re.search(r'recommended=([^,}]+)', content)
            if rm:
                rec["recommended"] = rm.group(1).strip().strip("'\"[]")
            om = re.search(r'options=\[([^\]]*)\]', content)
            if om:
                rec["options"] = [x.strip().strip("'\"") for x in om.group(1).split(",") if x.strip()]
            slots[slot] = rec
        dm = re.search(r'DATA_SCALE=\{(.+?)\}', kg_text)
        if dm:
            scale = {}
            for k in ["d_model", "n_layers", "n_heads"]:
                km = re.search(rf'{k}=(\d+)', dm.group(1))
                if km:
                    scale[k] = int(km.group(1))
            slots["_data_scale"] = scale
        return slots

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
