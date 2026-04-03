"""Brain — 오케스트레이터 (v2). KG Matcher 추가, 파이프라인 재구성."""
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

MAX_ITERATIONS = 3

ORCHESTRATOR_PROMPT = """\
You are CBALLM Brain v2, an AI data science orchestrator.
You manage 6 specialist workers to build the best forecasting model.

Workers:
- Scout: rule-based data profiling and EDA
- KG Matcher: rule-based profile → KG query → candidate recipes/combinations
- Architect: LLM decision protocol (KG-bounded)
- Engineer: rule-based feature engineering + leakage verification
- Trainer: deterministic training execution
- Critic: rule-based evaluation + feedback routing

Pipeline: Scout → KG Matcher → [Architect → Engineer → Trainer → Critic] × N

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
    """오케스트레이터 (v2).

    v2 파이프라인:
    Scout → KG Matcher → [Architect → Engineer → Builder → Trainer → Critic] × N
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

    def _rules_for(self, worker_name: str) -> str:
        return extract_worker_rules(self.all_rules, worker_name)

    def run_pipeline(self, data_path: str, target_col: str = "target",
                     prediction_length: int = 24, user_instructions: str = "") -> dict:
        """v2 파이프라인 실행."""
        print("CBALLM Brain v2 -- pipeline start")
        print(f"   Data: {data_path}")
        print(f"   Target: {target_col}, Horizon: {prediction_length}")
        if user_instructions:
            print(f"   Instructions: {user_instructions}")
        print()

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
        print(f"  [scout] Profiling... (rule-based)")
        scout = Scout(self.cwd)
        scout_result = scout.run(f"{contract}\n")
        self.log.append(scout_result)
        context["profile"] = scout_result.get("execution_result") or scout_result["response"]
        print(f"     done")

        # ── Step 2: KG Matcher (rule-based) ──
        print(f"  [kg_matcher] Matching... (rule-based)")
        kg_matcher = KGMatcher(self.cwd)
        kg_task = (
            f"{contract}\n"
            f"Scout Profile:\n{context['profile']}\n"
        )
        kg_result = kg_matcher.run(kg_task)
        self.log.append(kg_result)
        context["kg_match"] = kg_result.get("execution_result") or kg_result["response"]
        print(f"     done")

        # ── Iteration Loop ──
        verdict = {}
        for iteration in range(1, MAX_ITERATIONS + 1):
            print(f"\n{'='*60}")
            print(f"  Iteration {iteration}/{MAX_ITERATIONS}")
            print(f"{'='*60}\n")

            # ── Step 3: Architect (LLM Decision Protocol) ──
            architect_task = (
                f"Scout Profile:\n{context['profile']}\n\n"
                f"KG Matcher Result:\n{context['kg_match']}\n\n"
                f"PREDICTION_LENGTH = {prediction_length}\n"
            )

            if context.get("prev_configs"):
                architect_task += "\nPrevious iterations:\n"
                for prev in context["prev_configs"]:
                    architect_task += f"  - {prev['config']} -> norm_MSE={prev['norm_mse']}\n"

            if context.get("critic_feedback"):
                architect_task += f"\nCritic feedback:\n{context['critic_feedback']}\n"

            print(f"  [architect] Decision Protocol...")
            architect = Architect(self.cwd)
            architect_result = architect.run(architect_task)
            self.log.append(architect_result)

            config_json = architect_result["response"]
            context["config"] = config_json

            if architect_result.get("execution_result"):
                for line in architect_result["execution_result"].split("\n"):
                    if line.startswith("  ->"):
                        print(f"     {line}")
            print(f"     config: {config_json[:200]}")

            # ── Step 4: Engineer (rule-based) ──
            print(f"  [engineer] Feature engineering... (rule-based)")
            engineer_task = (
                f"{contract}\n"
                f"Scout Profile:\n{context['profile']}\n"
            )
            engineer = Engineer(self.cwd)
            engineer_result = engineer.run(engineer_task)
            self.log.append(engineer_result)
            context["features"] = engineer_result.get("execution_result") or engineer_result["response"]
            print(f"     done")

            # ── Step 5: Trainer (deterministic) ──
            train_data_path = feature_path if Path(feature_path).exists() else data_path
            if train_data_path != data_path:
                print(f"     Using feature file: {train_data_path}")

            trainer_contract = (
                f"DATA_PATH = '{train_data_path}'\n"
                f"TARGET_COL = '{target_col}'\n"
                f"PREDICTION_LENGTH = {prediction_length}\n"
            )
            trainer_task = (
                f"{trainer_contract}\n"
                f"Architect Config:\n{config_json}\n"
            )

            mode_label = "benchmark" if self.benchmark_mode else "CV"
            print(f"  [trainer] Training... ({mode_label} mode)")
            trainer = Trainer(self.cwd, benchmark_mode=self.benchmark_mode)
            trainer_result = trainer.run(trainer_task)
            self.log.append(trainer_result)

            context["training_result"] = trainer_result.get("execution_result") or trainer_result["response"]
            print(f"     done")

            # ── Step 6: Critic (rule-based) ──
            critic_task = (
                f"Training result:\n{context['training_result'][:2000]}\n\n"
                f"Iteration: {iteration}/{MAX_ITERATIONS}\n"
            )

            print(f"  [critic] Judging... (rule-based)")
            prev_mae = context.get("prev_mae")
            critic = Critic(self.cwd, prev_mae=prev_mae)
            critic_result = critic.run(critic_task)
            self.log.append(critic_result)

            verdict = json.loads(critic_result["response"])
            print(f"     verdict: {verdict['verdict']} | MAE: {verdict.get('best_metric', {}).get('MAE', 'N/A')}")

            # 피드백 구성
            feedback_parts = []
            if verdict.get("suggestions"):
                feedback_parts.extend(verdict["suggestions"])
            feedback_parts.append(f"verdict: {verdict['verdict']}")
            if verdict.get("analysis"):
                feedback_parts.append(f"analysis: {verdict['analysis']}")
            context["critic_feedback"] = "\n".join(feedback_parts)

            # 이전 config 기록
            norm_mse = verdict.get("best_metric", {}).get("norm_MSE", "N/A")
            try:
                parsed = json.loads(config_json)
                config_summary = json.dumps({
                    k: v.get("type") if isinstance(v, dict) else v
                    for k, v in parsed.items()
                    if k not in ("preprocessing", "input_design", "training", "_recipe_name")
                }, ensure_ascii=False)
            except (json.JSONDecodeError, AttributeError):
                config_summary = config_json[:100]

            try:
                recipe_name = json.loads(config_json).get("_recipe_name", "custom")
            except Exception:
                recipe_name = "custom"

            context.setdefault("prev_configs", []).append({
                "config": config_summary,
                "norm_mse": norm_mse,
                "recipe_name": recipe_name,
            })

            # MAE 저장
            current_mae = verdict.get("best_metric", {}).get("MAE")
            if current_mae is not None:
                context["prev_mae"] = current_mae

            # ── 분기 판단 ──
            v = verdict.get("verdict", "")
            if v == "DONE" or iteration == MAX_ITERATIONS:
                print(f"\n{'='*60}")
                print(f"  Pipeline completed (iteration {iteration})")
                print(f"{'='*60}\n")
                return self._build_final_report(context, verdict, iteration)

            # RETRY_BLOCK: KG에 대체 블록 쿼리
            if v == "RETRY_BLOCK":
                print(f"\n  RETRY_BLOCK: querying KG for replacement blocks...")
                # KG Matcher를 다시 실행하여 업데이트된 후보 제공
                kg_task_retry = (
                    f"{contract}\n"
                    f"Scout Profile:\n{context['profile']}\n"
                    f"Current config failed blocks:\n{config_json}\n"
                    f"Critic: {context['critic_feedback']}\n"
                )
                kg_retry_result = kg_matcher.run(kg_task_retry)
                context["kg_match"] = kg_retry_result.get("execution_result") or kg_retry_result["response"]
                self.log.append(kg_retry_result)

            print(f"\n  {v} -> next iteration")

        return self._build_final_report(context, verdict, MAX_ITERATIONS)

    def _build_final_report(self, context: dict, verdict: dict, iterations: int) -> dict:
        return {
            "status": "completed",
            "iterations": iterations,
            "best_model": verdict.get("best_model", "unknown"),
            "metrics": verdict.get("best_metric", {}),
            "analysis": verdict.get("analysis", ""),
            "profile_summary": context.get("profile", "")[:500],
            "features_used": context.get("features", "")[:500],
            "config_used": context.get("config", "")[:500],
            "log": self.log,
        }
