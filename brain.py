"""Brain — 오케스트레이터. 워커들을 조율하여 예측 파이프라인을 실행한다."""
from __future__ import annotations

import json
from pathlib import Path

from cballm.engine import chat
from cballm.session import WorkerSession
from cballm.workers.scout import Scout
from cballm.workers.engineer import Engineer
from cballm.workers.architect import Architect
from cballm.workers.trainer import Trainer
from cballm.workers.critic import Critic

MAX_ITERATIONS = 3

ORCHESTRATOR_PROMPT = """\
You are CBALLM Brain, an AI data science orchestrator.
You manage 5 specialist workers to build the best forecasting model.

Workers:
- Scout: data profiling and EDA
- Engineer: feature engineering
- Architect: model selection and strategy
- Trainer: model training execution
- Critic: result analysis and feedback

Your job is to:
1. Understand the user's request
2. Route tasks to the right workers in order
3. Pass results between workers
4. Summarize the final outcome

You do NOT do the work yourself. You delegate and coordinate.
"""


def load_rules(rules_dir: str) -> dict[str, str]:
    """rules/ 디렉토리에서 모든 .md 파일을 로드. 파일명 → 내용 딕셔너리 반환."""
    rules_path = Path(rules_dir)
    if not rules_path.exists():
        return {}
    result = {}
    for f in sorted(rules_path.glob("*.md")):
        result[f.stem] = f.read_text().strip()
    return result


def extract_worker_rules(all_rules: dict[str, str], worker_name: str) -> str:
    """전체 룰에서 해당 워커에 관련된 섹션만 추출한다. 컨텍스트 절약."""
    parts = []

    # 도메인 룰 (energy.md 등) — 항상 전부 포함 (짧음)
    for name, content in all_rules.items():
        if name == "general":
            continue  # general은 섹션별로 추출
        parts.append(content)

    # general.md에서 워커별 섹션 추출
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
            # 해당 섹션의 내용만 추출
            lines = general.split("\n")
            capturing = False
            section_lines = []
            for line in lines:
                if any(kw in line for kw in keywords):
                    capturing = True
                elif line.startswith("## ") and capturing:
                    # 다음 대섹션 시작 → 중단
                    if not any(kw in line for kw in keywords):
                        capturing = False
                if capturing:
                    section_lines.append(line)
            if section_lines:
                parts.append("\n".join(section_lines))

        # 7. 업계 표준 — 모든 워커에게 요약만
        if "7." in general:
            idx = general.find("## 7.")
            if idx >= 0:
                summary = general[idx:idx+1500]  # 처음 1500자만
                parts.append(summary)

    return "\n\n".join(parts) if parts else ""


class Brain:
    """오케스트레이터. 워커들을 순서대로 호출하며 파이프라인을 실행한다."""

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
        """워커에 해당하는 룰 섹션만 추출."""
        return extract_worker_rules(self.all_rules, worker_name)

    def run_pipeline(self, data_path: str, target_col: str = "target",
                     prediction_length: int = 24, user_instructions: str = "") -> dict:
        """전체 예측 파이프라인을 실행한다."""
        print("🧠 CBALLM Brain — 파이프라인 시작")
        print(f"   데이터: {data_path}")
        print(f"   타겟: {target_col}, 예측 길이: {prediction_length}")
        if user_instructions:
            print(f"   지시사항: {user_instructions}")
        print()

        context = {
            "data_path": data_path,
            "target_col": target_col,
            "prediction_length": prediction_length,
            "user_instructions": user_instructions,
        }

        # ── 공통 계약 (모든 워커에 전달) ──
        from pathlib import Path
        data_stem = Path(data_path).stem
        feature_path = f"{self.cwd}/benchmark_data/{data_stem}_features.parquet"
        results_path = f"{self.cwd}/benchmark_data/{data_stem}_results.json"

        contract = (
            f"[공통 계약 — 반드시 준수]\n"
            f"DATA_PATH = '{data_path}'\n"
            f"TARGET_COL = '{target_col}'\n"
            f"PREDICTION_LENGTH = {prediction_length}\n"
            f"FEATURE_OUTPUT_PATH = '{feature_path}'\n"
            f"RESULTS_OUTPUT_PATH = '{results_path}'\n"
            f"코드에서 위 변수명을 그대로 사용할 것. 경로·컬럼명 추측 금지.\n"
            f"코드 시작부에 반드시 위 변수를 정의하고 사용할 것.\n"
        )

        # ── Step 1: Scout (rule-based, LLM 불필요) ──
        print(f"  🔍 [scout] 프로파일링 중... (rule-based)")
        scout = Scout(self.cwd)
        scout_task = f"{contract}\n"
        scout_result = scout.run(scout_task)
        self.log.append(scout_result)
        context["profile"] = scout_result.get("execution_result") or scout_result["response"]
        print(f"     ✓ 프로파일 완료")

        # ── Iteration Loop ──
        for iteration in range(1, MAX_ITERATIONS + 1):
            print(f"\n{'='*60}")
            print(f"  🔄 Iteration {iteration}/{MAX_ITERATIONS}")
            print(f"{'='*60}\n")

            # ── Step 2: Engineer (rule-based, LLM 불필요) ──
            print(f"  🔧 [engineer] 피쳐 생성 중... (rule-based)")
            engineer_task = (
                f"{contract}\n"
                f"Scout 프로파일:\n{context['profile']}\n"
            )
            engineer = Engineer(self.cwd)
            engineer_result = engineer.run(engineer_task)
            self.log.append(engineer_result)
            context["features"] = engineer_result.get("execution_result") or engineer_result["response"]
            print(f"     ✓ 피쳐 생성 완료")

            # ── Step 3: Architect (Decision Protocol) ──
            architect_task = (
                f"Scout 프로파일:\n{context['profile']}\n"
                f"예측 길이: {prediction_length}\n"
            )

            if context.get("prev_configs"):
                architect_task += "\n이전 iteration 결과:\n"
                for prev in context["prev_configs"]:
                    architect_task += f"  - {prev['config']} → norm_MSE={prev['norm_mse']}\n"

            if context.get("critic_suggestions"):
                architect_task += f"\nCritic 피드백:\n{context['critic_suggestions']}\n"

            print(f"  🏗️ [architect] Decision Protocol 실행 중...")
            architect = Architect(self.cwd)
            architect_result = architect.run(architect_task)
            self.log.append(architect_result)

            config_json = architect_result["response"]  # 이미 JSON 문자열
            context["config"] = config_json

            # 모델링 리포트 출력
            if architect_result.get("execution_result"):
                for line in architect_result["execution_result"].split("\n"):
                    if line.startswith("  →"):
                        print(f"     {line}")
            print(f"     📋 config: {config_json[:200]}")

            # ── Step 4: Trainer (템플릿 실행기, LLM 불필요) ──
            trainer_task = (
                f"{contract}\n"
                f"Architect 설계:\n{config_json}\n\n"
                f"데이터 경로: {data_path}\n"
                f"타겟: {target_col}\n"
                f"예측 길이: {prediction_length}\n"
            )

            mode_label = "benchmark" if self.benchmark_mode else "CV"
            print(f"  🏋️ [trainer] 학습 중... ({mode_label} mode)")
            trainer = Trainer(self.cwd, benchmark_mode=self.benchmark_mode)
            trainer_result = trainer.run(trainer_task)
            self.log.append(trainer_result)

            context["training_result"] = trainer_result.get("execution_result") or trainer_result["response"]
            print(f"     ✓ 학습 완료")

            # ── Step 5: Critic (rule-based, LLM 불필요) ──
            critic_task = (
                f"학습 결과:\n{context['training_result'][:2000]}\n\n"
                f"Iteration: {iteration}/{MAX_ITERATIONS}\n"
            )

            print(f"  📊 [critic] 판정 중... (rule-based)")
            prev_mae = context.get("prev_mae")
            critic = Critic(self.cwd, prev_mae=prev_mae)
            critic_result = critic.run(critic_task)
            self.log.append(critic_result)

            # Critic 결과는 이미 structured JSON
            verdict = json.loads(critic_result["response"])
            critic_summary = json.dumps(verdict, ensure_ascii=False)
            print(f"     ✓ 판정: {verdict['verdict']} | MAE: {verdict.get('best_metric', {}).get('MAE', 'N/A')}")

            context["critic_suggestions"] = "\n".join(verdict.get("suggestions", []))

            # 이전 config + 결과 기록 (Architect 피드백용)
            norm_mse = verdict.get("best_metric", {}).get("norm_MSE", "N/A")
            try:
                parsed = json.loads(config_json)
                config_summary = json.dumps({
                    k: v.get("type") if isinstance(v, dict) else v
                    for k, v in parsed.items() if k != "reasoning"
                }, ensure_ascii=False)
            except (json.JSONDecodeError, AttributeError):
                config_summary = config_json[:100]
            context.setdefault("prev_configs", []).append({
                "config": config_summary,
                "norm_mse": norm_mse,
            })

            # 다음 iteration 비교용 MAE 저장
            current_mae = verdict.get("best_metric", {}).get("MAE")
            if current_mae is not None:
                context["prev_mae"] = current_mae

            if verdict.get("verdict") == "DONE" or iteration == MAX_ITERATIONS:
                print(f"\n{'='*60}")
                print(f"  ✅ 파이프라인 완료 (iteration {iteration})")
                print(f"{'='*60}\n")
                return self._build_final_report(context, verdict, iteration)

            print(f"\n  🔄 Critic: {verdict.get('verdict', 'RETRY')} — 다음 iteration으로")

        return self._build_final_report(context, verdict, MAX_ITERATIONS)

    def _run_worker(self, worker, task: str) -> dict:
        """워커를 실행하고 로그에 기록한다."""
        model_label = "Coder" if worker.model_profile == "code" else "Qwopus"
        print(f"  🔧 [{worker.name}] 작업 중... (모델: {model_label})")
        result = worker.run(task)

        # 요약 출력
        response_preview = result["response"][:200].replace("\n", " ")
        print(f"     ✓ 응답: {response_preview}...")
        if result.get("execution_result"):
            exec_preview = result["execution_result"][:150].replace("\n", " ")
            print(f"     ✓ 실행: {exec_preview}...")

        self.log.append(result)
        return result

    # _parse_critic_verdict 제거됨 — Critic이 이제 structured JSON을 직접 반환

    def _build_final_report(self, context: dict, verdict: dict, iterations: int) -> dict:
        """최종 리포트를 구성한다."""
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
