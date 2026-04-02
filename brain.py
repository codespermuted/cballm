"""Brain — 오케스트레이터. 워커들을 조율하여 예측 파이프라인을 실행한다."""
from __future__ import annotations

import json
from pathlib import Path

from cortex.engine import chat
from cortex.session import WorkerSession
from cortex.workers.scout import Scout
from cortex.workers.engineer import Engineer
from cortex.workers.architect import Architect
from cortex.workers.trainer import Trainer
from cortex.workers.critic import Critic

MAX_ITERATIONS = 3

ORCHESTRATOR_PROMPT = """\
You are Cortex Brain, an AI data science orchestrator.
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

    def __init__(self, cwd: str, rules_dir: str = ""):
        self.cwd = cwd
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
        print("🧠 Cortex Brain — 파이프라인 시작")
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

        # ── Step 1: Scout ──
        scout_result = self._run_worker(
            Scout(self.cwd, self._rules_for("scout")),
            f"다음 데이터를 프로파일링해줘:\n"
            f"- 경로: {data_path}\n"
            f"- 타겟 컬럼: {target_col}\n"
            f"- 예측 길이: {prediction_length}",
        )
        context["profile"] = scout_result.get("execution_result") or scout_result["response"]

        # ── Iteration Loop ──
        for iteration in range(1, MAX_ITERATIONS + 1):
            print(f"\n{'='*60}")
            print(f"  🔄 Iteration {iteration}/{MAX_ITERATIONS}")
            print(f"{'='*60}\n")

            # ── Step 2: Engineer ──
            engineer_task = (
                f"Scout 프로파일:\n{context['profile'][:2000]}\n\n"
                f"타겟: {target_col}\n"
                f"예측 길이: {prediction_length}\n"
            )
            if context.get("critic_suggestions"):
                engineer_task += f"\nCritic 피드백:\n{context['critic_suggestions']}\n"
            if user_instructions:
                engineer_task += f"\n사용자 지시:\n{user_instructions}\n"

            engineer_result = self._run_worker(
                Engineer(self.cwd, self._rules_for("engineer")),
                engineer_task,
            )
            context["features"] = engineer_result.get("execution_result") or engineer_result["response"]

            # ── Step 3: Architect ──
            architect_task = (
                f"Scout 프로파일:\n{context['profile'][:1500]}\n\n"
                f"Engineer 피쳐:\n{context['features'][:1500]}\n\n"
                f"타겟: {target_col}, 예측 길이: {prediction_length}\n"
            )
            if context.get("critic_suggestions"):
                architect_task += f"\nCritic 피드백:\n{context['critic_suggestions']}\n"

            architect_result = self._run_worker(
                Architect(self.cwd, self._rules_for("architect")),
                architect_task,
            )
            context["config"] = architect_result["response"]

            # ── Step 4: Trainer ──
            trainer_task = (
                f"Architect 설계:\n{context['config'][:2000]}\n\n"
                f"데이터 경로: {data_path}\n"
                f"타겟: {target_col}\n"
                f"예측 길이: {prediction_length}\n"
            )

            trainer_result = self._run_worker(
                Trainer(self.cwd, self._rules_for("trainer")),
                trainer_task,
            )
            context["training_result"] = trainer_result.get("execution_result") or trainer_result["response"]

            # ── Step 5: Critic ──
            critic_task = (
                f"학습 결과:\n{context['training_result'][:2000]}\n\n"
                f"Scout 프로파일:\n{context['profile'][:1000]}\n\n"
                f"Architect 설계:\n{context['config'][:1000]}\n\n"
                f"Iteration: {iteration}/{MAX_ITERATIONS}\n"
            )

            critic_result = self._run_worker(
                Critic(self.cwd, self._rules_for("critic")),
                critic_task,
            )

            # Critic 판정 파싱
            verdict = self._parse_critic_verdict(critic_result["response"])
            context["critic_suggestions"] = verdict.get("suggestions", "")

            if verdict.get("verdict") == "DONE" or iteration == MAX_ITERATIONS:
                print(f"\n{'='*60}")
                print(f"  ✅ 파이프라인 완료 (iteration {iteration})")
                print(f"{'='*60}\n")
                return self._build_final_report(context, verdict, iteration)

            print(f"\n  🔄 Critic: {verdict.get('verdict', 'RETRY')} — 다음 iteration으로")

        return self._build_final_report(context, verdict, MAX_ITERATIONS)

    def _run_worker(self, worker, task: str) -> dict:
        """워커를 실행하고 로그에 기록한다."""
        print(f"  🔧 [{worker.name}] 작업 중...")
        result = worker.run(task)

        # 요약 출력
        response_preview = result["response"][:200].replace("\n", " ")
        print(f"     ✓ 응답: {response_preview}...")
        if result.get("execution_result"):
            exec_preview = result["execution_result"][:150].replace("\n", " ")
            print(f"     ✓ 실행: {exec_preview}...")

        self.log.append(result)
        return result

    def _parse_critic_verdict(self, response: str) -> dict:
        """Critic 응답에서 JSON verdict를 추출한다."""
        import re
        # JSON 블록 추출
        json_match = re.search(r'\{[^{}]*"verdict"[^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        # fallback
        if "DONE" in response:
            return {"verdict": "DONE"}
        return {"verdict": "RETRY_BOTH", "suggestions": response[:500]}

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
