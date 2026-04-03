"""워커 베이스 — 모든 워커가 상속."""
from __future__ import annotations

import re
import subprocess
from abc import ABC

from cballm.session import WorkerSession
from cballm.engine import chat


class BaseWorker(ABC):
    name: str = "base"
    description: str = ""
    system_prompt: str = ""
    model_profile: str = "reasoning"  # "reasoning" (Qwopus) or "code" (Qwen Coder)

    def __init__(self, cwd: str, rules: str = ""):
        self.cwd = cwd
        prompt = self.system_prompt
        if rules:
            prompt += f"\n\n# Domain Rules\n{rules}"
        self.session = WorkerSession(worker_name=self.name, system_prompt=prompt)

    MAX_RETRIES = 2  # 코드 실행 실패 시 최대 재시도 횟수

    def run(self, task: str) -> dict:
        """작업 실행 → 응답 + 코드 자동 실행 결과 반환. 실패 시 자동 재시도."""
        self.session.add_user(task)
        response = chat(
            self.session.system_prompt,
            self.session.messages,
            max_tokens=4096,
            model=self.model_profile,
        )
        self.session.add_assistant(response)

        # 코드 블록 자동 실행 + 에러 시 재시도 루프
        code = self._extract_code(response)
        exec_result = None

        if code:
            exec_result = self._run_code(code)

            # 에러 감지 → LLM에 에러 피드백 → 코드 재생성
            for retry in range(self.MAX_RETRIES):
                if not self._has_error(exec_result):
                    break
                print(f"     ⚠️ 코드 실행 실패 (retry {retry+1}/{self.MAX_RETRIES})")
                error_feedback = (
                    f"[코드 실행 에러 — 수정 필요]\n{exec_result}\n\n"
                    f"위 에러를 수정한 전체 코드를 다시 ```python``` 블록으로 작성해줘. "
                    f"같은 실수를 반복하지 말고, import 누락·경로·컬럼명을 확인해."
                )
                self.session.add_user(error_feedback)
                response = chat(
                    self.session.system_prompt,
                    self.session.messages,
                    max_tokens=4096,
                    model=self.model_profile,
                )
                self.session.add_assistant(response)
                code = self._extract_code(response)
                if code:
                    exec_result = self._run_code(code)
                else:
                    break

        # 실행 결과가 있으면 세션에 추가 (다음 대화에서 참조 가능)
        if exec_result:
            self.session.add_user(f"[코드 실행 결과]\n{exec_result}")

        return {
            "worker": self.name,
            "response": response,
            "code": code,
            "execution_result": exec_result,
        }

    @staticmethod
    def _has_error(exec_result: str) -> bool:
        """실행 결과에 에러가 있는지 판단."""
        if not exec_result:
            return False
        error_indicators = ["[STDERR]", "Traceback", "Error:", "error:", "(오류:", "(타임아웃"]
        return any(ind in exec_result for ind in error_indicators)

    def follow_up(self, message: str) -> dict:
        """이전 컨텍스트를 유지하면서 추가 작업."""
        return self.run(message)

    def reset(self):
        self.session.clear()

    def _extract_code(self, text: str) -> str:
        blocks = re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)
        return "\n\n".join(blocks) if blocks else ""

    def _run_code(self, code: str, timeout: int = 300) -> str:
        import os
        env = os.environ.copy()
        # C-BAL 및 기타 프로젝트 경로 추가
        extra_paths = [
            "/workspace/Desktop/myforecaster-project",
            "/workspace/Desktop/cballm",
            "/workspace/Desktop/qwen_claude_distill",
        ]
        env["PYTHONPATH"] = ":".join(extra_paths) + ":" + env.get("PYTHONPATH", "")
        env["PYTHONUNBUFFERED"] = "1"

        try:
            result = subprocess.run(
                ["python3", "-c", code],
                capture_output=True, text=True,
                cwd=self.cwd, timeout=timeout,
                env=env,
            )
            output = result.stdout
            if result.stderr:
                # 경고는 무시, 에러만 포함
                errors = [l for l in result.stderr.split("\n")
                         if "Error" in l or "Traceback" in l or "error" in l.lower()]
                if errors:
                    output += "\n[STDERR]\n" + "\n".join(errors[-10:])
            return (output or "(출력 없음)")[:5000]
        except subprocess.TimeoutExpired:
            return f"(타임아웃 {timeout}초 — 학습 시간 초과)"
        except Exception as e:
            return f"(오류: {e})"
