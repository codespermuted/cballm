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

    def __init__(self, cwd: str, rules: str = ""):
        self.cwd = cwd
        prompt = self.system_prompt
        if rules:
            prompt += f"\n\n# Domain Rules\n{rules}"
        self.session = WorkerSession(worker_name=self.name, system_prompt=prompt)

    def run(self, task: str) -> dict:
        """작업 실행 → 응답 + 코드 자동 실행 결과 반환."""
        self.session.add_user(task)
        response = chat(
            self.session.system_prompt,
            self.session.messages,
            max_tokens=4096,
        )
        self.session.add_assistant(response)

        # 코드 블록 자동 실행
        code = self._extract_code(response)
        exec_result = self._run_code(code) if code else None

        # 실행 결과가 있으면 세션에 추가 (다음 대화에서 참조 가능)
        if exec_result:
            self.session.add_user(f"[코드 실행 결과]\n{exec_result}")

        return {
            "worker": self.name,
            "response": response,
            "code": code,
            "execution_result": exec_result,
        }

    def follow_up(self, message: str) -> dict:
        """이전 컨텍스트를 유지하면서 추가 작업."""
        return self.run(message)

    def reset(self):
        self.session.clear()

    def _extract_code(self, text: str) -> str:
        blocks = re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)
        return "\n\n".join(blocks) if blocks else ""

    def _run_code(self, code: str) -> str:
        try:
            result = subprocess.run(
                ["python3", "-c", code],
                capture_output=True, text=True,
                cwd=self.cwd, timeout=120,
            )
            output = result.stdout
            if result.stderr:
                output += "\n[STDERR]\n" + result.stderr
            return (output or "(출력 없음)")[:3000]
        except subprocess.TimeoutExpired:
            return "(타임아웃 120초)"
        except Exception as e:
            return f"(오류: {e})"
