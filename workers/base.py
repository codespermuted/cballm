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
