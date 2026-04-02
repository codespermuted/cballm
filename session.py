"""워커 세션 — 각 워커의 독립적인 대화 컨텍스트."""
from __future__ import annotations

from dataclasses import dataclass, field

CHARS_PER_TOKEN = 3
MAX_CONTEXT_TOKENS = 28000  # 32K에서 여유분 확보


@dataclass
class WorkerSession:
    """워커 하나의 독립 세션. 메인 컨텍스트를 오염시키지 않는다."""
    worker_name: str
    system_prompt: str
    messages: list[dict] = field(default_factory=list)

    def add_user(self, content: str):
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str):
        self.messages.append({"role": "assistant", "content": content})

    def get_messages(self) -> list[dict]:
        """LLM에 전달할 메시지 목록 (토큰 예산 내에서)."""
        budget = MAX_CONTEXT_TOKENS - self._estimate(self.system_prompt)
        selected = []
        used = 0
        for msg in reversed(self.messages):
            tokens = self._estimate(msg["content"])
            if used + tokens > budget:
                break
            selected.append(msg)
            used += tokens
        selected.reverse()
        return [{"role": "system", "content": self.system_prompt}] + selected

    def clear(self):
        self.messages.clear()

    @staticmethod
    def _estimate(text: str) -> int:
        return max(1, len(text) // CHARS_PER_TOKEN)
