"""CBALLM 엔진 — Qwopus의 LLM 엔진을 가져다 쓴다."""
from __future__ import annotations

import sys
from pathlib import Path

# 의존 프로젝트 경로 추가
_project_root = Path(__file__).parent.parent
for _dep in ["qwen_claude_distill", "myforecaster-project"]:
    _dep_path = _project_root / _dep
    if _dep_path.exists() and str(_dep_path) not in sys.path:
        sys.path.insert(0, str(_dep_path))
# cballm 자체도 PYTHONPATH에
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from harness.engine import get_llm, chat_completion, chat_completion_stream, strip_thinking, get_n_ctx


def chat(system_prompt: str, messages: list[dict], max_tokens: int = 4096, temperature: float = 0.3) -> str:
    """시스템 프롬프트 + 메시지로 LLM 추론. thinking 자동 제거."""
    full_messages = [{"role": "system", "content": system_prompt}] + messages
    response = chat_completion(full_messages, max_tokens=max_tokens, temperature=temperature)
    raw = response["choices"][0]["message"]["content"]
    _, answer = strip_thinking(raw)
    return answer


def chat_stream(system_prompt: str, messages: list[dict], max_tokens: int = 4096, temperature: float = 0.3):
    """스트리밍 추론. 토큰 단위로 yield."""
    full_messages = [{"role": "system", "content": system_prompt}] + messages
    yield from chat_completion_stream(full_messages, max_tokens=max_tokens, temperature=temperature)
