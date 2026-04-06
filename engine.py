"""CBALLM 엔진 — 듀얼 백엔드 (Local GGUF / API).

사용법:
    # 로컬 (기본)
    cballm data.csv --target OT

    # API (Claude Sonnet)
    cballm data.csv --target OT --engine api

    # API (모델 지정)
    cballm data.csv --target OT --engine api --api-model gpt-4o-mini

환경변수:
    ANTHROPIC_API_KEY  — Claude API 사용 시
    OPENAI_API_KEY     — OpenAI API 사용 시
"""
from __future__ import annotations

import gc
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════
#  Engine Configuration
# ══════════════════════════════════════════════════════════════

# 엔진 타입: "local" | "api"
_engine_type: str = "local"

# API 설정
_api_provider: str = "anthropic"      # "anthropic" | "openai"
_api_model: str = "claude-sonnet-4-20250514"

# API 모델 프리셋 (사용자 편의)
API_MODEL_PRESETS = {
    # Anthropic
    "claude-sonnet":     {"provider": "anthropic", "model": "claude-sonnet-4-20250514"},
    "claude-haiku":      {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"},
    "claude-opus":       {"provider": "anthropic", "model": "claude-opus-4-20250514"},
    # OpenAI
    "gpt-4o":            {"provider": "openai", "model": "gpt-4o"},
    "gpt-4o-mini":       {"provider": "openai", "model": "gpt-4o-mini"},
    "gpt-4.1-mini":      {"provider": "openai", "model": "gpt-4.1-mini"},
    "gpt-4.1-nano":      {"provider": "openai", "model": "gpt-4.1-nano"},
}

# 로컬 모델 프로파일
LOCAL_MODEL_PROFILES = {
    "reasoning": {
        "repo": "mradermacher/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF",
        "file": "Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled.Q5_K_M.gguf",
        "label": "Qwopus (추론/판단)",
    },
    "code": {
        "repo": "bartowski/Qwen2.5-Coder-32B-Instruct-GGUF",
        "file": "Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf",
        "label": "Qwen Coder 32B (코드 생성)",
    },
}

MODEL_DIR = Path.home() / "models"


def configure_engine(engine: str = "local", api_model: str = "claude-sonnet",
                     api_key: str | None = None):
    """엔진을 설정한다. CLI 또는 코드에서 호출.

    Args:
        engine: "local" 또는 "api"
        api_model: API_MODEL_PRESETS의 키 또는 직접 모델 ID
        api_key: API 키 (None이면 환경변수에서 읽음)
    """
    global _engine_type, _api_provider, _api_model

    _engine_type = engine

    if engine == "api":
        if api_model in API_MODEL_PRESETS:
            preset = API_MODEL_PRESETS[api_model]
            _api_provider = preset["provider"]
            _api_model = preset["model"]
        else:
            # 직접 모델 ID 지정 — provider 자동 감지
            _api_model = api_model
            if "claude" in api_model.lower() or "anthropic" in api_model.lower():
                _api_provider = "anthropic"
            elif "gpt" in api_model.lower() or "o1" in api_model.lower():
                _api_provider = "openai"
            else:
                _api_provider = "openai"  # 기타는 OpenAI 호환으로

        # API 키 설정
        if api_key:
            env_key = "ANTHROPIC_API_KEY" if _api_provider == "anthropic" else "OPENAI_API_KEY"
            os.environ[env_key] = api_key

        logger.info("Engine: API (%s / %s)", _api_provider, _api_model)
    else:
        logger.info("Engine: Local")


def get_engine_info() -> dict:
    """현재 엔진 설정 정보를 반환."""
    if _engine_type == "api":
        return {
            "type": "api",
            "provider": _api_provider,
            "model": _api_model,
        }
    return {
        "type": "local",
        "profiles": list(LOCAL_MODEL_PROFILES.keys()),
    }


# ══════════════════════════════════════════════════════════════
#  Local Engine (llama.cpp)
# ══════════════════════════════════════════════════════════════

_llm = None
_current_profile: str | None = None
_llama_config: dict | None = None

# 의존 프로젝트 경로 추가
_project_root = Path(__file__).parent.parent
for _dep in ["qwen_claude_distill", "myforecaster-project"]:
    _dep_path = _project_root / _dep
    if _dep_path.exists() and str(_dep_path) not in sys.path:
        sys.path.insert(0, str(_dep_path))
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def _ensure_model_downloaded(profile_name: str) -> Path:
    from huggingface_hub import hf_hub_download

    profile = LOCAL_MODEL_PROFILES[profile_name]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / profile["file"]

    if not model_path.exists():
        print(f"  모델 다운로드 중: {profile['file']}")
        hf_hub_download(
            repo_id=profile["repo"],
            filename=profile["file"],
            local_dir=str(MODEL_DIR),
            local_dir_use_symlinks=False,
        )
        print("  다운로드 완료.")

    return model_path


def _get_llama_config() -> dict:
    global _llama_config
    if _llama_config is not None:
        return _llama_config

    from harness.engine import strip_thinking
    from harness.gpu import detect_gpus, build_llama_config, print_gpu_summary

    gpus = detect_gpus()
    if gpus:
        print_gpu_summary(gpus)
    _llama_config = build_llama_config(gpus)
    return _llama_config


def _unload_model():
    global _llm, _current_profile
    if _llm is not None:
        del _llm
        _llm = None
        _current_profile = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass


def unload_local_model():
    """로컬 LLM을 VRAM에서 해제. GPU를 학습에 사용하기 전 호출."""
    _unload_model()


def _load_model(profile_name: str):
    global _llm, _current_profile
    from llama_cpp import Llama

    profile = LOCAL_MODEL_PROFILES[profile_name]
    model_path = _ensure_model_downloaded(profile_name)
    config = _get_llama_config()

    from harness.gpu import detect_gpus
    n_gpus = len(detect_gpus())
    print(f"  모델 로드: {profile['label']} (n_ctx={config['n_ctx']}, GPU {n_gpus}개)")

    _llm = Llama(model_path=str(model_path), **config)
    _current_profile = profile_name
    print(f"  {profile['label']} 로드 완료.\n")


def swap_model(profile_name: str):
    if profile_name not in LOCAL_MODEL_PROFILES:
        raise ValueError(f"Unknown profile: {profile_name}. Available: {list(LOCAL_MODEL_PROFILES.keys())}")
    if _current_profile == profile_name:
        return
    _unload_model()
    _load_model(profile_name)


def _chat_local(system_prompt: str, messages: list[dict], max_tokens: int,
                temperature: float, model: str) -> str:
    from harness.engine import strip_thinking

    swap_model(model)
    full_messages = [{"role": "system", "content": system_prompt}] + messages
    response = _llm.create_chat_completion(
        messages=full_messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.9,
    )
    raw = response["choices"][0]["message"]["content"]
    _, answer = strip_thinking(raw)
    return answer


# ══════════════════════════════════════════════════════════════
#  API Engine (Anthropic / OpenAI)
# ══════════════════════════════════════════════════════════════

_api_client = None


def _get_api_client():
    """API 클라이언트를 싱글턴으로 생성."""
    global _api_client
    if _api_client is not None:
        return _api_client

    if _api_provider == "anthropic":
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic 패키지가 필요합니다: pip install anthropic"
            )
        _api_client = anthropic.Anthropic()

    elif _api_provider == "openai":
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai 패키지가 필요합니다: pip install openai"
            )
        _api_client = openai.OpenAI()

    return _api_client


def _chat_api_anthropic(system_prompt: str, messages: list[dict],
                        max_tokens: int, temperature: float) -> str:
    client = _get_api_client()
    response = client.messages.create(
        model=_api_model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_prompt,
        messages=messages,
    )
    return response.content[0].text


def _chat_api_openai(system_prompt: str, messages: list[dict],
                     max_tokens: int, temperature: float) -> str:
    client = _get_api_client()
    full_messages = [{"role": "system", "content": system_prompt}] + messages
    response = client.chat.completions.create(
        model=_api_model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=full_messages,
    )
    return response.choices[0].message.content


def _chat_api(system_prompt: str, messages: list[dict], max_tokens: int,
              temperature: float) -> str:
    if _api_provider == "anthropic":
        return _chat_api_anthropic(system_prompt, messages, max_tokens, temperature)
    else:
        return _chat_api_openai(system_prompt, messages, max_tokens, temperature)


# ══════════════════════════════════════════════════════════════
#  Unified Interface
# ══════════════════════════════════════════════════════════════

def chat(system_prompt: str, messages: list[dict], max_tokens: int = 4096,
         temperature: float = 0.3, model: str = "reasoning") -> str:
    """통합 chat 인터페이스.

    engine="local"이면 로컬 GGUF, engine="api"이면 API 호출.
    model 파라미터는 로컬에서만 사용 (reasoning/code 프로파일 선택).
    """
    if _engine_type == "api":
        return _chat_api(system_prompt, messages, max_tokens, temperature)
    else:
        return _chat_local(system_prompt, messages, max_tokens, temperature, model)
