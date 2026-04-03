"""CBALLM 엔진 — 듀얼 모델 스왑 방식. Qwopus(추론) + Qwen Coder(코드 생성)."""
from __future__ import annotations

import gc
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# 의존 프로젝트 경로 추가
_project_root = Path(__file__).parent.parent
for _dep in ["qwen_claude_distill", "myforecaster-project"]:
    _dep_path = _project_root / _dep
    if _dep_path.exists() and str(_dep_path) not in sys.path:
        sys.path.insert(0, str(_dep_path))
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from harness.engine import strip_thinking
from harness.gpu import detect_gpus, build_llama_config, print_gpu_summary

# ── 모델 프로파일 ──
MODEL_PROFILES = {
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

# ── 싱글턴 상태 ──
_llm = None
_current_profile: str | None = None
_llama_config: dict | None = None


def _ensure_model_downloaded(profile_name: str) -> Path:
    """모델 파일이 없으면 다운로드. 경로 반환."""
    from huggingface_hub import hf_hub_download

    profile = MODEL_PROFILES[profile_name]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / profile["file"]

    if not model_path.exists():
        print(f"⬇️  모델 다운로드 중: {profile['file']}")
        hf_hub_download(
            repo_id=profile["repo"],
            filename=profile["file"],
            local_dir=str(MODEL_DIR),
            local_dir_use_symlinks=False,
        )
        print("✅ 다운로드 완료.")

    return model_path


def _get_llama_config() -> dict:
    """GPU 설정을 감지하고 캐시."""
    global _llama_config
    if _llama_config is not None:
        return _llama_config

    gpus = detect_gpus()
    print("🔍 GPU 감지:")
    if gpus:
        print_gpu_summary(gpus)
    _llama_config = build_llama_config(gpus)
    return _llama_config


def _unload_model():
    """현재 모델을 VRAM에서 해제."""
    global _llm, _current_profile
    if _llm is not None:
        print(f"🔄 모델 언로드: {MODEL_PROFILES[_current_profile]['label']}")
        del _llm
        _llm = None
        _current_profile = None
        gc.collect()
        # CUDA 캐시 해제
        try:
            import torch
            torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass


def _load_model(profile_name: str):
    """지정된 프로파일의 모델을 로드."""
    global _llm, _current_profile
    from llama_cpp import Llama

    profile = MODEL_PROFILES[profile_name]
    model_path = _ensure_model_downloaded(profile_name)
    config = _get_llama_config()

    n_gpus = len(detect_gpus())
    print(f"🚀 모델 로드: {profile['label']} (n_ctx={config['n_ctx']}, GPU {n_gpus}개)")

    _llm = Llama(model_path=str(model_path), **config)
    _current_profile = profile_name
    print(f"✅ {profile['label']} 로드 완료.\n")


def swap_model(profile_name: str):
    """필요시 모델을 교체한다. 이미 로드된 모델이면 스킵."""
    if profile_name not in MODEL_PROFILES:
        raise ValueError(f"Unknown profile: {profile_name}. Available: {list(MODEL_PROFILES.keys())}")

    if _current_profile == profile_name:
        return  # 이미 올바른 모델

    _unload_model()
    _load_model(profile_name)


def chat(system_prompt: str, messages: list[dict], max_tokens: int = 4096,
         temperature: float = 0.3, model: str = "reasoning") -> str:
    """시스템 프롬프트 + 메시지로 LLM 추론. 모델 프로파일 지정 가능."""
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
