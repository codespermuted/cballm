"""CBALLM CLI — AI 데이터 사이언티스트 오케스트레이터."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cballm.brain import Brain
from cballm.engine import configure_engine, get_engine_info, API_MODEL_PRESETS


def main():
    parser = argparse.ArgumentParser(
        description="CBALLM — AI Data Scientist Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_build_epilog(),
    )
    parser.add_argument("data", help="데이터 파일 경로 (csv, parquet)")
    parser.add_argument("--target", default="target", help="타겟 컬럼명 (기본: target)")
    parser.add_argument("--horizon", type=int, default=24, help="예측 길이 (기본: 24)")
    parser.add_argument("--rules", default="", help="도메인 룰 디렉토리 경로")
    parser.add_argument("--cwd", default=".", help="작업 디렉토리")
    parser.add_argument("--instructions", default="", help="추가 지시사항")
    parser.add_argument("--output", default="cballm_report.json", help="결과 저장 경로")

    # ── 엔진 선택 ──
    engine_group = parser.add_argument_group("Engine (LLM 선택)")
    engine_group.add_argument(
        "--engine", choices=["local", "api"], default="local",
        help="LLM 엔진 (기본: local)",
    )
    engine_group.add_argument(
        "--api-model", default="claude-sonnet",
        help=f"API 모델 프리셋 또는 모델 ID (기본: claude-sonnet). "
             f"프리셋: {', '.join(API_MODEL_PRESETS.keys())}",
    )
    engine_group.add_argument(
        "--api-key", default=None,
        help="API 키 (미지정 시 ANTHROPIC_API_KEY 또는 OPENAI_API_KEY 환경변수 사용)",
    )

    parser.add_argument(
        "--prior", default="",
        help="도메인 prior yaml 파일 경로 (예: domain_priors/energy_smp.yaml)",
    )

    args = parser.parse_args()

    # ── 엔진 설정 ──
    configure_engine(
        engine=args.engine,
        api_model=args.api_model,
        api_key=args.api_key,
    )
    engine_info = get_engine_info()
    if engine_info["type"] == "api":
        print(f"  Engine: API ({engine_info['provider']} / {engine_info['model']})")
    else:
        print(f"  Engine: Local ({', '.join(engine_info['profiles'])})")

    # 룰 디렉토리: 명시 안 하면 현재 디렉토리의 rules/ 확인
    rules_dir = args.rules
    if not rules_dir:
        default_rules = Path(args.cwd) / "rules"
        if default_rules.exists():
            rules_dir = str(default_rules)

    brain = Brain(cwd=args.cwd, rules_dir=rules_dir)

    report = brain.run_pipeline(
        data_path=args.data,
        target_col=args.target,
        prediction_length=args.horizon,
        user_instructions=args.instructions,
    )

    # 리포트 저장
    output_path = Path(args.output)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print(f"\n  Report saved: {output_path}")


def _build_epilog() -> str:
    return """\
Examples:
  # 로컬 LLM (기본)
  cballm data.csv --target OT --horizon 96

  # Claude API
  cballm data.csv --target OT --engine api

  # GPT-4o-mini API
  cballm data.csv --target OT --engine api --api-model gpt-4o-mini

  # API 키 직접 지정
  cballm data.csv --target OT --engine api --api-key sk-...
"""


if __name__ == "__main__":
    main()
