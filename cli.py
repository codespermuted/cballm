"""CBALLM CLI — AI 데이터 사이언티스트 오케스트레이터."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cballm.brain import Brain


def main():
    parser = argparse.ArgumentParser(description="CBALLM — AI Data Scientist Orchestrator")
    parser.add_argument("data", help="데이터 파일 경로 (csv, parquet)")
    parser.add_argument("--target", default="target", help="타겟 컬럼명 (기본: target)")
    parser.add_argument("--horizon", type=int, default=24, help="예측 길이 (기본: 24)")
    parser.add_argument("--rules", default="", help="도메인 룰 디렉토리 경로")
    parser.add_argument("--cwd", default=".", help="작업 디렉토리")
    parser.add_argument("--instructions", default="", help="추가 지시사항")
    parser.add_argument("--output", default="cballm_report.json", help="결과 저장 경로")

    args = parser.parse_args()

    # 룰 디렉토리: 명시 안 하면 현재 디렉토리의 rules/ 확인
    rules_dir = args.rules
    if not rules_dir:
        default_rules = Path(args.cwd) / "rules"
        if default_rules.exists():
            rules_dir = str(default_rules)

    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║  🧠 CBALLM — AI Data Scientist Orchestrator     ║")
    print("║     Scout → Engineer → Architect → Trainer →    ║")
    print("║     Critic → (iterate)                          ║")
    print("╚══════════════════════════════════════════════════╝")
    print()

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
    print(f"\n📄 리포트 저장: {output_path}")

    # 요약 출력
    print(f"\n{'='*50}")
    print(f"  📊 최종 결과")
    print(f"{'='*50}")
    print(f"  반복 횟수: {report['iterations']}")
    print(f"  최적 모델: {report['best_model']}")
    print(f"  메트릭: {report['metrics']}")
    print(f"  분석: {report.get('analysis', '')[:200]}")
    print()


if __name__ == "__main__":
    main()
