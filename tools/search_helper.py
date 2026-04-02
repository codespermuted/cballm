"""검색 헬퍼 — 워커의 코드에서 호출 가능한 검색 함수."""
from __future__ import annotations

import sys
sys.path.insert(0, "/workspace/Desktop/qwen_claude_distill")

from harness.search import github_search, scholar_search, stackoverflow_search
from harness.web import web_search, web_fetch


def search_models(query: str) -> str:
    """시계열 모델 관련 Scholar + GitHub 검색."""
    results = []
    results.append("=== Google Scholar ===")
    results.append(scholar_search(query, max_results=3, year_from=2023, exclude_survey=True))
    results.append("\n=== GitHub ===")
    results.append(github_search(query, max_results=3, language="python"))
    return "\n".join(results)


def search_technique(query: str) -> str:
    """기법/방법론 검색."""
    results = []
    results.append("=== Scholar ===")
    results.append(scholar_search(query, max_results=3, year_from=2022))
    results.append("\n=== StackOverflow ===")
    results.append(stackoverflow_search(query, max_results=3))
    return "\n".join(results)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--type", default="models", choices=["models", "technique"])
    args = parser.parse_args()

    if args.type == "models":
        print(search_models(args.query))
    else:
        print(search_technique(args.query))
