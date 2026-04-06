"""ArchitectPromptBuilder — 계층적 프롬프트 빌더 + RoundCompactor.

패턴 B: claw-code의 SystemPromptBuilder 참고.
패턴 C: claw-code의 compact.rs 참고.

Section별 토큰 예산 관리:
  Section 1: Task Definition    (~200 tokens, 고정)
  Section 2: Available Blocks   (~300 tokens, 고정)
  Section 3: Diagnostic Directive (~500 tokens, 가변)
  Section 4: Round History      (~400 tokens, compacted)
  Section 5: Question           (~100 tokens, 고정)
  총 예산: ~1500 tokens

LLM은 최종 단답 선택에만 사용한다. 구조화/필터링/요약은 전부 rule-based.
"""
from __future__ import annotations

import json

from cballm.hooks import (
    DiversityLedger,
    PostTrainDiagnosis,
    RoundRecord,
)

CHARS_PER_TOKEN = 3  # 한국어+영어 혼합 추정


class RoundCompactor:
    """라운드 이력을 요약 압축한다.

    규칙:
    - 최근 1라운드: 원문 보존 (진단 포함)
    - 이전 라운드: compact 1줄 요약
    - merge_compact_summaries: 누적 요약끼리 병합
    """

    @staticmethod
    def compact(ledger: DiversityLedger,
                preserve_recent: int = 1,
                max_chars: int = 1200) -> str:
        """라운드 이력을 압축된 문자열로 반환."""
        rounds = ledger.rounds
        if not rounds:
            return "이전 라운드 없음."

        parts: list[str] = []
        budget = max_chars

        # ── 이전 라운드: compact 요약 ──
        old_rounds = rounds[:-preserve_recent] if len(rounds) > preserve_recent else []
        if old_rounds:
            summaries = [r.compact_summary() for r in old_rounds]
            merged = RoundCompactor._merge_summaries(summaries)
            parts.append("--- 이전 라운드 (요약) ---")
            for s in merged:
                if budget - len(s) < 0:
                    parts.append("... (이전 라운드 일부 생략)")
                    break
                parts.append(s)
                budget -= len(s)

        # ── 최근 라운드: 원문 보존 ──
        recent = rounds[-preserve_recent:]
        if recent:
            parts.append("--- 최근 라운드 (상세) ---")
            for r in recent:
                detail = RoundCompactor._format_recent(r)
                if budget - len(detail) < 0:
                    # 예산 초과 시에도 최근은 compact로라도 포함
                    parts.append(r.compact_summary())
                else:
                    parts.append(detail)
                    budget -= len(detail)

        return "\n".join(parts)

    @staticmethod
    def _format_recent(record: RoundRecord) -> str:
        """최근 라운드의 상세 포맷."""
        lines = [record.compact_summary()]
        if record.diagnosis:
            diag = record.diagnosis
            if diag.directive and diag.directive != "진단 이상 없음":
                lines.append(f"  진단: {diag.directive}")
            for attr in diag.attributions:
                lines.append(f"  {attr.summary()}")
        return "\n".join(lines)

    @staticmethod
    def _merge_summaries(summaries: list[str]) -> list[str]:
        """동일 패턴의 요약을 병합.

        예: R1, R2 모두 LinearMix로 실패 → 하나로 합침.
        """
        if len(summaries) <= 2:
            return summaries

        # 같은 mixer로 실패한 것들을 그룹핑
        groups: dict[str, list[str]] = {}
        for s in summaries:
            # mixer 이름 추출
            key = s.split("(")[0] if "(" in s else s[:20]
            groups.setdefault(key, []).append(s)

        merged = []
        for key, items in groups.items():
            if len(items) == 1:
                merged.append(items[0])
            else:
                # 그룹 내 최고/최저만 보존
                merged.append(items[0])
                if len(items) > 2:
                    merged.append(f"  ... ({len(items) - 2}개 유사 라운드 생략)")
                merged.append(items[-1])
        return merged


class ArchitectPromptBuilder:
    """Architect에게 보내는 프롬프트를 section별로 조립한다.

    각 section은 독립적으로 빌드되며, 토큰 예산을 초과하면 truncate.
    """

    # 토큰 예산 (chars 기준, 1 token ≈ 3 chars)
    BUDGET = {
        "task": 600,       # ~200 tokens
        "blocks": 900,     # ~300 tokens
        "diagnostic": 1500, # ~500 tokens
        "history": 1200,   # ~400 tokens
        "question": 300,   # ~100 tokens
    }

    def __init__(self, profile: dict, kg_match: str,
                 prediction_length: int):
        self.profile = profile
        self.kg_match = kg_match
        self.prediction_length = prediction_length

    def build(self, ledger: DiversityLedger,
              diagnosis: PostTrainDiagnosis | None = None,
              verdict: dict | None = None,
              retry_type: str = "") -> str:
        """전체 프롬프트를 조립."""
        sections = [
            self._section_task(),
            self._section_blocks(ledger),
            self._section_diagnostic(diagnosis, verdict),
            self._section_history(ledger),
            self._section_question(retry_type, ledger),
        ]
        return "\n\n".join(s for s in sections if s)

    def _section_task(self) -> str:
        """Section 1: Task Definition (고정)."""
        # profile에서 핵심만 추출
        lines = ["[Task]"]
        lines.append(f"PREDICTION_LENGTH = {self.prediction_length}")

        # profile 텍스트에서 핵심 4줄만
        if isinstance(self.profile, str):
            for line in self.profile.split("\n"):
                if any(k in line for k in ["Shape:", "Target stats:", "Seasonality:", "Regime:"]):
                    lines.append(line.strip())
        elif isinstance(self.profile, dict):
            lines.append(f"Shape: ({self.profile.get('n_rows', '?')}, {self.profile.get('n_features', '?')})")
            lines.append(f"Stationary: {self.profile.get('is_stationary', '?')}")

        return _truncate("\n".join(lines), self.BUDGET["task"])

    def _section_blocks(self, ledger: DiversityLedger) -> str:
        """Section 2: Available Blocks (고정, 시도하지 않은 것 위주)."""
        lines = ["[Available Blocks]"]

        # KG match에서 유효 블록 목록 추출
        for field in ["VALID_ENCODERS", "VALID_TEMPORAL_MIXERS",
                       "VALID_CHANNEL_MIXERS", "VALID_LOSSES"]:
            if field in self.kg_match:
                # 이미 시도한 블록 표시
                slot = field.replace("VALID_", "").lower()
                slot_map = {
                    "encoders": "encoder",
                    "temporal_mixers": "temporal_mixer",
                    "channel_mixers": "channel_mixer",
                    "losses": "loss",
                }
                actual_slot = slot_map.get(slot, slot)
                tried = set(ledger.get_tried_blocks(actual_slot))

                # 원문에서 해당 라인 추출
                for line in self.kg_match.split("\n"):
                    if field in line:
                        # 시도한 블록을 별도 표기 (블록 이름 자체는 변경하지 않음)
                        if tried:
                            lines.append(line.strip())
                            lines.append(f"  (tried: {', '.join(tried)})")
                        else:
                            lines.append(line.strip())
                        break

        # 후보 레시피 (상위 3개)
        recipe_lines = []
        for line in self.kg_match.split("\n"):
            if line.strip().startswith("[") and "capacity=" in line:
                recipe_lines.append(line.strip())
        if recipe_lines:
            lines.append("Candidate Recipes: " + " | ".join(recipe_lines[:3]))

        return _truncate("\n".join(lines), self.BUDGET["blocks"])

    def _section_diagnostic(self, diagnosis: PostTrainDiagnosis | None,
                            verdict: dict | None) -> str:
        """Section 3: Diagnostic Directive (가변).

        Diagnosis 스키마 → pre-digested 텍스트 변환:
        1. Diagnosis.to_prompt_directive() → 구조화된 문제/잔차/방향
        2. Critic verdict → 보조 분석
        3. PostTrainHook directive → 상세 보충
        """
        if diagnosis is None and verdict is None:
            return ""

        lines = ["[Diagnostic]"]

        # (1) Diagnosis 스키마 → 구조화된 지시문 (최우선)
        if diagnosis and diagnosis.diagnosis:
            diag = diagnosis.diagnosis
            prompt_text = diag.to_prompt_directive()
            if prompt_text:
                lines.append(prompt_text)

            # 블록 블랙리스트 명시
            if diag.block_blacklist:
                lines.append(f"Blacklist: {', '.join(diag.block_blacklist)}")

        # (2) Critic verdict (보조)
        if verdict:
            v = verdict.get("verdict", "")
            analysis = verdict.get("analysis", "")
            lines.append(f"Verdict: {v}")
            if analysis:
                lines.append(f"Analysis: {analysis[:200]}")

        # (3) PostTrainHook 상세 보충 (잔차 통계 등)
        if diagnosis and diagnosis.directive:
            # Diagnosis.to_prompt_directive()와 중복되지 않는 부분만 (exact match)
            existing = set(lines)
            for line in diagnosis.directive.split("\n"):
                if line.startswith("[잔차]") or line.startswith("[액션]"):
                    if line not in existing:
                        lines.append(line)
                        existing.add(line)

        return _truncate("\n".join(lines), self.BUDGET["diagnostic"])

    def _section_history(self, ledger: DiversityLedger) -> str:
        """Section 4: Round History (compacted)."""
        if not ledger.rounds:
            return ""

        history = RoundCompactor.compact(ledger, preserve_recent=1,
                                          max_chars=self.BUDGET["history"])
        return f"[History]\n{history}"

    def _section_question(self, retry_type: str,
                          ledger: DiversityLedger) -> str:
        """Section 5: Question (고정)."""
        lines = ["[Question]"]

        if retry_type == "RETRY_RECIPE":
            tried = ledger.get_tried_blocks("temporal_mixer")
            if tried:
                lines.append(f"이미 시도: {', '.join(tried)}")
            lines.append("다른 레시피를 선택하세요. 레시피 이름만.")
        elif retry_type == "RETRY_HP":
            lines.append("진단을 참고하여 HP를 조정하세요.")
        elif retry_type == "RETRY_BLOCK":
            lines.append("진단에서 무효/한계로 판정된 블록을 교체하세요.")
        else:
            lines.append("첫 시도입니다. DLinear(baseline)을 권장합니다. 레시피를 선택하세요.")

        return _truncate("\n".join(lines), self.BUDGET["question"])


# ══════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════

def _truncate(text: str, max_chars: int) -> str:
    """텍스트를 max_chars 이내로 truncate."""
    if len(text) <= max_chars:
        return text
    # 마지막 완전한 줄에서 자르기
    truncated = text[:max_chars]
    last_newline = truncated.rfind("\n")
    if last_newline > max_chars * 0.5:
        truncated = truncated[:last_newline]
    return truncated + "\n... (truncated)"


def estimate_tokens(text: str) -> int:
    """토큰 수 추정."""
    return max(1, len(text) // CHARS_PER_TOKEN)
