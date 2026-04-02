# 🧠 Cortex

**AI 데이터 사이언티스트 오케스트레이터** — 도메인 룰 기반 자동 시계열 예측 파이프라인

LLM이 5명의 전문 워커를 조율하여, 도메인 지식을 반영한 정교한 모델링을 수행합니다.

## 아키텍처

```
사용자: "이 데이터로 SMP 예측해줘"
  ↓
🧠 Brain (오케스트레이터) — 27B LLM, 프롬프트 스위칭
  ├→ 📊 Scout     데이터 프로파일링, regime 진단, prior 후보 수집
  ├→ 🔧 Engineer  피쳐 엔지니어링 (Observational Bias 주입)
  ├→ 🏗️ Architect  모델 선택 + 3-Bias Prior Injection
  ├→ 🏋️ Trainer    학습 실행 (temporal validation 엄수)
  └→ 🔍 Critic     결과 분석, 정상/극단 분리 평가, 피드백 라우팅
                      ↓
                  DONE? → 완료
                  RETRY? → 해당 워커로 복귀 (최대 3회)
```

## 핵심 설계 철학

- **정교한 단일 모델 + prior 주입 우선, 앙상블은 마지막 수단**
- **3-Bias Prior Injection** (PIML 표준): Observational → Inductive → Learning
- **도메인 룰은 코드화된 자산** — `rules/` 디렉토리에 마크다운으로 관리
- **모델 1개, 세션 여러 개** — VRAM 추가 0, 프롬프트 스위칭으로 워커 특화

## 사용법

```bash
# 기본 실행
python -m cortex data.parquet --target smp --horizon 24

# 도메인 룰 지정
python -m cortex data.parquet --target smp --horizon 24 --rules rules/

# 추가 지시사항
python -m cortex data.parquet --target smp --horizon 24 --instructions "Dip 구간 성능 중시"
```

## 룰 시스템

`rules/` 디렉토리에 `.md` 파일로 도메인 지식을 정리하면 워커에 자동 주입됩니다:

- `rules/general.md` — 범용 시계열 모델링 노하우 (워커별 해당 섹션만 추출)
- `rules/energy.md` — 에너지/전력 시장 특화 룰

워커별로 필요한 섹션만 주입하여 컨텍스트를 절약합니다.

## 의존성

- **Qwopus** — LLM 엔진 (27B 로컬 모델)
- **C-BAL** (선택) — AutoML 시계열 예측 라이브러리

## 라이선스

MIT
