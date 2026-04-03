# Data Leakage 방지 규칙 — 절대 위반 금지

> 이 규칙은 모든 워커가 반드시 준수해야 하며, 위반 시 전체 실험 결과가 무효화된다.

## 1. Lag / Rolling Feature의 Leakage 방지

### 핵심 원칙
- **예측 시점 t에서, t 이후의 정보는 절대 사용할 수 없다.**
- lag feature: `target_{t-k}` 에서 k ≥ prediction_length 이어야 안전
- rolling feature: `rolling_mean(target, window)` 는 window의 마지막이 t-prediction_length 이전이어야 함

### Lag Feature 규칙

```
prediction_length = H 일 때:

안전한 lag:
  target_{t-H}, target_{t-H-1}, target_{t-H-24}, target_{t-H-168}
  → 모두 예측 시작 시점보다 과거

위험한 lag (LEAKAGE):
  target_{t-1}, target_{t-2}, ..., target_{t-H+1}
  → 예측 구간 내 값을 참조 → 실제 서빙 시 존재하지 않는 정보

예외: known covariate (calendar, 예보 등)는 미래값 사용 가능
```

### Rolling Feature 규칙

```
rolling_mean(target, window=W) at time t:
  → 실제 계산: mean(target[t-W+1 : t+1])
  → 이 중 target[t-H+1 : t+1]은 예측 시점에 알 수 없음

안전한 rolling:
  rolling_mean(target, window=W).shift(H)
  → 또는 rolling 계산 후 H만큼 shift

위험한 rolling (LEAKAGE):
  rolling_mean(target, window=W)  # shift 없이
  → 직전 H개 스텝의 target 포함
```

### 검증 방법

```python
# 모든 feature에 대해 이 검증을 수행할 것:

def check_leakage(df, feature_col, target_col, datetime_col, prediction_length):
    """feature가 미래 target 정보를 포함하는지 검증."""
    # 1. feature를 계산하는 데 사용된 target의 최신 시점 확인
    # 2. 그 시점이 (현재 - prediction_length) 이전인지 확인
    # 3. 위반 시 즉시 에러
    pass
```

## 2. Train / Validation / Test 분할

- **시간 순서 엄수 — random split 절대 금지**
- train의 마지막 시점 < validation의 첫 시점
- validation의 마지막 시점 < test의 첫 시점
- **train-val 사이에 gap = prediction_length** (leakage 방지)

```
|--- train ---|-- gap(=H) --|--- val ---|-- gap(=H) --|--- test ---|
              ^                         ^
              train 끝                  val 끝
              val에서 train 마지막 H개 참조 불가
```

### ⛔ Test Set 절대 규칙

**Test set은 최종 평가에만 사용한다. 모델 선택/튜닝에 절대 사용하지 않는다.**

- 모델 선택: validation 성능으로만 판단
- 하이퍼파라미터 튜닝: validation 성능으로만 판단
- feature 선택: validation 성능으로만 판단
- test set은 최종 선택된 모델에 대해 **딱 1번** 평가
- test 결과를 보고 모델을 수정하면 → 그건 더 이상 test가 아님 → 일반화 성능 보장 불가

```
올바른 흐름:
  1. train으로 학습
  2. val로 모델 선택/튜닝 (여러 번 반복 가능)
  3. 최종 모델 확정
  4. test로 1회 평가 → 이 숫자가 진짜 성능

금지되는 흐름:
  1. train으로 학습
  2. test로 평가 → "성능 안 좋네" → 모델 수정 → 다시 test
  → 이건 test에 과적합한 것
```

### Time-Aware Cross Validation + Refit 프로토콜

**단일 holdout보다 Time-Aware CV를 기본으로 사용한다.**

```
방법: Sliding Window (또는 Expanding Window) CV

Fold 1: |--- train_1 ---|-- gap --|-- val_1 --|
Fold 2:    |--- train_2 ---|-- gap --|-- val_2 --|
Fold 3:       |--- train_3 ---|-- gap --|-- val_3 --|
...
→ 각 fold의 val 메트릭 평균 + 표준편차로 모델 선택
→ 분산이 큰 모델은 불안정 → 페널티
```

**규칙:**
- fold 수 ≥ 3 (최소 3개 시간 창으로 검증)
- 각 fold 사이 gap = prediction_length (leakage 방지)
- fold별 메트릭을 모두 기록 (평균만 보면 불안정성이 가려짐)
- **모델 선택은 CV 평균 메트릭으로, 최종 보고는 test set으로**

**Refit (최종 모델 확정 후 필수):**

```
올바른 최종 흐름:
  1. Time-Aware CV로 최적 모델 + HP 선택
  2. 최종 모델 확정 (여기서 모델 구조/HP 동결)
  3. train + val 전체 데이터로 refit (test는 제외)
  4. refit된 모델로 test set 1회 평가 → 이 숫자가 최종 성능

이유:
  - CV 중 val로 쓰였던 데이터도 학습에 활용 → 더 많은 데이터로 학습
  - HP는 이미 확정되었으므로 과적합 위험 없음
  - 실제 배포 시에도 가용 데이터 전부로 refit하는 것이 표준
```

**⛔ Refit 시 주의:**
- refit 후 HP를 다시 튜닝하면 안 됨 (그건 test에 대한 간접 과적합)
- refit 전후 train loss가 크게 다르면 데이터 분포 변화 의심 → 경고

## 3. Feature Engineering 시 절대 규칙

- **target에서 파생된 feature는 반드시 shift(prediction_length) 적용**
- **known covariate는 미래값 사용 가능** (calendar, weather forecast 등)
- **unknown covariate는 반드시 과거값만** (실측 수요, 실제 가격 등)
- **혼동될 때는 안 넣는 게 낫다** — leakage 있는 feature 하나가 전체를 오염시킴

## 4. 데이터 값 절대 추측 금지

- **파일을 읽지 않고 컬럼명이나 값을 추측하지 않는다**
- **컬럼명이 불확실하면 반드시 데이터를 먼저 로드하여 확인한다**
- **통계값(mean, std 등)을 코드 실행 없이 말하지 않는다**
- **"아마 이럴 것이다"는 금지 — 확인하고 말한다**

## 5. Learning Curve를 통한 과적합/일반화 판단

**모든 학습 후 learning curve를 반드시 확인한다.**

```python
# Learning curve 패턴
train_sizes = [0.2, 0.4, 0.6, 0.8, 1.0]  # 데이터 비율
for size in train_sizes:
    train_subset = train[:int(len(train) * size)]
    model.fit(train_subset)
    train_score = evaluate(model, train_subset)
    val_score = evaluate(model, val)
    print(f"size={size:.0%}: train={train_score:.4f}, val={val_score:.4f}")
```

판단 기준:
| 패턴 | 진단 | 대응 |
|------|------|------|
| train↓ val↓ (둘 다 감소, gap 좁음) | 정상 학습 | 데이터 더 모으면 개선 |
| train↓ val→ (val 정체) | 모델 용량 부족 | 더 복잡한 모델 또는 feature 추가 |
| train↓↓ val↑ (val 증가) | **과적합** | 모델 단순화, regularization, 데이터 증량 |
| train→ val→ (둘 다 정체) | ceiling 도달 | 새로운 외부 데이터 필요 |

**train-val gap이 클수록 과적합.** Gap이 줄어드는 추세면 건강한 학습.

## 6. Hallucination 방지 체크리스트

매 워커 실행 후 자동 확인:
- [ ] 데이터를 실제로 로드했는가? (추측이 아닌 실행 결과인가?)
- [ ] 모든 컬럼명이 실제 데이터에 존재하는가?
- [ ] lag/rolling feature에 shift(prediction_length) 적용했는가?
- [ ] known/unknown covariate 분류가 정확한가?
- [ ] train/val/test 분할이 시간 순서를 따르는가?
- [ ] gap이 prediction_length 이상인가?
