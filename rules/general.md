# CBALLM AutoML — 범용 시계열 모델링 규칙·노하우·업계 표준

> 이 문서는 LLM 오케스트레이터(CBALLM)의 각 워커가 모델링 의사결정 시 참조하는  
> **일반화된 규칙 + 실전 노하우 + 업계 표준**을 집대성한 것이다.  
> 어떤 도메인의 시계열에든 적용 가능하도록 설계되었다.

---

## 1. DATA PROFILING — Scout 워커용 규칙

### 1.1 데이터 품질 진단 (모델링 전 필수)

**결측·gap 탐지**  
시계열 frequency가 불규칙(IRREG)이면 매 prediction step마다 리샘플링 오버헤드가 발생한다. 전처리 단계에서 gap을 ffill/interpolate로 메우고 frequency를 확정할 것. 결측 비율이 feature별로 크게 다르면 해당 feature는 제외 후보로 먼저 플래그.

**이상치·구조적 예측 불가 영역 분리**  
target이 물리적·제도적 이유로 극단값(0, 상한/하한 clamp)을 찍는 구간이 존재하는 경우, 이는 연속값 예측 모델의 구조적 한계 영역이다. 이런 구간은 별도 분류기(binary/multi-class)로 처리하고, 메인 회귀 모델의 평가에서는 분리 보고할 것. 평가에 섞이면 전체 메트릭을 오염시킨다.

**Regime 진단**  
외부 충격(정책 변경, 팬데믹, 시장 구조 전환 등)으로 target의 운동성(dynamics)이 시기별로 다른지 확인. 전체를 통으로 쓰면 "평균적인 과거"를 학습해 현재 성능이 떨어진다. 확인 방법: rolling mean/std의 수준 변화, CUSUM/Bai-Perron 구조 변화 검정, 또는 regime detection 알고리즘.

**Target 포화 탐지**  
covariate를 바꿔도 MAE가 거의 움직이지 않으면, feature engineering이 아닌 모델 구조 변경이 필요하다는 신호. 더 이상 같은 모델 프레임에서 feature만 바꿔봤자 한계.

### 1.2 학습 기간 선택 전략

- **최근 데이터 우선 원칙**: 시장/환경 구조가 바뀌었다면 과거 전체를 통으로 쓰는 건 마지막 옵션. 최근 regime의 데이터만으로 먼저 baseline을 잡을 것.
- **Regime-selective cherry-pick**: 과거에서 현재와 유사한 regime 구간만 선별 append. 과거 전체를 쓰지 않되, 극단 이벤트 학습 데이터는 확보하는 전략. 실험적으로, 전체 데이터 사용 대비 성능이 더 좋은 경우가 많다.
- **학습 길이 ≥ 계절 주기 × 2 이상**: 연간 계절성이 있으면 최소 2~3년. DL 모델(DeepAR, TFT 등)은 통계 모델보다 더 많은 데이터 필요.
- **Retraining window vs 전체**: 5년 전 데이터가 지금과 무관하다면, 최근 1~2년 sliding window retraining이 전체 데이터 대비 성능 우위. (2026 production drift 가이드에서도 "ancient patterns from 5 years ago may no longer be relevant" 경고)

### 1.3 Feature vs Target 관계 사전 점검

- **단위·스케일 확인**: target과 직접 비교 가능한 단위의 feature가 가장 효과적. 변환이 필요한 raw 값은 모델이 한 단계 더 학습해야 하므로 비효율.
- **Known vs Unknown covariate 엄격 구분**: 예측 시점에 미래값을 알 수 있는 feature(calendar, 예보, 확정 스케줄)는 known covariate, 실현 후에만 알 수 있는 feature(실제 수요, 실측값)는 unknown covariate. 혼동하면 data leakage.
- **Data leakage 선제 차단**: target-derived feature(rolling mean of target 등)는 반드시 예측 시점 이전 정보만 사용. "well-defined forecast horizon이 없으면 leakage가 발생한다" (Karmaker 2025, Feature Engineering in Time Series Forecasting).

---

## 2. FEATURE ENGINEERING — Engineer 워커용 규칙

### 2.1 핵심 원칙

**원가/레벨 자체보다 변동성 시그널로 변환**  
외부 변수의 절대 수준보다 변화율/변동성/이탈도가 예측에 더 유용한 경우가 많다. 평상시에는 ~0이라 multicollinearity 없고, 이벤트 시에만 activate되는 feature가 가장 robust하다.

```
변환 패턴:
  change_rate = (x_t - x_{t-N}) / x_{t-N}
  volatility = rolling_std(x, window)
  deviation = x_t - rolling_mean(x, long_window)
  shock_flag = 1 if |change_rate| > threshold else 0
```

**넣을 수 있다고 다 넣지 않는다**  
직관적으로 관련 있어 보이는 feature도 실험적으로 underperform 확인되면 제거. 잘못된 feature는 noise를 추가할 뿐. "Feature engineering에서 가장 중요한 결정은 무엇을 빼는가이다."

**도메인 지식 기반 파생 feature > raw feature**  
두 변수의 비율, 차이, 교차항이 raw 변수 자체보다 signal-to-noise ratio가 높은 경우가 빈번하다. 예: `supply_demand_ratio`, `load_factor`, `capacity_utilization`.

**교차항(interaction)으로 동시 발생 이벤트 포착**  
`feature_A × feature_B` — 둘 다 높을 때만 값이 커져서 동시 발생 조건 감지. 비선형 모델(GBM)은 자동으로 잡지만, 명시적 교차항이 linear 모델과 앙상블 diversity를 높인다.

**미분 feature(ramp/변화율)로 선행 예측**  
`ramp = x_t - x_{t-k}` — 급변이 시작되는 초기 신호를 포착. 선행 지표(leading indicator) 역할.

### 2.2 표준 Feature 카테고리 (모든 시계열에 적용)

| 카테고리 | 예시 | 역할 |
|---|---|---|
| **Lag features** | target_{t-1}, target_{t-24}, target_{t-168} | 자기상관 패턴 |
| **Rolling statistics** | rolling_mean, rolling_std, rolling_min/max (다양한 window) | 추세·변동성 |
| **Calendar features** | hour_sin/cos, dow_sin/cos, month_sin/cos, is_holiday, is_weekend | 주기성 (Fourier encoding 권장 — one-hot보다 연속적이고 차원 절약) |
| **Differencing** | target_t - target_{t-1}, target_t - target_{t-season} | 정상성 확보, 변화량 포착 |
| **EWMA** | exponentially_weighted_mean(α) | 최근 값에 가중치, 장기 추세 동시 포착 |
| **Group-based stats** | category별 mean/median/percentile | 계층 구조가 있는 multi-series에 필수 |
| **Domain-specific** | ratio, interaction, threshold flag | 도메인 지식 인코딩 |

### 2.3 Feature 진단 자동화

| 방법 | 포착하는 관계 | 적용 시점 |
|---|---|---|
| **Granger Causality** | 시차 선형 인과 | feature → target 방향 인과 존재 여부 |
| **Transfer Entropy** | 시차 비선형 인과 | Granger로 안 잡히는 비선형 관계 |
| **Mutual Information** | 동시점 비선형 상관 | feature-target 전반적 정보 공유량 |
| **Cointegration (Johansen/Engle-Granger)** | 장기 균형 관계 | 비정상 시계열 간 공적분 |
| **Partial Autocorrelation (PACF)** | 직접 자기상관 | 최적 lag 차수 결정 |

진단 결과 유의하지 않은 feature는 ablation 대상. 통과한 feature만 모델에 투입.

### 2.4 Ablation 프로토콜

- 모든 feature 추가/제거는 **반드시 ablation study** 동반. 단일 변수 제거(leave-one-out) + 그룹 제거.
- 실험 추적 시스템(MLflow 등)에 모든 ablation 결과 기록. run_name에 feature set 명시.
- **정상/극단 구간 분리 평가** — "정상 시 성능 유지 + 극단 이벤트 대응력 향상"을 동시에 확인해야 한다. 전체 MAE만 보면 극단 대응력이 가려진다.

---

## 3. MODEL ARCHITECTURE — Architect 워커용 규칙

### 3.1 앙상블 설계 철학

**"모델 수보다 다양성이 중요하다."** (M4/M5 대회 핵심 교훈)

앙상블이 작동하려면 모델 간 **error correlation이 낮아야** 한다. 4개 축으로 다양성 확보:

| 축 | A 극 | B 극 | 왜 중요한가 |
|---|---|---|---|
| **Local vs Global** | ETS, ARIMA, Theta (시리즈별 독립) | DeepAR, TFT, LightGBM (cross-series) | 데이터 많으면 Global 유리, 적으면 Local robust |
| **Linear vs Nonlinear** | ARIMA, DLinear, Linear Regression | GBM, TCN, TFT | 선형 모델이 잡는 clean trend/seasonality는 비선형이 못 잡는 경우 있음 |
| **Point vs Distributional** | LightGBM, XGBoost | DeepAR, TFT, MQCNN | 불확실성 정량화 필요 시 distributional 필수 |
| **Architecture Family** | Statistical / Tree / MLP / RNN / Transformer / CNN | — | inductive bias가 서로 달라 error pattern 상이 |

### 3.2 Preset 레시피 (범용)

```
fast (5~6개, <10분):
  SeasonalNaive, ETS, LightGBM(lag features), DeepAR, DLinear, Theta

medium (8~10개, 30~60분):
  + AutoARIMA, CatBoost, NHiTS, PatchTST or TFT

best_quality (12~15개, 수 시간):
  + CrostonSBA(intermittent demand용), MSTL(multi-seasonal)
  + iTransformer, TimesNet, N-BEATS(generic+interpretable 둘 다)
  + XGBoost(다른 HP space), TCN/WaveNet
```

**Preset 선택 가이드:**
- 시리즈 수 많고 빠른 결과 필요 → fast
- 단일/소수 시리즈, 성능 중시 → best_quality
- DL 모델 포함 preset은 GPU 필수. CPU only면 Statistical + Tree 위주로.

### 3.3 Greedy Weighted Ensemble (Caruana et al. 2004)

```
알고리즘:
1. 모든 모델을 validation loss 기준 정렬
2. Best 모델로 ensemble 초기화
3. 각 라운드: 나머지 모델 중 ensemble에 추가 시 val loss 최소화하는 모델 선택
   → with replacement 허용 (자연스럽게 weighted average 형성)
4. 개선 없을 때까지 반복 (통상 20~50 라운드면 수렴)
5. weight = 각 모델 선택 횟수 / 총 라운드 수
```

핵심 성질:
- **Diversity 자동 반영**: 이미 비슷한 모델이 있으면 추가 이득 없어 자연스럽게 다양한 모델 선택됨.
- **Validation은 반드시 temporal split (last window)**. Random split 절대 금지.
- **Greedy ensemble > Stacking (시계열에서)**: overfitting 리스크 낮고, val 데이터가 적은 시계열 특성에 유리. (M4/M5 실증)
- 모델 zoo가 커도 greedy selection이 필터 역할 → "대충 다 넣어도 작동하지만, 학습 시간 비용은 증가."

### 3.4 Prior Injection — 도메인 사전지식을 체계적으로 녹이는 방법론

> **핵심 철학**: 데이터만으로는 학습할 수 없거나 학습에 과도한 데이터가 필요한 패턴이 존재한다.  
> 도메인 사전지식(prior)을 모델에 체계적으로 주입하면, **데이터 효율성·일반화 성능·물리적 타당성**이 동시에 개선된다.  
> 단, 잘못된 prior는 성능을 악화시키므로 **반드시 데이터 기반 검증(ablation)을 거쳐야** 한다.

#### Prior Injection 워크플로우

```
Step 1: 도메인 서치 — 해당 분야의 알려진 법칙, 경험칙, 인과관계를 수집
Step 2: 데이터 EDA — 수집한 prior가 실제 데이터에서 관찰되는지 확인
Step 3: 확신도 분류 — 각 prior를 확신도에 따라 Data / Architecture / Loss 중 어디에 주입할지 결정
Step 4: 구현 & Ablation — 주입 후 반드시 with/without 비교. 성능 미기여 시 제거.
Step 5: 반복 — Critic의 피드백에 따라 prior 추가/수정/제거
```

#### 3-Bias Taxonomy (PIML 표준 분류, Karniadakis et al.)

Physics-Informed Machine Learning 분야에서 **3가지 bias를 통한 prior 주입**이 표준 분류로 자리잡았다 (Springer 2025 survey, ACM Computing Surveys 2025, Applied Sciences 2023). CBALLM은 이 분류를 시계열 예측에 맞게 적용한다.

**① Observational Bias — 데이터 레벨 주입**

prior를 **feature engineering 또는 data augmentation**으로 반영. 모델 구조를 건드리지 않으므로 가장 범용적이고 적용 비용이 낮다.

| 방법 | 설명 | 예시 |
|---|---|---|
| **Domain feature** | 도메인 지식을 파생 변수로 인코딩 | supply/demand ratio, capacity utilization, ramp rate |
| **변동성 시그널 변환** | 레벨 대신 변화율/이탈도 feature | change_rate, rolling_std, deviation_from_MA, shock_flag |
| **Simulation 데이터 보강** | 물리 시뮬레이터로 부족한 regime의 합성 데이터 생성 | 극단 이벤트 시나리오 시뮬레이션 |
| **Regime-selective sampling** | 과거에서 현재와 유사한 regime만 cherry-pick | 특정 이벤트 구간만 학습 데이터에 append |

**적합한 prior**: 불확실한 가설, "이 변수가 관련 있을 것 같다" 수준. 모델이 스스로 유용성을 판단하도록 위임.

**장점**: 모델 아키텍처 변경 불필요, 어떤 모델에든 적용 가능, AutoML 파이프라인과 자연스럽게 호환.  
**한계**: 모델이 prior를 무시할 수 있음, feature가 많아지면 noise 증가 가능.

**② Inductive Bias — 모델 아키텍처 레벨 주입 (Hard Constraint)**

prior를 **네트워크 구조 자체에 내장**하여 모델이 물리적으로 불가능한 출력을 절대 생성하지 못하게 한다. 가장 강력하지만 모델 설계에 직접 개입해야 하므로 비용이 높다.

| 방법 | 설명 | 예시 |
|---|---|---|
| **Monotonic network** | 특정 입력에 대해 출력이 단조증가/감소하도록 구조 보장 | 수요↑→가격↑, 온도↑→냉방부하↑ |
| **Positivity / bound constraint** | 출력 범위를 물리적 가능 범위로 제한 | softplus로 양수 보장, sigmoid로 [0,1] 제한 |
| **Fourier time encoding** | 주기적 시간 패턴을 사전에 아키텍처에 내장 | sin/cos encoding of hour, day, week |
| **Equivariance / symmetry** | 입력 변환에 대한 출력 불변성/등변성 보장 | 회전 대칭, 치환 불변 (graph neural net) |
| **Regime gate** | 정상/극단 구간을 다른 head로 처리하는 분기 구조 | normal head + extreme head, soft switch |
| **Skip connection (residual)** | base 예측에 delta만 학습하도록 구조 유도 | y = base_prediction + f(x) |
| **Causal architecture** | 미래 정보가 과거로 흐르지 않도록 구조적 차단 | causal convolution, masked attention |

**적합한 prior**: 확실한 물리 법칙, 정의상 성립하는 관계. "이건 절대 위반될 수 없다" 수준.

**장점**: 모델이 위반할 수 없으므로 가장 강력한 보장. 데이터 부족 시에도 물리적 타당성 유지. data efficiency 향상 효과가 가장 크다 (Springer 2025: "architecture-level inductive bias has advantages over data augmentation and loss functions in prediction performance and data efficiency").  
**한계**: 모델 설계에 직접 개입 필요, 잘못된 hard constraint는 모델을 망칠 수 있음, AutoML zoo의 범용 모델에는 적용 어려움 (custom model에 적합).

**③ Learning Bias — Loss 레벨 주입 (Soft Constraint)**

prior를 **loss function의 추가 항(penalty term)**으로 반영. 위반 시 비용을 부과하되, 데이터가 강하게 반대하면 학습이 override 할 수 있다.

| 방법 | 설명 | 예시 |
|---|---|---|
| **Smoothness penalty** | 인접 시점 예측값의 급변을 억제 | L2 norm of (y_t - y_{t-1}) |
| **Monotonicity penalty** | 특정 조건에서 출력 순서 관계 유도 | penalty if f(x_high) < f(x_low) |
| **Regime-conditional weighting** | 구간별로 다른 loss 가중치 | 극단 구간에 5× weight, 정상 구간에 1× |
| **Asymmetric loss** | 과대/과소예측에 다른 비용 | quantile loss, Linex loss |
| **Convexity / shape constraint** | 출력의 형태(볼록/오목)를 유도 | 공급곡선 형태 유도 |
| **Physical residual** | PDE/ODE 잔차를 loss에 추가 | 에너지 보존, 질량 보존 법칙 위반 penalty |
| **Consistency regularization** | 다중 출력 간 정합성 | 부분 예측의 합 = 전체 예측 |

**적합한 prior**: "대체로 맞지만 예외가 있을 수 있다" 수준의 경험칙. 정상 구간에서는 맞지만 극단 구간에서는 깨질 수 있는 관계.

**장점**: hard constraint보다 유연하여 잘못된 prior의 피해가 제한적. weight 조절로 prior의 강도를 튜닝 가능. 기존 모델에 loss term 추가만으로 적용 가능.  
**한계**: prior가 약하면 모델이 무시할 수 있음. loss term 간 경쟁(gradient conflict) 발생 가능 — curriculum training으로 완화.

#### 확신도 기반 의사결정 매트릭스

```
Prior 수집 후, 각 prior에 대해:

  Q1. "이 관계가 데이터에서 관찰되는가?" (EDA 확인)
    → NO → 제거 (데이터와 모순되는 prior는 해롭다)
    → YES → Q2로

  Q2. "이 관계에 예외가 존재하는가?"
    → NO (절대적 법칙) → Architecture (hard constraint) ②
    → YES (대체로 맞음) → Q3로

  Q3. "예외 구간을 명시적으로 정의할 수 있는가?"
    → YES → Loss (soft constraint) + regime-conditional ③
    → NO  → Feature로 제공 (모델이 판단하도록 위임) ①

  모든 주입 후:
  Q4. "with/without ablation에서 성능 기여가 확인되는가?"
    → NO → 제거. 안 넣는 게 낫다.
    → YES → 유지.
```

#### Prior 충돌 감지 & 해소

복수의 prior가 서로 충돌할 수 있다. 예: "출력은 항상 양수(hard)" vs "극단 이벤트 시 음수 가능(도메인 사실)". 충돌 발견 시:

1. **설계 단계에서 감지**: 모든 prior를 리스트업하고 pair-wise 충돌 가능성 검토. 이 단계를 건너뛰면 배포 후에 발견됨.
2. **Regime 분리로 해소**: "정상 구간에서는 prior A 적용, 극단 구간에서는 prior A 비활성화"와 같이 구간별로 다른 constraint set 적용.
3. **우선순위 부여**: hard constraint가 soft보다 우선. 충돌 시 soft constraint의 weight를 낮추거나 해당 구간에서 비활성화.

#### Loss 설계 시 Curriculum Training 전략

loss term이 3개 이상일 때, 모든 loss를 처음부터 동시에 켜면 gradient conflict로 학습 불안정.

```
Phase 1 (warm-up): data loss만 활성화 — 모델이 대략적 패턴을 먼저 학습
Phase 2 (ramp-up): constraint loss를 서서히 증가 (λ: 0 → target)
Phase 3 (full): 모든 loss term 활성화, 전체 최적화

각 phase 전환 조건:
  - val loss가 plateau에 도달
  - 또는 사전에 정한 epoch 수 경과
```

#### Prior 주입의 실전 효과 (검증된 패턴)

| 상황 | Prior 없이 | Prior 주입 후 | 핵심 원인 |
|---|---|---|---|
| 데이터 충분, 정상 구간 | 성능 좋음 | 거의 동일 또는 미미한 개선 | 데이터가 이미 패턴을 충분히 제공 |
| 데이터 부족 | 과적합 또는 불안정 | 대폭 개선 | prior가 regularizer 역할 |
| 극단 이벤트 (1~3% 빈도) | 대부분 놓침 | regime gate + loss weighting으로 포착 | 다수 정상 구간에 묻혀 소수 이벤트 학습 실패 방지 |
| 분포 변화 (concept drift) | 급격한 성능 하락 | 완만한 하락 (robust) | prior가 anchor 역할, 데이터 변해도 물리적 타당성 유지 |
| 해석 가능성 요구 | black box | 구조적 해석 가능 | monotonic branch, regime gate 등이 의미 있는 구조 제공 |

#### AutoML 파이프라인에서의 Prior 활용 전략

CBALLM의 AutoML 맥락에서 prior 주입은 3가지 경로로 적용:

**경로 A: 범용 모델 zoo에 feature-level prior (Observational Bias)**  
AutoGluon/LightGBM 등 범용 모델을 그대로 쓰되, Engineer 워커가 도메인 feature를 생성. 가장 낮은 비용으로 가장 넓은 적용 범위. 모든 프로젝트의 기본.

**경로 B: Custom 모델을 zoo에 추가 (Inductive + Learning Bias)**  
도메인 특화 아키텍처(monotonic net, regime gate 등)와 custom loss를 가진 모델을 별도로 만들어 앙상블 zoo에 한 구성원으로 추가. 범용 모델과 error pattern이 달라서 앙상블 diversity에 크게 기여.

**경로 C: 후처리 override (별도 모듈)**  
메인 모델의 출력을 분류기/규칙으로 보정. 구조적으로 예측 불가한 영역(극단 이벤트)을 별도 처리. 메인 모델은 건드리지 않으므로 정상 구간 성능 무영향.

**핵심**: 경로 A → B → C 순서로 점진적으로 적용하되, 각 경로 추가 시 ablation으로 효과 확인.

### 3.5 Regime-Aware 설계

**정상 구간과 극단 구간은 다른 세계다.** 하나의 모델/constraint가 양쪽에 다 맞을 수 없다.

- **Regime gate**: normal head / extreme head를 soft switch하는 learnable gate — end-to-end 최적화. 전체 데이터의 99%인 정상 구간에 묻혀 1%인 극단이 무시되는 문제 방지.
- **Regime-aware ensemble**: `final = base_pred × (1-w) + extreme_pred × w`, w는 volatility/anomaly score 기반 동적 조절.
- **Extreme override**: 별도 분류기의 확률로 soft override. `pred × (1-prob) + fallback × prob`. 평상시 성능을 전혀 건드리지 않으면서 극단 구간만 보정.

### 3.6 구조 설계 패턴

**Cascade (직렬 분해)**  
target을 구성 요소로 분해하고 순차 예측. 예: level → deviation → residual. 또는 base → spread → derived. 각 단계의 모델이 더 단순한 문제를 풂.

**Residual (잔차 보정)**  
`최종 = 규칙 기반/도메인 모델 + ML 앙상블(잔차)`. 도메인 모델이 큰 구조를 잡고, ML이 비선형 잔차를 학습. 도메인 모델의 설명력과 ML의 유연성을 결합.

**Hierarchical (계층적)**  
multi-series에서 상위 수준(총합) 예측과 하위 수준(개별) 예측의 일관성 확보. Top-down, bottom-up, reconciliation (MinTrace 등).

### 3.7 모델 선택 시 주의사항

- **Simple model(DLinear, ETS, Theta)이 복잡한 DL보다 평시를 더 잘 잡는 경우가 빈번하다.** 앙상블에서 이들을 빼면 성능이 떨어진다 (M4 실증).
- **단일 도메인 모델이 앙상블을 이기기 어렵지만**, 그 모델을 앙상블에 추가하면 앙상블이 더 이긴다. 최적 전략 = 도메인 특화 모델을 앙상블의 한 구성원으로 포함.
- **LightGBM + lag/rolling feature는 시계열에서 DL과 대등** (M5에서 입증). Tabular feature engineering이 중요.
- **Foundation model(Chronos-2, TimesFM 등)은 zero-shot baseline으로 유용**하지만, 도메인 특화 fine-tuning 없이는 전용 모델을 넘기 어렵다.

---

## 4. TRAINING — Trainer 워커용 규칙

### 4.1 Validation 프로토콜

**Temporal split only — random split 절대 금지.**

| 방법 | 설명 | 적용 |
|---|---|---|
| **Single holdout (last window)** | 마지막 N일을 test | 가장 빠름, baseline |
| **Expanding window CV** | train을 점진 확장, val은 고정 크기 forward | 안정적 성능 추정 |
| **Sliding window CV** | 고정 크기 train window가 전진 | 최근 데이터에 가중, drift 있는 경우 유리 |
| **Walk-forward (rolling refit)** | 매 step마다 refit + 1-step-ahead 예측 | 실전 시뮬레이션에 가장 가까움 |

- **gap 설정**: train과 val 사이에 forecast horizon 크기의 gap. Data leakage 방지. (scikit-learn TimeSeriesSplit의 `gap` 파라미터)
- **Grid search는 val만 평가. Test는 선정된 config만 1회 평가.** 모든 조합에 test를 돌리는 건 사실상 test set에 대한 multiple testing → 과적합.

### 4.2 학습 전략

**Curriculum training (복잡한 custom 모델 시)**  
Phase 1: data loss만으로 대략적 패턴 → Phase 2: constraint loss 서서히 증가 → Phase 3: 전체 활성화. 초기에 잘못된 방향 수렴 방지. loss term이 3개 이상일 때 특히 유효.

**Asymmetric loss 설계**  
비즈니스에서 과대예측과 과소예측의 비용이 다르면, asymmetric loss(quantile loss, Linex loss)를 사용. 예: 재고 예측에서 과소가 더 비쌈 → quantile 0.7 타겟.

**Imbalanced regime 처리**  
극단 이벤트가 전체의 1~3%면, 해당 구간에 loss weight 3~5× 부여. 또는 Focal Loss로 easy sample down-weight.

**Auto featurization < 수동 feature + AutoML HPO**  
자동 featurization은 수동 feature engineering에 못 미치지만, 수동 feature를 넣고 HPO를 자동화하면 최강 조합이 된다 (ScienceDirect 2024 비교 연구 결론). Feature는 사람이, HP tuning은 기계가.

### 4.3 HPO 전략

- **Bayesian Optimization > Grid Search**: 시계열은 single eval 비용 높음 → sample-efficient HPO 필수.
- **Early stopping**: val metric 개선 없으면 조기 종료. patience 설정.
- **HPO + Ensemble 동시**: 모델별 HP search space를 preset에 포함시키면, 다양한 HP 설정의 모델들이 앙상블 diversity에도 기여.
- **Time limit 설정**: AutoML에 제한 없으면 무한 탐색. 리소스 예산에 맞게 제한.

### 4.4 실험 추적 필수 기록 (MLflow 등)

```
매 실험:
  - Hyperparameters (전체)
  - Feature set 명시 (어떤 feature가 들어갔는지)
  - Metrics: 전체 + 정상/극단 구간 분리
  - fit_time, predict_time
  - Leaderboard (모델별 성능 비교표)
  - Ensemble weight 분포
  - Prediction CSV + 시각화 (artifact)
  - Random seed (재현성)
  - 데이터 hash (dataset versioning)
```

---

## 5. EVALUATION & FEEDBACK — Critic 워커용 규칙

### 5.1 메트릭 체계

| 메트릭 | 용도 | 주의사항 |
|---|---|---|
| **MAE** | 주력 point forecast 메트릭 | target 단위로 직관적 해석 |
| **RMSE** | 큰 에러에 민감 | outlier 있으면 MAE와 괴리 |
| **MAPE/sMAPE** | 비율 비교 | target ≈ 0 구간에서 발산 → 해당 구간 제외 필요 |
| **CRPS** | 확률 예측 품질 | distributional model 필수 메트릭 |
| **Calibration (PI coverage)** | 예측 구간 신뢰도 | 90% PI가 실제로 90% 포함하는지 |
| **Precision/Recall/F1** | 이벤트 detection 전용 | classification 관점 평가 |
| **Winkler Score** | 예측 구간 sharpness + coverage | 좁으면서 커버하면 좋음 |

### 5.2 결과 해석 규칙

- **전체 MAE의 작은 개선(1~2%)이라도**, 극단 이벤트가 계절적으로 증가하는 시기에는 누적 개선폭이 훨씬 커진다. 항상 구간별 분리 평가.
- **Recall 높은데 FP 폭발**: 모델이 특정 feature(rolling count 등)에 과의존. Feature가 binary-like(0 or 1)면 중간 확률이 없어 threshold 조절이 무의미. → Feature 재설계 필요.
- **Normal 구간 MAE 변화 ≤ 0.01**: 무시 가능. FP 1~2건 영향.
- **Ensemble이 base와 비슷한 성능**: 테스트 구간이 정상(regime A)이면, regime B 전용 모델의 가중치가 ~0이 됨. 이는 **설계대로 동작한 것**. 진짜 검증은 regime B 구간에서.

### 5.3 Uncertainty Quantification (UQ)

**Conformal Prediction (CP) — 2025~2026 핵심 트렌드**

모델에 무관하게(model-agnostic) 예측 구간을 생성하는 post-hoc 프레임워크. 분포 가정 없이 finite-sample coverage guarantee를 제공.

- **Split Conformal**: calibration set의 residual 분위수로 PI 생성. 가장 단순.
- **ACI (Adaptive Conformal Inference)**: 시계열의 비교환성(non-exchangeability) 대응. coverage level을 online으로 동적 조정.
- **CPTC (NeurIPS 2025)**: change point가 있는 시계열에 특화. Switching Dynamics System과 결합하여 regime 전환 시 PI를 선제 조정. 짧은 시계열에서 특히 우위.

실전 적용: point forecast AutoML 위에 conformal prediction을 post-hoc으로 얹으면, 추가 학습 없이 불확실성 정량화가 가능하다 (Nixtla neuralforecast의 PredictionIntervals 참조).

### 5.4 피드백 루프 라우팅

```
Critic 판단 → 워커 라우팅:
  "feature 정보 부족 / 관련 없는 feature"    → Engineer
  "모델 구조가 패턴에 안 맞음"                → Architect
  "학습 설정/HP 문제"                        → Trainer
  "데이터 품질 / regime 미파악"               → Scout
  "구조적으로 예측 불가한 영역"               → 별도 후처리 모듈 설계 (classification + override)
```

### 5.5 개선 Ceiling 판단

- **모든 시계열에는 irreducible noise가 존재**. 외부 정보(외생 예보 등) 없이 도달 가능한 ceiling이 있다.
- **Ceiling 판단 기준**: 다양한 모델 + feature 조합에서 MAE가 수렴하면 ceiling에 도달한 것. 추가 개선은 새로운 외부 데이터 소스 확보가 필요.
- **극단 이벤트 예측 ceiling**: 첫 발생 이벤트(no prior signal)는 어떤 모델도 못 잡음. 연속 발생 이벤트는 rolling feature로 일부 포착 가능.

---

## 6. SERVING & PRODUCTION — 배포·운영 규칙

### 6.1 모델 서빙 아키텍처

| 방식 | 적합 상황 | 장점 | 단점 |
|---|---|---|---|
| **REST API (FastAPI/Flask)** | 예측 빈도 ≤ 분 단위, 내부 서비스 | 가장 간단, 빠른 구현 | 고처리량에 한계 |
| **Docker Sidecar** | 이종 언어 백엔드와 통합 | 같은 Pod, localhost 통신, 배포 독립 | 컨테이너 관리 필요 |
| **Triton / DJL Serving** | GPU 모델, 동적 배칭 필요 | 프로덕션 최고 처리량 | 설정 복잡 |
| **ONNX Runtime** | 경량 모델(GBM, MLP) 서빙 | 크로스 플랫폼, 빠른 추론 | 복잡한 앙상블/FM 변환 불가 |

**ONNX 변환 주의**: AutoML 앙상블(AutoGluon 등)의 전체 파이프라인 → ONNX 직접 변환은 불가인 경우 많음. Best single model 추출 후 개별 변환이 현실적.

### 6.2 Refit 전략

- **모델은 decay한다**: 시계열 모델은 시간이 지남에 따라 정확도가 떨어진다. 주기적 refit이 필수.
- **Refit 주기**: 데이터 변화 속도에 비례. 일변 데이터면 주간~월간. 실시간 데이터면 일간.
- **Refit full**: 최종 선택 모델을 전체(train+val) 데이터로 재학습 (bagging 없이 단일 모델 → 추론 속도 향상).
- **AutoML로 refit 자동화**: batch job으로 모델 재구축 + 이전 모델 대비 성능 비교 + 개선 시에만 교체.

### 6.3 Drift Detection & Monitoring

**Drift 유형:**
- **Data drift**: 입력 feature 분포 변화. KS test, PSI, Wasserstein distance로 탐지.
- **Concept drift**: 입력-출력 관계 변화 (같은 입력인데 다른 출력). 예측 정확도 모니터링으로 탐지.
- **Feature attribution drift**: retrain 반복으로 feature 가중치 자체가 변함.

**모니터링 파이프라인:**
```
Production data → schema/range 검증 → distribution drift 검정
  → threshold 초과 시 alert
  → 성능 저하 확인 시 refit trigger
  → 새 모델 canary 배포 → A/B 비교 → 승격 or 롤백
```

- **Feature store 사용 권장**: 학습과 추론에서 동일한 코드로 feature 생성 → training-serving skew 방지.
- **Shadow mode / canary**: 새 모델을 실서비스에 바로 적용하지 않고, shadow mode로 병행 운영 후 성능 확인.
- **Rollback 가능 필수**: "빠르게 롤백할 수 없다면 신뢰할 수 있는 ML 시스템이 아니다." (2026 production drift guide)

### 6.4 모델 메타데이터 (서빙팀 전달용)

```json
{
  "model_name": "...",
  "model_version": "semantic (major.minor.patch)",
  "created_at": "ISO 8601",
  "input_schema": {
    "features": ["feature_1", "feature_2", ...],
    "target": "target_name",
    "dtype": "float32",
    "shape": "(batch, seq_len, n_features)",
    "frequency": "h"
  },
  "output_schema": {
    "prediction_length": 24,
    "quantiles": [0.1, 0.5, 0.9],
    "dtype": "float32"
  },
  "preprocessing": {
    "scaling_method": "robust | standard | minmax",
    "missing_value_strategy": "ffill | interpolate",
    "included_in_onnx": true | false
  },
  "training_info": {
    "train_period": "start ~ end",
    "val_metric": "MAE",
    "val_score": 0.0,
    "best_model": "model_name",
    "ensemble_weights": {"model_a": 0.4, "model_b": 0.3, ...}
  }
}
```

---

## 7. 업계 표준 & 연구 기반 Best Practice 종합

### 7.1 M4/M5 Competition 교훈 (가장 큰 시계열 대회)

1. **Statistical + ML 조합 > 단일 DL 모델**: M4 우승팀(Smyl 2020) = ES-RNN (Exponential Smoothing + RNN). 순수 DL은 statistical baseline을 이기지 못한 경우가 많았음.
2. **LightGBM + lag/rolling feature ≈ DL 성능**: M5 Walmart 대회에서 GBM이 top-tier 차지.
3. **Simple model(ETS, Theta, Naive)이 앙상블에서 빠지면 성능 하락**: 다른 모델이 못 잡는 clean trend/seasonality를 커버.
4. **모델 수보다 다양성(diversity)이 중요**: 같은 family 5개 < 다른 family 5개.
5. **Greedy ensemble > Stacking**: robustness가 더 높음, 특히 val 데이터가 적은 시계열에서.

### 7.2 AutoML 플랫폼 공통 패턴 (AutoGluon, Azure, SageMaker, Databricks)

- **자동 lag/rolling/calendar feature 생성** → 하지만 수동 domain feature와 결합 시 성능 극대화.
- **Multi-model search + HPO** → Bayesian optimization 또는 genetic algorithm.
- **Temporal cross-validation** → 시간 순서 엄수.
- **자동 앙상블(greedy weighted or stacked)** → 최종 예측 생성.
- **MLflow/W&B 연동** → 모든 실험 추적.
- **Refit full** → 최종 모델을 전체 데이터로 재학습 (추론 최적화).

### 7.3 Conformal Prediction (NeurIPS 2025 트렌드)

- **Model-agnostic UQ**: 어떤 point forecast 모델 위에든 예측 구간 생성 가능.
- **Distribution-free**: Gaussian 가정 불필요. Heavy tail, skew에도 작동.
- **CPTC**: change point가 있는 시계열에서 state-conditional conformal prediction으로 adaptivity 향상. 짧은 시계열(T<300)에서 특히 기존 방법 대비 우위.
- **실전 적용**: point forecast AutoML + conformal PI가 "가장 적은 비용으로 가장 robust한 UQ"를 제공.

### 7.4 Production Drift Management (2025~2026 MLOps 표준)

- **Drift = 보장됨**: 환경 변화, 사용자 변화, 계절성이 있으면 drift는 불가피.
- **Monitoring = SRE 수준**: latency/error budget처럼 drift metric에 대시보드 + 알림 + 대응 runbook.
- **Tiered response**: 작은 drift → auto refit, 중간 → human review, 심각 → 긴급 개입.
- **Data drift를 자동 refit trigger로 쓰면 위험**: drift 원인이 data quality bug일 수 있음. 데이터 유효성 검증 후 refit.
- **Self-healing pipeline**: 자동 drift 탐지 → 자동 refit → canary 배포 → 성능 비교 → 승격/롤백. 수동 개입 최소화.

---

## 8. CODING STANDARDS

```
Python 3.10+, type hints everywhere
Ruff linting, mypy type checking
pytest (temporal split 기반 test), pytest-cov
Pydantic v2 for all data models / configs
Training configs as dataclasses (not loose dicts)
Reproducibility: seed everything, log all HP
Dataset versioning: hash-based checksums
CLI 기반 workflow (notebook 지양 — 재현성 + 자동화 유리)
Feature store 패턴: 학습/추론 동일 feature 생성 코드
Git tag로 모델 버전 + 데이터 snapshot 연결
```

---

## 요약: CBALLM 워커별 핵심 한 줄

| 워커 | 핵심 규칙 |
|---|---|
| **Scout** | 품질 진단 + regime 파악 먼저. Gap/이상치/구조변화 확인 없이 모델링 시작하지 말 것. **도메인 서치도 여기서** — 알려진 법칙·경험칙·인과관계를 수집하여 Prior 후보 리스트 생성. |
| **Engineer** | 레벨 자체보다 변동성 시그널. 모든 feature 추가는 ablation 동반. **Observational Bias (feature-level prior)** 담당. 넣을 수 있다고 다 넣지 않는다. |
| **Architect** | 다양성 기반 앙상블 + **확신도 기반 3-Bias Prior 주입** (Data/Architecture/Loss). Simple model 빼지 말 것. Regime-aware 필수. Custom 도메인 모델은 zoo에 한 구성원으로 추가. |
| **Trainer** | Temporal split only. Rolling refit 시뮬레이션. Grid search는 val만. **Curriculum training으로 multi-loss 안정화**. 실험 추적 필수. |
| **Critic** | 정상/극단 분리 평가. Ceiling 판단. Conformal PI로 UQ. 피드백은 원인 워커로 라우팅. **Prior 충돌 감지 및 ablation 기반 prior 유지/제거 판단**. |
