# CBALLM v2: Ontology-Guided Composable Neural Architecture for Time Series Forecasting
# 통합 프롬프팅 문서 — Architecture SPEC + Block Catalog + Rules

---

## 1. Project Identity

CBALLM은 시계열 예측 모델의 서브-아키텍처 블록을 **Knowledge Graph 온톨로지**로 정의하고,
블록 간 호환성 규칙을 KG에서 관리하며, LLM이 데이터 특성 기반으로 유효한 조합을 판단하고,
성공 경험이 레시피로 축적되는 시스템이다.

### 핵심 차별점
- **기존 AutoML** (AutoGluon, AutoTS): 모델 zoo에서 통째로 선택 → 앙상블. "왜?"에 대한 답 없음.
- **기존 NAS** (DARTS 등): 미분 가능 탐색으로 블록 조합. 온톨로지/호환성 보장 없음. GPU 비용 막대.
- **기존 LLM+AutoML** (AIDE, AutoML-Agent): LLM이 코드 생성. 구조적 호환성 보장 없음.
- **CBALLM v2**: KG 온톨로지가 호환성을 보장 → LLM은 유효 조합 내에서만 판단 → 경험 축적.

### 선행 연구와의 관계
| 선행 연구 | 빌려오는 것 | 차이점 |
|-----------|------------|--------|
| ExeKGLib (2023) | KG 기반 파이프라인 시맨틱 검증 | 시계열 모델 내부 블록 수준으로 세분화 |
| ML Ontology (MDPI 2026) | SPARQL constraint checking | 시계열 특화 블록 + LLM 판단 통합 |
| AIMKG (Frontiers 2024) | 파이프라인 메타데이터 축적 | 블록 수준 레시피 + 데이터 특성 매칭 |
| ZooCast (2025) | 모델-태스크 임베딩 매칭 | 모델이 아닌 블록 단위 매칭 |
| Decision Protocol (CBALLM v1) | 코드가 질문하고 LLM이 답하는 구조 | KG로 질문 범위를 제약 |

---

## 2. Hierarchical Ontology (4-Level)

모든 설계 판단은 아래 4개 레벨로 계층화된다. **상위 레벨이 먼저 결정되어야 하위 레벨 판단이 가능하다.**

```
Level 0: Pipeline Topology        ← rule-based (LLM 불필요)
  "encoder-only인가, encoder-decoder인가?"

Level 1: Functional Slot           ← rule-based
  "어떤 기능 슬롯이 필요한가? (Normalizer, Encoder, Mixer, Head...)"

Level 1.5: Computational Primitive ← KG 필터링 + LLM 판단 (핵심)
  "그 슬롯을 어떤 계산 원리로 수행할 것인가? (Linear, MLP, Attention, Conv...)"

Level 2: Block Configuration       ← rule 기본값 + LLM 미세조정
  "그 primitive의 세부 HP는? (d_model, n_heads, dropout...)"
```

### LLM 역할 분배
| Level | 판단 주체 | LLM 역할 | 난이도 |
|-------|----------|---------|--------|
| 0 | Rule engine | 없음 | - |
| 1 | Rule engine | 없음 | - |
| 1.5 | KG 필터 + LLM | **후보 중 선택** (매칭) | 낮음 (힌트 있으니까) |
| 2 | Rule 기본값 + LLM | **동의/조정** (숫자) | 매우 낮음 |

---

## 2.5. Forecasting Settings (S / MS / M)

시계열 예측은 입력/출력 변수 구성에 따라 세 가지 설정이 존재한다.
**이 설정이 파이프라인 전체에 영향을 미치므로, Level 0에서 topology와 함께 결정되어야 한다.**

### 설정 정의

| 설정 | 약어 | 입력 | 출력 | Head output_dim | 대표 유즈케이스 |
|------|------|------|------|----------------|---------------|
| Univariate | S | (B,T,1) | (B,H,1) | 1 | 단일 변수만 존재하는 데이터 |
| Multivariate→Single | MS | (B,T,C) | (B,H,1) | 1 | **가장 실용적.** C개 변수로 1개 타겟 예측 |
| Multivariate→Multi | M | (B,T,C) | (B,H,C) | C | 모든 변수를 동시 예측. 논문 벤치마크 표준 |

### 설정별 파이프라인 영향

```yaml
forecasting_setting_effects:

  S_univariate:
    input_dim: 1
    output_dim: 1
    encoder: "d_model은 작아도 됨 (입력 1개)"
    channel_mixer: "null (변수 1개이므로 의미 없음)"
    channel_independent: "해당 없음"
    head: "LinearHead(d_model → 1)"
    use_when: "변수가 타겟 하나만 있을 때"
    benchmark_note: "논문의 'Univariate' 실험 (PatchTST Table 6 등)"

  MS_multivariate_to_single:
    input_dim: C
    output_dim: 1
    encoder: "C개 변수를 d_model로 임베딩"
    channel_mixer: "선택적 — 변수 간 상관이 타겟 예측에 도움되면 사용"
    channel_independent:
      true: "각 변수가 독립적으로 인코딩 → Head에서 타겟 변수만 출력"
      false: "변수 간 정보를 섞은 뒤 → Head에서 1개 출력"
    head: "LinearHead(d_model → 1)"
    use_when: "외생변수가 타겟 예측에 도움되는 실무 시나리오 (에너지, 수요예측)"
    benchmark_note: "논문의 'MS' 실험. 일부 논문만 리포트."

  M_multivariate_to_multi:
    input_dim: C
    output_dim: C
    encoder: "C개 변수를 d_model로 임베딩"
    channel_mixer: "중요 — C개 변수 간 상호작용이 핵심"
    channel_independent:
      true: "각 변수가 독립적으로 자기 자신을 예측 (PatchTST 방식)"
      false: "변수 간 mixing 후 C개를 동시 예측 (iTransformer 방식)"
    head: "LinearHead(d_model → C) 또는 channel-independent면 LinearHead(d_model → 1) × C"
    use_when: "모든 변수를 동시에 예측해야 할 때. 논문 벤치마크 기본 설정."
    benchmark_note: "**대부분 논문의 기본 설정.** TSLib leaderboard = M 설정."
```

### Channel-Independent의 의미가 설정에 따라 달라짐

```
M 설정 + channel_independent=true (PatchTST 방식):
  각 변수 c에 대해: x[:,c] → Encoder → TemporalMixer → Head → y_hat[:,c]
  weight를 변수 간 공유하되, 각 변수가 독립적으로 자기 자신을 예측.
  장점: 과적합 방지, 변수 수에 무관한 파라미터.
  단점: 변수 간 상호작용 무시.

M 설정 + channel_independent=false (iTransformer 방식):
  x[:, all_channels] → Encoder → TemporalMixer → ChannelMixer → Head → y_hat[:, all_channels]
  변수를 토큰으로 취급하여 attention으로 상호작용 학습.
  장점: 변수 간 관계 포착 (Traffic/ECL에서 강점).
  단점: 변수 수에 비례하는 파라미터, 과적합 위험.

MS 설정:
  x[:, all_channels] → Encoder → TemporalMixer → [ChannelMixer?] → Head → y_hat[:, target_only]
  모든 변수를 입력하되, 타겟 하나만 예측.
  channel_independent=true면: 타겟 변수만 독립 처리 (외생변수 정보 미활용).
  channel_independent=false면: 외생변수 정보를 mixing 후 타겟 예측에 활용.
```

### KG 규칙에 반영

```yaml
forecasting_setting_rules:
  - setting: S
    force: {channel_mixer: null, output_dim: 1}
    
  - setting: MS
    force: {output_dim: 1}
    recommend: {channel_independent: false}  # 외생변수 활용을 위해
    note: "channel_independent=true면 외생변수 무시됨. 실무에서 보통 false."
    
  - setting: M
    force: {output_dim: C}
    note: "channel_independent 여부는 KG 추천 + LLM 판단"
    
  - auto_detect:
      if: "사용자가 target_col 1개 지정 + 다른 변수 존재"
      then: MS
      if: "사용자가 target_col 미지정 또는 '전부'"
      then: M
      if: "변수가 1개"
      then: S
```

---

## 3. Block Catalog (Single Source of Truth)

**모든 블록 정의는 이 섹션에만 존재한다.** 다른 곳에서 블록을 참조할 때는 이름만 사용.

### 참조 벤치마크 (Multivariate, pred_len=96, lookback=96)

출처: TSLib leaderboard, 각 논문 Table

| Dataset | DLinear | PatchTST | iTransformer | TimeMixer | TimesNet |
|---------|--------|----------|-------------|-----------|---------|
| ETTh1 | 0.386 | 0.370 | 0.386 | 0.375 | 0.384 |
| ETTh2 | 0.333 | 0.302 | 0.297 | 0.325 | - |
| Weather | 0.196 | 0.149 | 0.174 | 0.164 | 0.172 |
| ECL | 0.197 | 0.181 | 0.148 | 0.168 | - |
| Traffic | 0.625 | 0.481 | 0.395 | 0.462 | - |

관찰:
- 데이터 크고 변수 많으면 (Traffic, ECL) → iTransformer, PatchTST 강세
- 데이터 작고 단순하면 (ETT) → DLinear도 competitive
- 변수 간 상관 높으면 (Traffic) → channel mixing 모델 압승

---

### 3.1 Normalizer Slot

#### RevIN
- **출처**: Kim et al., 2022 (ICLR)
- **primitive**: instance_norm
- **shape**: (B,T,C) → (B,T,C), **has_reverse=true** (Head 이후 reverse 필수)
- **trainable_params**: 2C (affine γ, β)
- **config**:
  - affine: bool, default=true. "학습 가능한 scale/shift. 거의 항상 true"
  - subtract_last: bool, default=false. "mean 대신 마지막 값 빼기. strong trend 시 true"
- **use_when**: non-stationary (ADF p>0.05), 분포 shift 의심. 대부분의 SOTA 기본 채택.
- **use_conditions**: `is_stationary == false`
- **avoid_when**: 없음 (DatasetNorm과 중첩 가능)
- **stacks_with**: DatasetNorm. RevIN은 DatasetNorm 위에서 instance-level 추가 정규화로 동작한다. 논문 표준(PatchTST, iTransformer 등)이 이 조합을 사용하며, 실증적으로 DatasetNorm만 사용하는 것보다 성능이 우수하다.
- **prior_hint**: PatchTST, TSMixer, iTransformer 표준 구성요소. non-stationary에서 5-15% 개선.
- **⚠ reverse 시 target_idx 의존성**: output_dim < n_features일 때(S/MS setting), reverse는 target_idx에 해당하는 채널의 mean/std만 사용하여 역변환한다. Builder가 target_idx를 ForecastModel에 전달해야 함. 잘못된 target_idx는 역변환 결과를 완전히 오염시킨다.

#### RobustScaler
- **출처**: sklearn 표준
- **primitive**: statistical_norm
- **shape**: (B,T,C) → (B,T,C), has_reverse=true
- **config**: quantile_range: tuple, default=[25.0, 75.0]
- **use_when**: 이상치 매우 많은 데이터 (outlier_ratio > 5%)
- **use_conditions**: `outlier_ratio > 0.05`
- **avoid_when**: 이상치 적으면 RevIN으로 충분
- **stacks_with**: DatasetNorm (RobustScaler도 DatasetNorm 위에서 instance-level로 동작)

#### DatasetNorm (기본 내장)
- 명시적 블록이 아닌 **Trainer 내장 동작**. **항상 적용.**
- train split의 mean/std로 standardization.
- **stacks_with**: RevIN, RobustScaler — 이들은 DatasetNorm 위에서 instance-level 추가 정규화로 동작한다.

---

### 3.2 Encoder Slot

#### LinearProjection
- **출처**: DLinear (Zeng et al., 2023)
- **primitive**: Linear
- **shape**: (B,T,C) → (B,T,d_model)
- **params**: C × d_model
- **capacity**: minimal
- **config**: d_model: int[16~256], default=64. "C≤7이면 32~64, C>20이면 128~256"
- **use_when**: 기본값. 확신 없으면 이것. 어떤 TemporalMixer와도 호환.
- **use_conditions**: `default: true` (항상 후보에 포함)

#### PatchEmbedding
- **출처**: PatchTST (Nie et al., 2023, ICLR)
- **primitive**: Convolution (Conv1d)
- **shape**: (B,T,C) → **(B, n_patch, d_model)** — shape이 다름!
- **params**: C × patch_len × d_model
- **capacity**: low
- **config**:
  - d_model: int[32~256], default=128
  - patch_len: int[8~64], default=16. "dominant_period의 1/4~1/2 권장"
  - stride: int[4~32], default=8. "patch_len의 1/2"
- **use_when**: seq_len≥96, local temporal pattern 중요, lookback 336~512에서 최적
- **use_conditions**: `seq_len >= 96`
- **avoid_when**: seq_len < 64 (패치 수 부족), downstream Patch 전용 Mixer 없을 때
- **prior_hint**: PatchTST 핵심 혁신. patch_len=16, stride=8이 범용적. MSE 21% 감소.
- **⚠ shape 제약**: 후속 TemporalMixer가 PatchMLPMix 또는 PatchAttentionMix여야 함.

#### FourierEmbedding
- **출처**: FEDformer (Zhou et al., 2022), N-BEATS
- **primitive**: Fourier
- **shape**: (B,T,C) → (B,T,d_model)
- **capacity**: low
- **config**:
  - d_model: int[16~256], default=64
  - n_harmonics: int[1~20], default=3. "ACF>0.7인 주기 수에 맞추기"
  - learnable_freq: bool, default=false
- **use_when**: 강한 주기성 (ACF>0.7인 주기 1개 이상), 알려진 주기 존재
- **use_conditions**: `has_strong_seasonality == true`
- **avoid_when**: 주기성 약한 데이터 (max ACF<0.5)
- **prior_hint**: sin/cos로 명시적 주기 인코딩. n_harmonics=강한 주기 수로 맞추면 효과적.

---

### 3.3 TemporalMixer Slot

#### LinearMix
- **출처**: DLinear (Zeng et al., 2023, AAAI)
- **primitive**: Linear
- **shape**: (B,T,d_model) → (B,H,d_model)
- **capacity**: minimal, min_data: 500
- **config**: channel_independent: bool, default=true
- **use_when**: **항상 첫 번째 baseline으로 시도.** smooth/stationary, <5000행, 해석 필요.
- **use_conditions**: `default_baseline: true`
- **prior_hint**: DLinear 논문 핵심 — 단순 Linear가 Transformer 이김. ETTh1=0.386, Weather=0.196. **반드시 baseline 확보 후 복잡한 모델.**

#### MLPMix
- **출처**: TSMixer TMix-Only (Chen et al., 2023)
- **primitive**: MLP
- **shape**: (B,T,d_model) → (B,H,d_model)
- **capacity**: low, min_data: 2000
- **config**: hidden_dim[64~512]=256, dropout[0~0.5]=0.1, activation=GELU, n_layers[1~3]=1
- **use_when**: Linear baseline 후 비선형 의심 시. 2000~10000행.
- **use_conditions**: `n_rows >= 2000, linear_baseline_tested == true`
- **prior_hint**: TSMixer TMix-Only도 PatchTST에 근접. GELU>ReLU.

#### GatedMLPMix
- **출처**: TSMixer IC (Ekambaram et al., 2023, KDD)
- **primitive**: MLP + Gating
- **shape**: (B,T,d_model) → (B,H,d_model)
- **capacity**: medium, min_data: 5000
- **config**: hidden_dim[64~512]=256, dropout=0.1
- **use_when**: noisy feature 많을 때 (n_features>10). gating이 불필요한 feature 억제.
- **use_conditions**: `n_features > 10, n_rows >= 5000`
- **avoid_when**: feature ≤7이면 gating 오버헤드 불필요

#### PatchMLPMix
- **출처**: TSMixer CI (Chen et al., 2023)
- **primitive**: MLP
- **shape**: **(B, n_patch, d_model)** → (B,H,d_model)
- **capacity**: medium, min_data: 5000
- **config**: hidden_dim[64~512]=256, dropout=0.1
- **use_when**: PatchEmbedding과 함께. Attention보다 경량.
- **use_conditions**: `encoder == PatchEmbedding, n_rows >= 5000`
- **⚠ requires_encoder**: PatchEmbedding (hard requirement)
- **prior_hint**: PatchTST의 attention을 MLP로 대체 — 비슷한 성능, 더 빠름.

#### AttentionMix
- **출처**: Vanilla Transformer
- **primitive**: Attention
- **shape**: (B,T,d_model) → (B,H,d_model)
- **capacity**: high, min_data: 10000
- **config**: n_heads[2~16]=4, n_layers[1~6]=2, d_ff[64~1024]=256, dropout=0.1
- **use_when**: ≥10000행, 장거리 의존성. **단독 사용보다 PatchEmbedding과 조합 권장.**
- **use_conditions**: `n_rows >= 10000`
- **avoid_when**: <10000행, smooth 시계열, 실시간 추론 (O(T²))
- **prior_hint**: vanilla attention은 DLinear보다 나쁨 (Zeng 2023). Patching 필수.

#### PatchAttentionMix
- **출처**: PatchTST (Nie et al., 2023, ICLR)
- **primitive**: Attention
- **shape**: **(B, n_patch, d_model)** → (B,H,d_model)
- **capacity**: high, min_data: 10000
- **config**: n_heads[2~16]=4, n_layers[1~6]=3, d_ff[64~1024]=256, dropout=0.1, positional_encoding=[sinusoidal,learnable,none]=learnable
- **use_when**: PatchEmbed + 이것 = PatchTST (가장 검증된 조합). ≥10000행.
- **use_conditions**: `encoder == PatchEmbedding, n_rows >= 10000`
- **⚠ requires_encoder**: PatchEmbedding
- **prior_hint**: PatchTST 핵심. learnable PE > sinusoidal. n_layers=3 표준. lookback=336~512 최적.

#### ConvMix
- **출처**: TimesNet (Wu et al., 2023, ICLR), ModernTCN
- **primitive**: Convolution
- **shape**: (B,T,d_model) → (B,H,d_model)
- **capacity**: medium, min_data: 3000
- **config**: kernel_size[3~25]=7(홀수), n_layers[1~4]=2, dilation=[none,exponential]=exponential, dropout=0.1
- **use_when**: local temporal pattern (spike, change point), multi-scale 패턴
- **use_conditions**: `n_rows >= 3000`
- **avoid_when**: global dependency만 중요한 smooth 시계열
- **prior_hint**: Conv 기반 SOTA. 학습 안정, GPU 효율, 결정론적. 5-task 종합 1위(TSLib).

#### RecurrentMix
- **출처**: DeepAR (Salinas et al., 2020)
- **primitive**: Recurrence
- **shape**: (B,T,d_model) → (B,H,d_model)
- **capacity**: medium, min_data: 2000
- **config**: cell_type=[LSTM,GRU]=GRU, hidden_dim[32~256]=128, n_layers[1~3]=2, dropout=0.1
- **use_when**: 순차적 의존성 강한 데이터, streaming/online, 확률적 예측 (DeepAR)
- **avoid_when**: seq_len>200 (vanishing gradient), 대규모 (학습 느림), 2023 이후 벤치마크 열세
- **prior_hint**: GRU > LSTM (25% 적은 파라미터, 비슷한 성능). DeepAR=LSTM+Gaussian은 수요예측 산업 표준.

#### SSMMix (Phase 2)
- **출처**: S4 (Gu et al., 2022), Mamba (Gu & Dao, 2023)
- **primitive**: StateSpace
- **shape**: (B,T,d_model) → (B,H,d_model)
- **capacity**: medium, min_data: 5000
- **config**: d_state[8~64]=16, expand_factor[1~4]=2, selective=true
- **use_when**: 매우 긴 시퀀스 (>500), O(T·logT)로 Attention보다 효율적
- **avoid_when**: 짧은 시퀀스 (<200), 벤치마크에서 아직 일관적 우세 아님
- **status**: Phase 2 지원 예정.

---

### 3.4 ChannelMixer Slot (Optional)

#### None (기본값)
- **use_when**: **기본값.** 변수간 상관 약할 때 (max|r|<0.5), C≤7
- **prior_hint**: PatchTST 논문 — channel-independent가 dependent보다 나은 경우 많음. spurious correlation 방지.

#### FeatureMLPMix
- **출처**: TSMixer IC (Ekambaram et al., 2023)
- **primitive**: MLP
- **shape**: (B,H,d_model) → (B,H,d_model)
- **config**: hidden_dim[32~256]=64, dropout=0.1
- **use_when**: 변수간 상관 높을 때 (cross_corr>0.5 쌍 3개 이상). 외생변수 풍부.
- **use_conditions**: `high_cross_corr_pairs >= 3`
- **avoid_when**: 상관 약하면 noise만 추가. C≤7이면 오버헤드.
- **prior_hint**: 학술 벤치마크(ETT)에서 효과 미미. 실무(M5 소매)에서 큰 차이.

#### InvertedAttentionMix
- **출처**: iTransformer (Liu et al., 2024, ICLR)
- **primitive**: Attention (변수축)
- **shape**: (B,H,d_model) → (B,H,d_model)
- **config**: n_heads[2~8]=4, n_layers[1~3]=2, dropout=0.1
- **use_when**: C>10이고 상호작용 복잡. Traffic(862변수), ECL(321변수)에서 SOTA.
- **use_conditions**: `n_features > 10, n_rows >= 10000, high_cross_corr_pairs >= 5`
- **avoid_when**: C≤10, 데이터 부족
- **prior_hint**: TSLib 장기예측 1위 (2024.03). Traffic MSE=0.395 (DLinear 0.625 대비 37% 개선). **변수 수에 따라 가치 극변.**

---

### 3.5 Head Slot

Head의 output_dim은 forecasting setting에 따라 결정된다:
- **S, MS**: output_dim = 1
- **M + channel_independent**: output_dim = 1 (변수별 독립 적용, 실질적으로 1)
- **M + channel_dependent**: output_dim = C

#### LinearHead (기본값)
- **shape**: (B,H,d_model) → (B,H,output_dim)
- **config**: output_dim: int, default=1. "S/MS=1, M=C 또는 1(channel-independent)"
- **prior_hint**: head에서 복잡도 높이는 것은 거의 효과 없음. Linear이 표준.

#### FlattenLinearHead
- **shape**: (B,H,d_model) → (B,H,output_dim). params=H×d_model×output_dim.
- **use_when**: NHITS/NBEATS 스타일. 기본 LinearHead와 큰 차이 없음.

---

### 3.6 Constraint Slot (Optional, 복수 가능)

#### Positivity
- **use_conditions**: `can_be_negative == false AND target_min >= 0`
- **prior_hint**: Softplus > ReLU (gradient 안정). 에너지 도메인 필수.

#### Clamp
- **config**: min_val, max_val
- **use_conditions**: `known_physical_bounds == true`
- **prior_hint**: 도메인 지식을 가장 확실하게 반영.

#### Smoothness
- **config**: alpha[0.01~1.0]=0.1
- **use_when**: 예측이 급변하면 안 될 때 (온도, 부하)
- **avoid_when**: spike/change point가 중요할 때

---

### 3.7 Loss Slot

#### MAE
- **기본값.** 이상치에 강건. 대부분의 논문이 사용. **확신 없으면 MAE.**

#### MSE
- 큰 오차 강하게 페널티. 이상치 많으면 불리.

#### Huber
- **config**: delta[0.1~10.0]=1.0
- **use_conditions**: `target_skew > 1.5`
- **prior_hint**: MAE+MSE 결합. delta=1.0 범용적.

#### QuantileLoss
- **config**: quantile[0.01~0.99]=0.5
- **use_when**: 확률적 예측, prediction interval 필요

#### AsymmetricLoss
- **config**: over_weight[0.5~5.0]=1.0, under_weight[0.5~5.0]=2.0
- **use_conditions**: `extreme_ratio > 2.0`
- **use_when**: 과대/과소 비용 다를 때. 에너지: 과소 위험. 재고: 과대 위험.
- **prior_hint**: under_weight = extreme_ratio로 설정하면 직관적. **도메인→loss 가장 직접적 방법.**

---

## 4. Compatibility Rules (KG 호환성)

블록 간 관계를 KG edge로 정의. Builder가 조립 전에 검증.

### 4.1 Shape 호환성 (Hard Constraints)

```yaml
shape_rules:
  # PatchEmbedding 출력 shape이 다름 → 전용 Mixer만 가능
  - encoder: PatchEmbedding
    compatible_mixers: [PatchMLPMix, PatchAttentionMix]
    incompatible_mixers: [LinearMix, MLPMix, GatedMLPMix, AttentionMix, ConvMix, RecurrentMix, SSMMix]
    reason: "PatchEmbed 출력이 (B,n_patch,d_model). 비-Patch Mixer는 (B,T,d_model) 기대."

  # 비-Patch Encoder → 비-Patch Mixer만
  - encoder: [LinearProjection, FourierEmbedding]
    compatible_mixers: [LinearMix, MLPMix, GatedMLPMix, AttentionMix, ConvMix, RecurrentMix, SSMMix]
    incompatible_mixers: [PatchMLPMix, PatchAttentionMix]
    reason: "비-Patch Encoder 출력이 (B,T,d_model). Patch Mixer는 (B,n_patch,d_model) 기대."
```

### 4.2 Normalizer 중첩 규칙

```yaml
normalizer_rules:
  # DatasetNorm은 항상 적용 (train mean/std 기준). 벤치마크 비교의 기준 스케일.
  - rule: "Trainer.dataset_norm = always"
    reason: "논문 표준. DatasetNorm이 없으면 메트릭이 raw scale이 되어 벤치마크 비교 불가."

  # RevIN/RobustScaler는 DatasetNorm 위에서 instance-level 추가 정규화.
  - if: "normalizer == RevIN"
    then: "DatasetNorm(train mean/std) + RevIN(instance mean/std) 중첩 적용"
    reason: "PatchTST, iTransformer 등 논문 표준. 실증적으로 단독 DatasetNorm보다 우수."

  - if: "normalizer == RobustScaler"
    then: "DatasetNorm(train mean/std) + RobustScaler(instance IQR) 중첩 적용"

  - mutual_exclusive: [RevIN, RobustScaler]
    reason: "instance-level 정규화는 하나만. DatasetNorm과는 중첩 가능."

  # ⚠ RevIN reverse 시 target_idx 의존성
  - if: "normalizer == RevIN AND output_dim < n_features"
    then: "Builder가 target_idx를 ForecastModel에 전달. reverse 시 해당 채널의 통계만 사용."
    reason: "MS/S setting에서 target이 아닌 채널의 mean/std로 역변환하면 결과 오염."
```

### 4.3 Head 의존성 (Hard Constraints)

```yaml
head_rules:
  - head: ReconcileHead
    requires: "channel_mixer != null"
    reason: "reconciliation은 channel mixing 결과가 있어야 의미."
```

### 4.4 Capacity 제약 (Soft Constraints — 위반 시 경고)

```yaml
capacity_rules:
  - block: [AttentionMix, PatchAttentionMix, InvertedAttentionMix]
    condition: "n_rows < 10000"
    action: warn
    message: "Attention 블록은 10000행 이상 권장. 과적합 위험."

  - block: [PatchMLPMix, GatedMLPMix, ConvMix]
    condition: "n_rows < 5000"
    action: warn
    message: "Medium capacity 블록은 5000행 이상 권장."

  - block: [SSMMix]
    condition: "seq_len < 200"
    action: warn
    message: "SSM은 긴 시퀀스(≥200)에서 효율적. 짧으면 이점 없음."
```

### 4.5 추천 규칙 (Soft — KG Matcher가 후보 생성에 사용)

```yaml
recommendation_rules:
  # Encoder
  - condition: "max_acf_at_known_periods > 0.7"
    recommends: FourierEmbedding
    param_hint: "n_harmonics = count(ACF > 0.7)"

  - condition: "seq_len >= 96 AND multi_scale_suspected"
    recommends: PatchEmbedding
    param_hint: "patch_len = dominant_period // 4"

  # TemporalMixer
  - condition: "always"
    recommends: LinearMix
    priority: "baseline_first"

  - condition: "n_rows >= 10000 AND complex_temporal_pattern"
    recommends: PatchAttentionMix
    requires: PatchEmbedding

  - condition: "has_local_patterns (spike, change_point)"
    recommends: ConvMix

  # ChannelMixer
  - condition: "high_cross_corr_pairs >= 5 AND n_features > 10"
    recommends: InvertedAttentionMix

  - condition: "high_cross_corr_pairs >= 3 AND n_features <= 10"
    recommends: FeatureMLPMix

  # Loss
  - condition: "extreme_ratio > 2.0"
    recommends: AsymmetricLoss
    param_hint: "under_weight = min(extreme_ratio, 5.0)"

  - condition: "target_skew > 1.5"
    recommends: Huber

  # Constraint
  - condition: "can_be_negative == false AND target_min >= 0"
    recommends: Positivity
```

### 4.6 Forecasting Setting 규칙

```yaml
forecasting_setting_compatibility:
  S:
    head_output_dim: 1
    channel_mixer: null  # 강제 — 변수 1개이므로
    channel_independent: true  # 유일한 옵션
    encoder_input_dim: 1
    
  MS:
    head_output_dim: 1
    channel_mixer: "선택적 — 외생변수 활용 시"
    channel_independent: 
      recommended: false  # 외생변수 정보를 활용하려면 mixing 필요
      note: "true로 하면 타겟 변수만 독립 처리 → 외생변수 무시됨"
    encoder_input_dim: C  # 모든 변수 입력
    loss_target: "타겟 변수 1개에 대해서만 계산"
    
  M:
    head_output_dim: C
    channel_mixer: "추천 — 변수 간 상호작용이 핵심"
    channel_independent:
      true: "PatchTST 방식 — 각 변수 독립 예측, weight 공유"
      false: "iTransformer 방식 — 변수 간 attention/mixing"
      selection: "KG recommendation + LLM 판단"
    encoder_input_dim: C
    loss_target: "C개 변수 전체에 대해 평균 loss"
    note: "논문 벤치마크 기본 설정. 비교 시 이 설정 사용."
```

---

## 5. Verified Recipes

검증된 모델을 블록으로 분해. 각 레시피는 논문/벤치마크에서 검증된 조합.
**벤치마크 수치는 M(Multivariate→Multi) 설정 기준.** MS 설정에서는 수치가 다름.

### DLinear (항상 첫 번째 시도)
- **source**: Zeng et al., 2023 (AAAI)
- **setting**: M (channel-independent, 각 변수 독립 예측), S, MS 모두 지원
- **blocks**: RevIN → LinearProjection → LinearMix → null → LinearHead(output_dim=setting에 따라)
- **HP**: lr=1e-3, epochs=50, patience=10, wd=1e-4
- **benchmark**: ETTh1=0.386, ETTh2=0.333, Weather=0.196, ECL=0.197
- **best_for**: baseline, smooth, small_data, interpretable
- **notes**: 항상 첫 번째로 시도. 이것보다 나은 모델만 채택.

### PatchTST
- **source**: Nie et al., 2023 (ICLR)
- **blocks**: RevIN → PatchEmbedding(16,8) → PatchAttentionMix(4heads, 3layers) → null → LinearHead
- **HP**: lr=1e-4, epochs=100, patience=20, wd=1e-5
- **benchmark**: ETTh1=0.370, ETTh2=0.302, Weather=0.149, ECL=0.181
- **best_for**: large_data, long_sequence, complex_patterns
- **notes**: lookback=336~512에서 더 좋음. self-supervised pre-training 가능.

### TSMixer_CI
- **source**: Chen et al., 2023 (TMLR)
- **blocks**: RevIN → PatchEmbedding(16,8) → PatchMLPMix → null → LinearHead
- **HP**: lr=1e-4, epochs=100, patience=20
- **benchmark**: ETTh1≈0.375, Weather≈0.164
- **best_for**: moderate_data, efficient_alternative_to_PatchTST

### iTransformer
- **source**: Liu et al., 2024 (ICLR)
- **blocks**: RevIN → LinearProjection → LinearMix → InvertedAttentionMix(4heads, 2layers) → LinearHead
- **HP**: lr=1e-4, epochs=100, patience=20
- **benchmark**: ETTh2=0.297, ECL=0.148, Traffic=0.395
- **best_for**: many_features, high_cross_correlation
- **notes**: 변수 많을 때 최강. 변수 적으면 DLinear과 비슷.

### TimesNet_style
- **source**: Wu et al., 2023 (ICLR)
- **blocks**: RevIN → LinearProjection → ConvMix(kernel=7, 2layers, dilated) → null → LinearHead
- **HP**: lr=5e-4, epochs=100, patience=15
- **benchmark**: ETTh1=0.384, Weather=0.172
- **best_for**: multi_task, local_patterns, stable_training

### 커스텀 레시피 등록 형식

```yaml
custom_recipe:
  name: "DLinear_MLPmix_v1"
  origin: "DLinear의 LinearMix → MLPMix 교체"
  discovered_on: {dataset: "ETTh1", date: "2025-XX-XX"}
  blocks: RevIN → LinearProjection → MLPMix → null → LinearHead
  HP: {lr: 5e-4, epochs: 100}
  performance_delta: {ETTh1: "-0.012 MSE vs DLinear"}  # 개선이면 음수
  verified_on: {ETTh1: 0.374}  # 1개 데이터셋만 → verified=false
  verified: false  # 3개 이상 데이터셋에서 검증되면 true
  when_useful: "LinearMix가 비선형 패턴을 못 잡을 때"
```

---

## 6. Preprocessing (Trainer Config, 블록 아님)

전처리는 블록이 아닌 **Trainer의 config 옵션**으로 처리. shape 관리 복잡도를 피하기 위함.

```yaml
preprocessing_options:
  log_transform:
    description: "log1p 변환. 우편향 분포 안정화."
    when: "target_min > 0 AND target_skew > 2.0"
    reverse: "expm1로 역변환 (Trainer가 메트릭 계산 시 자동)"
    
  differencing:
    description: "1차 차분. non-stationarity 제거."
    when: "ADF p > 0.1 AND RevIN 미사용"
    reverse: "cumsum으로 역변환"
    note: "RevIN이 이미 non-stationarity를 처리하므로, RevIN 사용 시 differencing 불필요."
    
  seasonal_decompose:
    description: "STL 분해 → trend + seasonal + residual 분리 학습."
    when: "매우 강한 주기성 (ACF > 0.9) AND Autoformer/FEDformer 스타일"
    status: "Phase 2"
```

---

## 7. Feature Engineering Templates

Engineer가 사용하는 rule-based 피쳐 생성 템플릿. **모든 템플릿에 leakage 방지 규칙 내장.**

### ⚠ Leakage의 원리: 왜 pred_len이 기준인가

이 로직을 이해하지 못하면 모든 feature engineering이 오염된다. **반드시 이해하고 적용해야 한다.**

```
[시나리오] 시점 t에서 pred_len=96을 예측

  시간축:  ... t-2  t-1  [t]  t+1  t+2  ...  t+96
                         ↑예측시점  ↑─────────────↑ 예측 대상
  
  예측 시점 t에서:
    알 수 있는 것: y[t], y[t-1], y[t-2], ... (과거 전부)
    모르는 것:     y[t+1], y[t+2], ..., y[t+96] (이걸 예측해야 하니까)

[lag feature의 leakage]

  target_lag_k = target.shift(k)
  
  즉, 시점 t의 lag_k 값 = y[t-k]
  
  학습 데이터 한 쌍: (features[t], targets[t+1 : t+96])
    features[t]에 lag_1 = y[t-1] 포함 → 안전? 
    → 아니! 학습 시 모델이 (features[t], target[t+1:t+96])을 보는데,
      lag_1 = y[t-1]이고 target 시작 = y[t+1]. 
      하지만 sliding window로 다음 샘플을 보면:
      (features[t+1], target[t+2:t+97])에서 lag_1 = y[t]인데,
      이전 샘플의 target[t+1]의 바로 직전값 y[t]를 feature로 씀.
      
  → 일반화하면: lag_k에서 k < pred_len이면,
    feature[t]에 y[t-k]가 있고, target은 y[t+1:t+H] 범위인데,
    다른 샘플에서 y[t-k]가 target 범위 안에 들어갈 수 있음 → leakage.
    
  → k >= pred_len이면:
    feature[t]의 lag_k = y[t - pred_len] (또는 그 이전)
    target은 y[t+1 : t+pred_len]
    y[t-pred_len]은 target 범위의 어떤 샘플에서도 target이 아님 → 안전.

  ✅ 규칙: lag_k는 k >= pred_len인 것만 사용

[rolling feature의 leakage]

  rolling_mean_w = target.rolling(window=w).mean()
  
  시점 t의 rolling_mean_w = mean(y[t-w+1], ..., y[t])  ← y[t] 포함!
  
  이걸 그대로 feature로 쓰면:
    feature[t]에 y[t]의 정보가 포함 → 모델이 "현재 target의 이동평균"을 보고 예측
    → cheating (target 정보가 feature에 유출)
    
  shift(pred_len) 적용하면:
    feature[t] = rolling_mean_w[t - pred_len] = mean(y[t-pred_len-w+1], ..., y[t-pred_len])
    → y[t-pred_len] 이전의 정보만 사용 → 안전.

  ✅ 규칙: rolling feature는 반드시 .shift(pred_len) 적용

[known vs unknown covariate]

  known covariate: 예측 시점에 미래값을 알 수 있는 변수
    예: calendar (hour, dow — 미래 날짜를 알고 있음)
    예: 기상 예보 (weather forecast — 미래 예보값이 있음)
    → shift 불필요. 미래값을 쓸 수 있으니까.

  unknown covariate: 예측 시점에 미래값을 모르는 변수
    예: target 자체, 실측 기상, 실시간 수요
    → 반드시 shift(pred_len) 적용. 과거값만 사용 가능.

  ⚠ 판단 기준: "예측 시점 t에서 이 변수의 t+1~t+H 값을 알 수 있는가?"
    YES → known → shift 불필요
    NO  → unknown → shift(pred_len) 필수
    모르겠으면 → unknown으로 취급 (안전 우선)

[pred_len별 사용 가능한 lag 범위]

  pred_len=24:  lag_24, lag_48, lag_168, ... (24 이상)
  pred_len=96:  lag_96, lag_168, lag_336, ... (96 이상)
  pred_len=336: lag_336, lag_672, ...         (336 이상)
  
  → pred_len이 길수록 사용 가능한 lag이 줄어듦 → feature가 빈약해짐
  → 이것이 장기 예측이 어려운 이유 중 하나
  → 이때 calendar feature(known)의 가치가 상대적으로 커짐
```

```yaml
feature_templates:

  calendar_features:
    type: "known_covariate"  # 미래값 사용 가능 → shift 불필요
    when: "datetime 컬럼 존재"
    features:
      - "hour_sin = sin(2π·hour/24), hour_cos = cos(2π·hour/24)"
      - "dow_sin = sin(2π·dayofweek/7), dow_cos = cos(2π·dayofweek/7)"
      - "month_sin = sin(2π·month/12), month_cos = cos(2π·month/12)"
      - "is_weekend = (dayofweek >= 5)"
    shift_required: false  # known → 미래값 OK
    leakage_safe: true
    note: "pred_len이 길수록 이 feature의 상대적 가치가 커짐 (lag feature 사용 불가 시)"

  lag_features:
    type: "unknown_covariate"  # 미래값 불가 → shift 필수
    when: "Scout에서 유의미한 ACF lag가 있을 때"
    selection_rule: |
      1. Scout의 ACF 분석에서 ACF > 0.3인 lag 후보 추출
      2. 후보 중 k >= pred_len인 것만 남김 (LEAK-003)
      3. pred_len 자체도 항상 lag 후보에 포함
      예시 (pred_len=96):
        ACF 유의미한 lag: [1, 24, 48, 96, 168, 336]
        pred_len 이상만 필터: [96, 168, 336]
        생성: target_lag_96, target_lag_168, target_lag_336
    implementation: "target_lag_k = target.shift(k)  # k >= pred_len"
    shift_required: true
    leakage_safe: "k >= pred_len일 때만"
    common_patterns:
      hourly_data: "pred_len=96 → lag_96(4일전), lag_168(1주전), lag_336(2주전)"
      daily_data: "pred_len=7 → lag_7(1주전), lag_14(2주전), lag_30(1월전)"

  rolling_features:
    type: "unknown_covariate"
    when: "Scout에서 계절성 감지 또는 변동성 분석 필요"
    selection_rule: |
      1. window 크기: seasonality에서 ACF > 0.5인 주기에 해당하는 window
      2. 모든 rolling 결과에 .shift(pred_len) 적용 (LEAK-003)
      예시 (pred_len=96, 24h 주기 강함):
        rolling_mean_24 = target.rolling(24).mean().shift(96)
        rolling_std_24 = target.rolling(24).std().shift(96)
        → 시점 t의 값 = 96+24=120 시간 전부터 96 시간 전까지의 통계
    features:
      - "rolling_mean_{w} = target.rolling(w).mean().shift(pred_len)"
      - "rolling_std_{w} = target.rolling(w).std().shift(pred_len)"
    shift_required: true  # ⚠ 반드시
    leakage_safe: "shift(pred_len) 적용 시에만"
    note: |
      rolling + shift로 인해 앞쪽 (window + pred_len)개 행이 NaN → dropna 필요.
      pred_len이 길면 유효 데이터가 많이 줄어듦에 주의.

  exogenous_features:
    type: "판단 필요 — known인지 unknown인지 구분 필수"
    when: "외생변수 존재"
    classification_rule: |
      각 외생변수에 대해:
      Q: "예측 시점 t에서 이 변수의 t+1 ~ t+pred_len 값을 알 수 있는가?"
        YES → known_covariate: shift 불필요. 미래값 사용 가능.
          예: 기상 예보, 공휴일 일정, 계획된 이벤트
        NO  → unknown_covariate: shift(pred_len) 필수.
          예: 실측 기상, 실시간 수요, 주가
        불확실 → unknown으로 취급 (안전 우선)
    selection_rule: "Scout의 exog_correlations에서 |r| >= 0.3인 변수 우선"
    action: "보고만 하고 제거하지 않음 (모델이 자동 처리). 상관 분석 결과를 Architect에 전달."
```

### Leakage 사후 검증

```yaml
leakage_verification:
  method: "각 feature와 future_target(pred_len 후) 간 상관 계산"
  thresholds:
    auto_remove: "|r| > 0.98 — 높은 leakage 확신 → 자동 제거 + 경고"
    warn_only: "|r| > 0.95 — leakage 가능성 → 경고만"
    safe: "|r| <= 0.95 — 정상"
  note: "lag feature의 높은 상관(예: OT_lag_168 r=0.86)은 정상 — leakage 아님"
```

---

## 8. Mandatory Rules

### 8.1 Data Leakage Prevention

**상세 원리는 Section 7의 "Leakage의 원리" 참조.**

| ID | Rule | Enforcement |
|----|------|------------|
| LEAK-001 | Temporal split only. random split 절대 금지. | Trainer 자동 적용 |
| LEAK-002 | 정규화 통계는 train split에서만 계산. | Trainer가 train_end 이전 데이터만 사용 |
| LEAK-003 | Target-derived feature: lag_k는 k≥pred_len만, rolling은 .shift(pred_len) 필수. known covariate만 미래값 허용. | Engineer 자동 + 사후 검증 |
| LEAK-004 | Feature-future correlation check. \|r\|>0.98 제거, \|r\|>0.95 경고. | Engineer 사후 검증 |
| LEAK-005 | Train-val 사이 pred_len 크기 gap. val 첫 sample의 target이 train 마지막 데이터와 겹치지 않도록. | Trainer temporal_split |
| LEAK-006 | Forecasting setting이 MS일 때, 타겟 변수의 미래값이 외생변수 feature에 포함되지 않도록. 외생변수도 unknown이면 shift(pred_len). | Engineer 자동 |

### 8.2 Evaluation Integrity

| ID | Rule | Enforcement |
|----|------|------------|
| EVAL-001 | 모든 모델은 naive baseline(last-value repeat)을 이겨야 함. 미달 시 ERROR. | Critic 자동 비교 |
| EVAL-002 | 알려진 데이터셋은 표준 split 사용 (benchmark_mode). | Trainer STANDARD_SPLITS |
| EVAL-003 | Normal/Extreme 분리 평가. 극단값: train 기준 상/하위 5%. | Trainer 계산, Critic 분석 |
| EVAL-004 | 원본 + 정규화 스케일 메트릭 동시 리포트. | Trainer 양쪽 계산 |
| EVAL-005 | CV로 선택 → refit(train+val) → test 1회. 정규화 통계는 train-only 유지. | Trainer 자동 |

### 8.3 Training Stability

| ID | Rule | Enforcement |
|----|------|------------|
| TRAIN-001 | Backbone capacity별 HP preset 차등 적용. | Trainer 자동 |
| TRAIN-002 | best_epoch=0이면 학습 미시작. lr 조정 권장. | Critic 감지 |
| TRAIN-003 | val 대비 test 성능 2배 이상 나쁘면 분포 차이 경고. | Critic 비교 |

### HP Preset 기준

| Capacity | lr | epochs | patience | weight_decay |
|----------|----|--------|----------|-------------|
| minimal | 1e-3 | 50 | 10 | 1e-4 |
| low | 5e-4 | 100 | 15 | 1e-5 |
| medium | 1e-4 | 100 | 15 | 1e-5 |
| high | 1e-4 | 200 | 20 | 1e-5 |

---

## 9. Orchestration Pipeline

```
사용자 CSV + target + horizon
        ↓
   ┌─────────────┐
   │  Scout       │  (rule-based) 데이터 프로파일 생성
   │  통계, ACF, ADF, regime, 상관, 분포, outlier_ratio
   └──────┬──────┘
          ↓
   ┌─────────────┐
   │  KG Matcher  │  (rule-based) 프로파일 → KG 쿼리 → 유효 조합 필터링
   │  1. Level 0: topology 결정 (encoder-only)
   │  2. Level 1: 필요한 slot 결정
   │  3. use_conditions 자동 체크 → 블록 pre-filtering
   │  4. compatibility_rules 검증 → 유효 조합만
   │  5. 후보 레시피 3~5개 생성 + 각각의 근거
   └──────┬──────┘
          ↓
   ┌─────────────┐
   │  Architect   │  (LLM) Decision Protocol — KG가 준 후보 안에서만 판단
   │  Step 1: 레시피 선택 (후보 중 매칭)
   │  Step 2~3: HP 조정 (range 내 숫자)
   │  Step 4: Loss 선택 (추천 + 확인)
   │  Step 5: Constraint (추천 + 확인)
   │  Step 6: Training strategy (fold 수)
   │  → 모델링 리포트 (Q&A 로그) 자동 생성
   └──────┬──────┘
          ↓
   ┌─────────────┐
   │  Engineer    │  (rule-based) 피쳐 엔지니어링 + leakage 검증
   └──────┬──────┘
          ↓
   ┌─────────────┐
   │  Builder     │  (deterministic) config → 모델 조립
   │  1. KG 호환성 최종 검증 (build-time assert)
   │  2. 블록 인스턴스 생성
   │  3. shape assertion (dummy forward)
   │  4. RevIN이면 Trainer에 skip_dataset_norm 전달
   └──────┬──────┘
          ↓
   ┌─────────────┐
   │  Trainer     │  (deterministic) 학습 + 평가
   │  - preprocessing (log, diff) 적용
   │  - 정규화 (RevIN이면 skip, 아니면 DatasetNorm)
   │  - temporal CV + refit
   │  - capacity별 HP preset
   │  - naive baseline 계산
   │  - normal/extreme 분리 메트릭
   └──────┬──────┘
          ↓
   ┌─────────────┐
   │  Critic      │  (rule-based) 분석 + 피드백 분기
   │  DONE          → 최종 결과
   │  RETRY_HP      → Architect Step 2~6 재실행
   │  RETRY_RECIPE  → Architect Step 1 재실행 (다른 레시피)
   │  RETRY_BLOCK   → KG에 교체 가능 블록 쿼리 → Architect가 선택
   └──────┬──────┘
          ↓ (RETRY_BLOCK인 경우)
   KG: "현재 TemporalMixer=LinearMix를 대체 가능한 블록?"
     → 같은 slot, 같은 input shape, 호환 → [MLPMix, ConvMix]
     → Architect: "MLPMix vs ConvMix?" (유효 후보 내에서만)
          ↓
   Builder → Trainer → Critic (반복, 최대 3회)
          ↓
   성공 시 커스텀 레시피 등록 (verified=false)
```

---

## 10. Critic 피드백 분기 로직

```yaml
critic_decision_tree:
  - check: "model_metric > naive_metric"
    verdict: ERROR
    message: "naive보다 나쁨. 데이터/파이프라인 점검 필요."

  - check: "best_epoch == 0"
    verdict: RETRY_HP
    message: "학습 미시작. lr 낮추기 또는 warmup 추가."

  - check: "val_test_ratio > 2.0"
    verdict: RETRY_HP
    message: "val-test 갭 과대. 더 단순한 모델 또는 RevIN 권장."

  - check: "extreme_ratio > 3.0 AND iteration < max_iterations"
    verdict: RETRY_BLOCK
    message: "극단 구간 성능 불량. TemporalMixer capacity 상향 시도."

  - check: "improvement < 2% vs previous iteration"
    verdict: DONE
    message: "성능 ceiling. 현재 결과 확정."

  - check: "iteration == 1"
    verdict: RETRY_RECIPE
    message: "baseline 확보. 다른 레시피 시도."

  - default:
    verdict: RETRY_HP
    message: "HP 미세 조정 시도."
```

---

## 11. Implementation Priorities

| Phase | 핵심 작업 | 결과물 |
|-------|----------|--------|
| 1 | BaseBlock 인터페이스, 블록 구현 (Catalog 기준), KG 스키마 (YAML), Builder (호환성 검증 + 조립) | 블록이 독립 테스트 통과 |
| 2 | Scout, KG Matcher, Architect v2 (Decision Protocol + KG bounded), Engineer, Trainer, Critic | 풀 파이프라인 1회 실행 |
| 3 | 멀티 데이터셋 벤치마크, 레시피 verified_on 채우기, 블록 교체 경험 축적 | 검증된 결과표 |
| 4 | SSM/encoder-decoder topology 확장, NeuralForecast 블록 추출, KG 임베딩 (ZooCast 방식) | 확장된 카탈로그 |

---

## 12. File Structure

```
cballm/
├── ontology/
│   ├── block_catalog.yaml     # 이 문서의 Section 3 (블록 정의 + 힌트)
│   ├── compatibility.yaml     # Section 4 (호환성 규칙)
│   ├── recommendations.yaml   # Section 4.5 (추천 규칙)
│   └── kg_engine.py           # KG 쿼리 엔진
│
├── blocks/
│   ├── base.py                # BaseBlock 인터페이스
│   ├── normalizer.py          # RevIN, RobustScaler
│   ├── encoder.py             # LinearProjection, PatchEmbedding, FourierEmbedding
│   ├── temporal_mixer.py      # LinearMix, MLPMix, GatedMLPMix, PatchMLPMix,
│   │                          # AttentionMix, PatchAttentionMix, ConvMix, RecurrentMix
│   ├── channel_mixer.py       # FeatureMLPMix, InvertedAttentionMix
│   ├── head.py                # LinearHead, FlattenLinearHead
│   ├── constraint.py          # Positivity, Clamp, Smoothness
│   ├── loss.py                # MAE, MSE, Huber, Quantile, Asymmetric
│   └── builder.py             # KG-validated config → ForecastModel 조립
│
├── recipes/
│   ├── verified/              # DLinear.yaml, PatchTST.yaml, TSMixer_CI.yaml, ...
│   ├── custom/                # 파이프라인이 발견한 커스텀 레시피
│   └── registry.py            # 레시피 로드, 검색, 등록, 성능 비교
│
├── rules/
│   ├── leakage.yaml           # LEAK-001~005
│   ├── evaluation.yaml        # EVAL-001~005
│   └── training.yaml          # TRAIN-001~003
│
├── features/
│   ├── templates.yaml         # 피쳐 생성 템플릿
│   └── engineer.py            # rule-based 피쳐 엔지니어링 + leakage 검증
│
├── workers/
│   ├── scout.py               # rule-based 데이터 프로파일링
│   ├── kg_matcher.py          # 프로파일 → KG 쿼리 → 후보 레시피/조합
│   ├── architect.py           # LLM Decision Protocol (KG-bounded)
│   ├── trainer.py             # template-based 학습 엔진
│   └── critic.py              # rule-based 평가 + 피드백 분기
│
├── brain.py                   # 오케스트레이터
└── engine.py                  # LLM 엔진 (Qwopus + Coder swap)
```
