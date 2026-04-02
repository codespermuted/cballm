# 에너지/전력 시장 도메인 룰

## 피쳐 엔지니어링
- 전력 시장 데이터는 항상 요일(day_of_week) + 공휴일(is_holiday) 피쳐 포함
- SMP 예측 시 연료비(LNG, 석탄, 유가)를 무조건 포함
- 시간대별 패턴이 강하므로 hour, sin_hour, cos_hour 포함
- 수요(demand) 데이터가 있으면 반드시 포함 — SMP와 강한 상관
- 계절성: 24시간(일간), 168시간(주간) 패턴 존재 → lag-24, lag-168 필수

## Dip/Spike 처리
- Dip 구간: SMP < 40원 AND 시간 ∈ [10~16시] (태양광 과잉 구간)
- Spike: 제주-육지 spread < -10 구간
- Dip/Spike는 별도 분류 모델 또는 asymmetric loss로 대응

## 모델 선택
- 첫 시도는 LightGBM + PatchTST 조합 추천
- 15분 해상도 데이터는 N-HiTS가 효과적
- 1시간 해상도 DA 시장은 TFT 또는 iTransformer
- 데이터 1000행 미만이면 Statistical 모델 우선

## 평가 기준
- MAE가 주 메트릭
- MAPE는 SMP가 0 근처일 때 불안정하므로 보조만
- Dip 구간 MAE를 별도로 계산하여 보고
