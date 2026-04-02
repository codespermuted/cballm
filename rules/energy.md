# 에너지/전력 도메인 룰 (일반화)

## 피쳐 엔지니어링
- 전력/에너지 데이터는 항상 요일(day_of_week) + 공휴일(is_holiday) 피쳐 포함
- 가격 예측 시 연료비/원자재 가격을 포함하면 효과적
- 시간대별 패턴이 강하므로 hour_sin, hour_cos 포함
- 수요(demand/load) 데이터가 있으면 반드시 포함 — 가격과 강한 상관
- 계절성: 24시간(일간), 168시간(주간) 패턴 존재 → lag-24, lag-168 필수

## 극단 이벤트 처리
- 가격 급락/급등 구간은 별도 분류 모델 또는 asymmetric loss로 대응
- 극단 이벤트는 전체의 1~3% → loss weighting 또는 regime gate

## 모델 선택
- 첫 시도는 LightGBM + PatchTST 조합 추천
- 15분 해상도 데이터는 N-HiTS가 효과적
- 1시간 해상도는 TFT 또는 iTransformer
- 데이터 1000행 미만이면 Statistical 모델 우선

## 평가 기준
- MAE가 주 메트릭
- MAPE는 target이 0 근처일 때 불안정하므로 보조만
- 극단 구간 MAE를 별도로 계산하여 보고
