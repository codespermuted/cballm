"""CBALLM Blocks — 슬롯 기반 시계열 모델 조립 시스템.

파이프라인 슬롯:
  Input → [Encoder] → [Decomposer?] → [Backbone × RegimeGate?] → [Constraint?] → Output
                                                                        ↑
                                                                    [Loss]

텐서 규약:
  Encoder:     (B, T, raw_features) → (B, T, d_model)
  Decomposer:  (B, T, d_model) → List[(B, T, d_model)]
  Backbone:    (B, T, d_model) → (B, H, 1)  where H = prediction_length
  RegimeGate:  Backbone wrapper, 동일 시그니처
  Constraint:  (B, H, 1) → (B, H, 1)  output post-processing
  Loss:        (pred, target) → scalar
"""
