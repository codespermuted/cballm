"""CBALLM Blocks — KG 온톨로지 기반 슬롯 조립 시스템.

파이프라인 슬롯 (v2):
  Input → [Normalizer?] → [Encoder] → [TemporalMixer] → [ChannelMixer?] → [Head] → [Constraint?] → Output
                                                                                          ↑
                                                                                      [Loss]

텐서 규약:
  Normalizer:    (B, T, C) → (B, T, C), has_reverse
  Encoder:       (B, T, C) → (B, T, d_model) or (B, n_patch, d_model)
  TemporalMixer: (B, T|n_patch, d_model) → (B, H, d_model)
  ChannelMixer:  (B, H, d_model) → (B, H, d_model)
  Head:          (B, H, d_model) → (B, H, output_dim)
  Constraint:    (B, H, output_dim) → (B, H, output_dim)
  Loss:          (pred, target) → scalar
"""
