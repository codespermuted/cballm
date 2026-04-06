"""TemporalMixer 블록 — 시간축 mixing (v2).

backbone 개념을 대체. 모든 temporal mixer:
  (B, T|n_patch, d_model) → (B, H, d_model)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .base import BaseTemporalMixer


class LinearMix(BaseTemporalMixer):
    """DLinear 스타일 — 각 d_model 채널을 독립적으로 T→H 매핑."""

    def __init__(self, seq_len: int, pred_len: int, d_model: int,
                 channel_independent: bool = True):
        super().__init__()
        self.channel_independent = channel_independent
        # 각 채널 독립: T → H
        self.temporal_proj = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        x = x.permute(0, 2, 1)           # (B, d_model, T)
        x = self.temporal_proj(x)         # (B, d_model, H)
        return x.permute(0, 2, 1)         # (B, H, d_model)


class MLPMix(BaseTemporalMixer):
    """TSMixer TMix-Only — MLP 기반 temporal mixing."""

    def __init__(self, seq_len: int, pred_len: int, d_model: int,
                 hidden_dim: int = 256, dropout: float = 0.1,
                 activation: str = "GELU", n_layers: int = 1):
        super().__init__()
        act = nn.GELU() if activation == "GELU" else nn.ReLU()

        layers: list[nn.Module] = []
        in_dim = seq_len
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                act,
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, pred_len))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)           # (B, d_model, T)
        x = self.mlp(x)                  # (B, d_model, H)
        return x.permute(0, 2, 1)         # (B, H, d_model)


class GatedMLPMix(BaseTemporalMixer):
    """TSMixer IC — Gated MLP. noisy feature가 많을 때 gating으로 억제."""

    def __init__(self, seq_len: int, pred_len: int, d_model: int,
                 hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(seq_len, hidden_dim)
        self.gate_fc = nn.Linear(seq_len, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, pred_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)           # (B, d_model, T)
        h = nn.functional.gelu(self.fc1(x))
        gate = torch.sigmoid(self.gate_fc(x))
        h = self.dropout(h * gate)
        h = self.fc2(h)                  # (B, d_model, H)
        return h.permute(0, 2, 1)         # (B, H, d_model)


class PatchMLPMix(BaseTemporalMixer):
    """TSMixer CI — Patch 기반 MLP. PatchEmbedding과 함께 사용."""

    def __init__(self, n_patches: int, pred_len: int, d_model: int,
                 hidden_dim: int = 256, dropout: float = 0.1, **kwargs):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_patches, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_patch, d_model)
        x = x.permute(0, 2, 1)           # (B, d_model, n_patch)
        x = self.mlp(x)                  # (B, d_model, H)
        return x.permute(0, 2, 1)         # (B, H, d_model)


class AttentionMix(BaseTemporalMixer):
    """Vanilla Transformer Encoder 기반 temporal mixing."""

    def __init__(self, seq_len: int, pred_len: int, d_model: int,
                 n_heads: int = 4, n_layers: int = 2,
                 d_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.temporal_proj = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)              # (B, T, d_model)
        x = x.permute(0, 2, 1)           # (B, d_model, T)
        x = self.temporal_proj(x)         # (B, d_model, H)
        return x.permute(0, 2, 1)         # (B, H, d_model)


class PatchAttentionMix(BaseTemporalMixer):
    """PatchTST — Patch + Attention 기반 temporal mixing."""

    def __init__(self, n_patches: int, pred_len: int, d_model: int,
                 n_heads: int = 4, n_layers: int = 3,
                 d_ff: int = 256, dropout: float = 0.1,
                 positional_encoding: str = "learnable", **kwargs):
        super().__init__()

        # Positional encoding
        if positional_encoding == "learnable":
            self.pos_embed = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)
        elif positional_encoding == "sinusoidal":
            pe = torch.zeros(n_patches, d_model)
            position = torch.arange(0, n_patches, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pos_embed", pe.unsqueeze(0))
        else:
            self.register_buffer("pos_embed", torch.zeros(1, 1, 1))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.temporal_proj = nn.Linear(n_patches, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_patch, d_model)
        x = x + self.pos_embed[:, :x.shape[1], :]
        x = self.encoder(x)              # (B, n_patch, d_model)
        x = x.permute(0, 2, 1)           # (B, d_model, n_patch)
        x = self.temporal_proj(x)         # (B, d_model, H)
        return x.permute(0, 2, 1)         # (B, H, d_model)


class ConvMix(BaseTemporalMixer):
    """TimesNet/ModernTCN 스타일 Conv 기반 temporal mixing."""

    def __init__(self, seq_len: int, pred_len: int, d_model: int,
                 kernel_size: int = 7, n_layers: int = 2,
                 dilation: str = "exponential", dropout: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(n_layers):
            d = 2 ** i if dilation == "exponential" else 1
            pad = (kernel_size - 1) * d // 2
            layers.extend([
                nn.Conv1d(d_model, d_model, kernel_size, padding=pad, dilation=d),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        self.conv = nn.Sequential(*layers)
        self.temporal_proj = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        h = x.permute(0, 2, 1)           # (B, d_model, T)
        h = self.conv(h)                  # (B, d_model, T)
        h = self.temporal_proj(h)         # (B, d_model, H)
        return h.permute(0, 2, 1)         # (B, H, d_model)


class RecurrentMix(BaseTemporalMixer):
    """DeepAR 스타일 RNN 기반 temporal mixing."""

    def __init__(self, seq_len: int, pred_len: int, d_model: int,
                 cell_type: str = "GRU", hidden_dim: int = 128,
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.pred_len = pred_len
        rnn_cls = nn.GRU if cell_type == "GRU" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=d_model, hidden_size=hidden_dim,
            num_layers=n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_dim, d_model)
        self.temporal_proj = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        out, _ = self.rnn(x)             # (B, T, hidden_dim)
        out = self.fc(out)               # (B, T, d_model)
        out = out.permute(0, 2, 1)       # (B, d_model, T)
        out = self.temporal_proj(out)    # (B, d_model, H)
        return out.permute(0, 2, 1)      # (B, H, d_model)


class FrequencyMix(BaseTemporalMixer):
    """FEDformer 단순화 — frequency domain에서 top-k 성분만 선택+가중 mixing.

    FFT → top-k frequency 선택 → learnable weight 적용 → iFFT → Linear(T→H).
    주기성이 시간에 따라 변하는 데이터에 유효.

    k 가이드: small data(n_rows<5000) → k=3, large → k=8
    """

    def __init__(self, seq_len: int, pred_len: int, d_model: int,
                 top_k: int = 5, **kwargs):
        super().__init__()
        self.top_k = top_k
        self.d_model = d_model
        self.n_freq = seq_len // 2 + 1

        # frequency domain에서의 learnable complex weight
        # 전체 frequency bin에 대한 weight (top-k만 활성화됨)
        self.freq_weight = nn.Parameter(torch.ones(self.n_freq, d_model))

        # time domain projection: T → H
        self.temporal_proj = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        B, T, D = x.shape

        # FFT along time axis
        x_freq = torch.fft.rfft(x, dim=1)  # (B, n_freq, d_model) complex

        # top-k frequency selection by magnitude
        magnitudes = x_freq.abs().mean(dim=-1)  # (B, n_freq)
        # DC 성분 보존을 위해 masking에서 제외
        mag_no_dc = magnitudes.clone()
        mag_no_dc[:, 0] = 0
        _, topk_idx = mag_no_dc.topk(self.top_k, dim=1)  # (B, top_k)

        # top-k mask (DC 포함)
        mask = torch.zeros_like(magnitudes)
        mask[:, 0] = 1.0  # DC 보존
        mask.scatter_(1, topk_idx, 1.0)

        # learnable weight 적용 (non-inplace)
        weighted_freq = x_freq * (mask.unsqueeze(-1) * self.freq_weight.unsqueeze(0))

        # iFFT back to time domain
        x_time = torch.fft.irfft(weighted_freq, n=T, dim=1)  # (B, T, d_model)

        # residual connection
        x_out = x + x_time

        # T → H projection
        x_out = x_out.permute(0, 2, 1)     # (B, d_model, T)
        x_out = self.temporal_proj(x_out)   # (B, d_model, H)
        return x_out.permute(0, 2, 1)       # (B, H, d_model)


class MambaMix(BaseTemporalMixer):
    """S6 SSM (Mamba simplified) — 선택적 상태 공간 모델.

    input-dependent A, B, C로 시퀀스를 처리.
    O(T) 복잡도 — AttentionMix의 O(T²) 대비 효율적.
    long-range dependency + local selectivity.
    """

    def __init__(self, seq_len: int, pred_len: int, d_model: int,
                 state_dim: int = 16, n_layers: int = 2,
                 dropout: float = 0.1, **kwargs):
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.state_dim = state_dim

        # S6 layers
        self.proj_ins = nn.ModuleList()
        self.dt_projs = nn.ModuleList()
        self.A_logs = nn.ParameterList()
        self.Ds = nn.ParameterList()
        self.B_projs = nn.ModuleList()
        self.C_projs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.drops = nn.ModuleList()
        for _ in range(n_layers):
            self.proj_ins.append(nn.Linear(d_model, d_model * 2))
            self.dt_projs.append(nn.Linear(d_model, d_model))
            self.A_logs.append(nn.Parameter(torch.log(torch.randn(d_model, state_dim).abs() + 1e-4)))
            self.Ds.append(nn.Parameter(torch.ones(d_model)))
            self.B_projs.append(nn.Linear(d_model, state_dim))
            self.C_projs.append(nn.Linear(d_model, state_dim))
            self.norms.append(nn.LayerNorm(d_model))
            self.drops.append(nn.Dropout(dropout))

        self.temporal_proj = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        for i in range(self.n_layers):
            residual = x
            x = self.norms[i](x)

            # S6 scan
            xz = self.proj_ins[i](x)
            x_in, z = xz.chunk(2, dim=-1)

            # input-dependent parameters
            dt = nn.functional.softplus(self.dt_projs[i](x_in))  # (B, T, d_model)
            B = self.B_projs[i](x_in)                     # (B, T, state_dim)
            C = self.C_projs[i](x_in)                     # (B, T, state_dim)
            A = -torch.exp(self.A_logs[i].float())         # (d_model, state_dim)

            # discretize + scan (simplified sequential)
            Bt, T, D = x_in.shape
            h = torch.zeros(Bt, D, self.state_dim, device=x.device)
            ys = []
            for t in range(T):
                # h = A_bar * h + B_bar * x
                dt_t = dt[:, t, :].unsqueeze(-1)           # (B, D, 1)
                A_bar = torch.exp(A.unsqueeze(0) * dt_t)   # (B, D, N)
                B_bar = dt_t * B[:, t, :].unsqueeze(1)     # (B, D, N) via broadcast
                h = A_bar * h + B_bar * x_in[:, t, :].unsqueeze(-1)
                y_t = (h * C[:, t, :].unsqueeze(1)).sum(-1) # (B, D)
                ys.append(y_t)

            y = torch.stack(ys, dim=1)  # (B, T, D)
            y = y * torch.sigmoid(z)    # gating
            y = y + x_in * self.Ds[i]   # skip
            x = self.drops[i](y) + residual

        # T → H
        x = x.permute(0, 2, 1)
        x = self.temporal_proj(x)
        return x.permute(0, 2, 1)


TEMPORAL_MIXER_REGISTRY: dict[str, type[BaseTemporalMixer]] = {
    "LinearMix": LinearMix,
    "MLPMix": MLPMix,
    "GatedMLPMix": GatedMLPMix,
    "PatchMLPMix": PatchMLPMix,
    "AttentionMix": AttentionMix,
    "PatchAttentionMix": PatchAttentionMix,
    "ConvMix": ConvMix,
    "RecurrentMix": RecurrentMix,
    "FrequencyMix": FrequencyMix,
    "MambaMix": MambaMix,
    # v1 호환
    "Linear": LinearMix,
    "PatchMLP": PatchMLPMix,
}

# Patch 전용 mixer 목록
PATCH_MIXERS = {"PatchMLPMix", "PatchAttentionMix", "PatchMLP"}
