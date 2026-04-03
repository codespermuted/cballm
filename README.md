# CBALLM

**Explainable Time Series Modeling through Decision Protocol**

AutoML that tells you *why*, not just *what*.

## The Problem

Existing AutoML tools (AutoGluon, AutoTS) give you a number: "MSE = 0.37". But they can't answer: *Why this model? Why these features? What would change if the data had different characteristics?*

CBALLM doesn't try to beat ensemble AutoML on leaderboard scores. Instead, it produces a **modeling report** — a step-by-step record of every decision, with evidence and reasoning, that explains *why this model for this data*.

## How It Works

```
Your Data
  ↓
Scout (rule-based)     → Data profile: seasonality, stationarity, regime, correlations
  ↓
Engineer (rule-based)  → Feature engineering: lag, rolling, calendar (with leakage verification)
  ↓
Architect (LLM)        → Decision Protocol: 7 micro-questions, each with evidence + default
  ↓
Trainer (template)     → Block assembly + training + standard metrics
  ↓
Critic (rule-based)    → Evaluation + specific feedback → retry or done
```

**Only the Architect uses LLM.** Everything else is deterministic. The LLM answers narrow, bounded questions — not "design a model", but "harmonics = 2, agree?"

## Decision Protocol

The core innovation. Code asks, LLM answers:

```
Step 0: Preprocessing
  Evidence: target_min=2.67, skew=0.97
  Default:  no log transform
  LLM:      agree

Step 1: Input Design
  Evidence: dominant_period=24, pred_len=96
  Default:  seq_len=96
  LLM:      96

Step 2: Encoding
  Evidence: 24h ACF=0.94, 168h ACF=0.86 (2 strong periods)
  Default:  Fourier(n_harmonics=2)
  LLM:      1 | 24h is dominant, 168h is just a multiple

Step 3: Backbone
  Evidence: n_rows=17420, regime=stable
  Default:  Linear
  LLM:      agree | smooth data, complexity unnecessary

Step 4: Constraint + Loss
  Evidence: can_be_negative=true, extreme/normal ratio=1.3
  Default:  MAE
  LLM:      MAE

Step 5: Training Strategy
  Evidence: n_rows=17420
  Default:  3-fold CV
  LLM:      3

Step 6: Regime
  Evidence: stable (0 changes)
  Default:  no gate
  LLM:      agree
```

**This Q&A log IS the modeling report.** Every decision is traceable.

## Slot-Based Block System

Models are assembled from verified blocks via JSON config — no code generation by LLM:

```
Input → [Encoder] → [Backbone] → [Constraint?] → [Regime?] → Output
           ↑            ↑            ↑                ↑
        Linear       Linear      Positivity       SoftGate
        Fourier      PatchMLP    Clamp             HardGate
                                 Smoothness
                                 Monotonic
```

```python
from cballm.blocks.builder import build_model

config = {
    "encoder": {"type": "Fourier", "n_harmonics": 2},
    "backbone": {"type": "Linear"},
    "constraint": [],
    "loss": {"type": "MAE"},
}
model, loss_fn = build_model(config, seq_len=96, pred_len=96, n_features=7)
```

## Quick Start

```bash
# Run full pipeline
python -m cballm data.csv --target OT --horizon 96

# Benchmark mode (standard train/val/test split)
python -m cballm data.csv --target OT --horizon 96 --benchmark

# With domain rules
python -m cballm data.csv --target OT --horizon 96 --rules rules/
```

## Benchmark Results

ETTh1, MS setting (multivariate input → single target), pred_len=96:

| Model | norm_MSE | norm_MAE | Source |
|-------|----------|----------|--------|
| PatchTST | ~0.08 | ~0.19 | Paper (2023) |
| DLinear | ~0.09 | ~0.23 | Paper (2023) |
| **CBALLM (Linear)** | **0.084** | **0.216** | This project |
| CBALLM (PatchMLP) | 0.230 | 0.450 | This project |

CBALLM's Linear backbone achieves DLinear-level performance. The value is not in beating SOTA — it's in the **explainable modeling report** that accompanies every result.

## Architecture Decisions

| Component | Approach | Why |
|-----------|----------|-----|
| Scout | Rule-based (pandas + statsmodels) | Profiling is deterministic — ACF, ADF, correlations |
| Engineer | Rule-based (lag, rolling, calendar) | Feature patterns are well-defined, leakage verification built-in |
| Architect | LLM (Decision Protocol) | Only component needing judgment — "what fits this data?" |
| Trainer | Template engine (build_model + train loop) | Eliminates LLM code generation failures |
| Critic | Rule-based (metric parsing) | Evaluation criteria are objective |

**Why only Architect uses LLM?** We tried having LLM generate training scripts — syntax errors, hallucinated column names, worker confusion. We tried having LLM choose from a menu — it always picked the same complex config. Decision Protocol gives LLM the right scope: narrow, evidence-based judgment calls.

## Infrastructure

- **Standard benchmark splits** (ETTh1/h2: 12:4:4 ratio)
- **Train-only normalization** (no test information leakage)
- **Backbone-specific HP presets** (Linear: lr=1e-3, PatchMLP: lr=1e-4)
- **Leakage post-verification** (feature-future target correlation check)
- **Dual model engine** (Qwopus for reasoning, Qwen Coder for code — swappable)

## What's Not Done Yet

- **MLP/Transformer backbones**: Training instability (epoch-0 early stop). Disabled until standalone quality is verified.
- **Engineer → Trainer full integration**: Feature engineering results now flow to Trainer, but ablation showing feature value is pending.
- **Multi-dataset validation**: Tested on ETTh1/ETTh2. Weather, ECL, and other datasets needed.
- **Open-ended LLM modeling**: Currently "closed questions + confirm". Future: let LLM propose hypotheses from residual analysis.

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Local LLM engine (llama-cpp-python) with ~18GB VRAM
- Optional: statsmodels, neuralforecast

## License

MIT
