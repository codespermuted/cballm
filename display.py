"""CBALLM Display — 파이프라인 출력 포맷팅 (v2)."""
from __future__ import annotations

import json
from typing import Any


# ── ANSI Colors ──

class C:
    """ANSI color codes."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    # Colors
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"


def _b(text: str) -> str:
    return f"{C.BOLD}{text}{C.RESET}"


def _dim(text: str) -> str:
    return f"{C.DIM}{text}{C.RESET}"


def _c(text: str, color: str) -> str:
    return f"{color}{text}{C.RESET}"


# ── Box Drawing ──

def box(lines: list[str], color: str = C.CYAN, width: int = 56) -> str:
    """Draw a box around lines."""
    out = [f"{color}{'━' * width}{C.RESET}"]
    for line in lines:
        out.append(f"  {line}")
    out.append(f"{color}{'━' * width}{C.RESET}")
    return "\n".join(out)


def section_header(title: str, icon: str = "", color: str = C.CYAN) -> str:
    """Section header with line."""
    label = f" {icon} {title} " if icon else f" {title} "
    line_len = max(0, 56 - len(label))
    left = line_len // 2
    right = line_len - left
    return f"\n{color}{'─' * left}{C.BOLD}{label}{C.RESET}{color}{'─' * right}{C.RESET}"


def step_label(name: str, status: str = "", color: str = C.BLUE) -> str:
    """Worker step label."""
    s = f"  {color}{C.BOLD}{name}{C.RESET}"
    if status:
        s += f" {_dim(status)}"
    return s


# ── Pipeline Display ──

def print_banner():
    print()
    print(f"  {_b('CBALLM')} {_dim('AI Data Scientist Orchestrator')}")
    print(f"  {_dim('Scout → KG → Architect → Engineer → Trainer → Critic')}")
    print()


def print_pipeline_start(data_path: str, target_col: str, horizon: int,
                         instructions: str = ""):
    lines = [
        f"{_dim('Data')}     {_b(data_path)}",
        f"{_dim('Target')}   {_c(target_col, C.GREEN)}  {_dim('|')}  "
        f"{_dim('Horizon')}  {_c(str(horizon), C.GREEN)}",
    ]
    if instructions:
        lines.append(f"{_dim('Note')}     {instructions}")
    print(box(lines))


def print_scout_result(profile_text: str):
    """Scout 결과에서 핵심 정보만 추출하여 표시."""
    import re

    shape_m = re.search(r'\((\d+),\s*(\d+)\)', profile_text)
    rows = shape_m.group(1) if shape_m else "?"
    cols = shape_m.group(2) if shape_m else "?"

    freq_m = re.search(r'Frequency:\s*(\S+)', profile_text)
    freq = freq_m.group(1) if freq_m else "?"

    stat_m = re.search(r'(non-stationary|stationary)', profile_text, re.I)
    stationarity = stat_m.group(1) if stat_m else "?"

    season_m = re.search(r'Seasonality:\s*\[(.+?)\]', profile_text)
    seasonality = season_m.group(1).strip() if season_m else "none"

    miss_m = re.search(r'Missing:\s*([0-9.]+%?)', profile_text)
    missing = miss_m.group(1) if miss_m else "0%"

    print(f"  {_c(f'{rows:>7} rows', C.WHITE)}  {_dim('×')}  "
          f"{_c(f'{cols} features', C.WHITE)}  {_dim('|')}  "
          f"{_c(freq, C.YELLOW)}  {_dim('|')}  "
          f"{_c(stationarity, C.YELLOW)}")
    print(f"  {_dim('Seasonality:')} {seasonality}")
    if missing and missing not in ("0%", "0.0%", "0"):
        print(f"  {_c(f'Missing: {missing}', C.YELLOW)}")


def print_kg_result(kg_text: str):
    """KG Matcher 결과에서 핵심 정보 표시."""
    import re
    setting_m = re.search(r'FORECASTING_SETTING=(\S+)', kg_text)
    setting = setting_m.group(1) if setting_m else "?"

    recipes = []
    for m in re.finditer(r'\[(\d+)\]\s+(\S+)\s+\(capacity=(\w+)\)', kg_text):
        recipes.append(f"{m.group(2)} ({m.group(3)})")

    norm_m = re.search(r'NORMALIZER=(\S+)', kg_text)
    norm = norm_m.group(1) if norm_m else "none"

    print(f"  {_dim('Setting:')} {_c(setting, C.GREEN)}  {_dim('|')}  "
          f"{_dim('Normalizer:')} {_c(norm, C.GREEN)}")
    if recipes:
        print(f"  {_dim('Recipes:')} {', '.join(recipes)}")


def print_iteration_header(iteration: int, max_iter: int):
    bar = f"{C.CYAN}{'━' * 56}{C.RESET}"
    label = f"  Iteration {iteration}/{max_iter}"
    print(f"\n{bar}")
    print(f"  {_b(label.strip())}")
    print(bar)


def print_architect_decisions(decisions_text: str, config_json: str):
    """Architect 결정 사항을 테이블 형태로 표시."""
    import re

    # Parse decisions from execution_result
    decisions = []
    for m in re.finditer(
        r'Step (\S+): (.+?)\n\s*Evidence: (.+?)\n\s*Default: (.+?)\n\s*LLM: (.+?)\n(?:\s*Insight: (.+?)\n)?\s*-> Decision: (.+?)(?:\n|$)',
        decisions_text
    ):
        decisions.append({
            "step": m.group(1),
            "topic": m.group(2),
            "evidence": m.group(3),
            "llm": m.group(5),
            "insight": m.group(6) or "",
            "decision": m.group(7).strip(),
        })

    if not decisions:
        # fallback: just print -> Decision lines
        for line in decisions_text.split("\n"):
            if "-> Decision:" in line:
                print(f"    {_dim('->')} {line.split('-> Decision:')[1].strip()}")
        return

    # Recipe (step 1) highlighted
    recipe_d = next((d for d in decisions if d["step"] == "1"), None)
    if recipe_d:
        print(f"\n  {_dim('Recipe')}    {_c(_b(recipe_d['decision']), C.GREEN)}")
        if recipe_d["insight"]:
            print(f"             {_dim(recipe_d['insight'])}")

    # Architecture slots in compact form
    slots = []
    for d in decisions:
        if d["step"] in ("2", "3", "4a", "4b", "5"):
            label_map = {
                "2": "Encoder", "3": "Mixer", "4a": "Channel",
                "4b": "Loss", "5": "Constraint",
            }
            label = label_map.get(d["step"], d["topic"])
            slots.append(f"{_dim(label + ':')} {_c(d['decision'], C.WHITE)}")

    if slots:
        # Print 2-3 per line
        line1 = "  " + "  ".join(slots[:3])
        print(line1)
        if len(slots) > 3:
            print("  " + "  ".join(slots[3:]))

    # Training strategy (steps 6a, 6b, 6c)
    strategy_parts = []
    for d in decisions:
        if d["step"].startswith("6"):
            strategy_parts.append(d["decision"])
    if strategy_parts:
        print(f"  {_dim('Strategy:')} {' | '.join(strategy_parts)}")


def print_training_config(block_name: str, capacity: str, lr: float,
                          epochs: int, patience: int):
    print(f"  {_dim('HP:')} lr={lr}  epochs={epochs}  patience={patience}  "
          f"{_dim(f'({block_name}, {capacity})')}")


def print_data_info(n_total: int, n_features: int, target_col: str,
                    target_idx: int, preprocessing: dict):
    parts = [f"{n_total:,} rows", f"{n_features} features",
             f"target={target_col}[{target_idx}]"]
    prep_parts = []
    if preprocessing.get("log_transform"):
        prep_parts.append("log")
    if preprocessing.get("differencing"):
        prep_parts.append("diff")
    if prep_parts:
        parts.append(f"prep=[{'+'.join(prep_parts)}]")
    print(f"  {_dim('Data:')} {', '.join(parts)}")


def print_split_info(n_folds: int, test_start: int, test_end: int,
                     norm_type: str, train_end: int):
    print(f"  {_dim('Split:')} {n_folds} folds  "
          f"{_dim('|')}  test=[{test_start:,}:{test_end:,}]  "
          f"{_dim('|')}  norm=DatasetNorm"
          f"{f'+{norm_type}' if norm_type else ''}")


def print_fold_header():
    print()
    print(f"  {_dim('Fold')}  {_dim('Train'):>9}  {_dim('Val MAE'):>10}  "
          f"{_dim('nMSE'):>8}  {_dim('Epoch'):>7}  {_dim('Time'):>6}")
    print(f"  {_dim('─' * 52)}")


def print_fold_result(fold: int, n_folds: int, train_size: int,
                      mae: float, norm_mse: float, best_epoch: int,
                      epochs: int, fit_time: float):
    # Color MAE: green if good, yellow if medium, red if bad
    mae_color = C.GREEN if mae < 3.0 else (C.YELLOW if mae < 6.0 else C.RED)
    mse_color = C.GREEN if norm_mse < 1.0 else (C.YELLOW if norm_mse < 5.0 else C.RED)

    print(f"  {_dim(f'{fold}/{n_folds}')}   {train_size:>8,}  "
          f"{_c(f'{mae:>10.4f}', mae_color)}  "
          f"{_c(f'{norm_mse:>8.4f}', mse_color)}  "
          f"{best_epoch:>3}/{epochs:<3}  "
          f"{fit_time:>5.1f}s")


def print_cv_summary(cv_mean_mae: float, cv_std_mae: float):
    print(f"  {_dim('─' * 52)}")
    print(f"  {_dim('Mean')}           "
          f"{_b(f'{cv_mean_mae:>10.4f}')}  "
          f"{_dim(f'(+/-{cv_std_mae:.4f})')}")


def print_test_results(test_mae: float, norm_mse: float,
                       normal_mae: float, extreme_mae: float,
                       extreme_n: int):
    print(f"\n  {_dim('Test Results:')}")

    mae_color = C.GREEN if test_mae < 3.0 else (C.YELLOW if test_mae < 6.0 else C.RED)
    print(f"    Overall MAE  {_c(f'{test_mae:.4f}', mae_color)}  "
          f"{_dim(f'norm_MSE={norm_mse:.4f}')}")

    if normal_mae == normal_mae:  # not NaN
        n_color = C.GREEN if normal_mae < 3.0 else (C.YELLOW if normal_mae < 6.0 else C.RED)
        print(f"    Normal  MAE  {_c(f'{normal_mae:.4f}', n_color)}")
    if extreme_mae == extreme_mae:  # not NaN
        e_color = C.GREEN if extreme_mae < 5.0 else (C.YELLOW if extreme_mae < 10.0 else C.RED)
        print(f"    Extreme MAE  {_c(f'{extreme_mae:.4f}', e_color)}  "
              f"{_dim(f'({extreme_n:,} samples)')}")


def print_critic_verdict(verdict: str, mae: Any, analysis: str,
                         suggestions: list[str]):
    color_map = {
        "DONE": C.GREEN, "RETRY_HP": C.YELLOW,
        "RETRY_RECIPE": C.MAGENTA, "RETRY_BLOCK": C.RED,
    }
    color = color_map.get(verdict, C.WHITE)
    icon_map = {
        "DONE": "OK", "RETRY_HP": ">>",
        "RETRY_RECIPE": ">>", "RETRY_BLOCK": ">>",
    }
    icon = icon_map.get(verdict, "?")

    mae_str = f"MAE={mae:.4f}" if isinstance(mae, (int, float)) else f"MAE={mae}"

    print(f"\n  {_c(f'[{icon}]', color)} {_c(_b(verdict), color)}  {_dim(mae_str)}")
    if analysis:
        print(f"      {_dim(analysis[:120])}")
    for s in suggestions[:2]:
        print(f"      {_dim('> ' + s[:100])}")


def print_final_report(report: dict, verdict: dict):
    metrics = report.get("metrics", {})
    model = report.get("best_model", "?")
    iterations = report.get("iterations", 0)
    analysis = verdict.get("analysis", "")

    print(f"\n{C.CYAN}{'━' * 56}{C.RESET}")
    print(f"  {_b('FINAL RESULTS')}")
    print(f"{C.CYAN}{'━' * 56}{C.RESET}")

    print(f"\n  {_dim('Model')}       {_c(_b(model), C.GREEN)}")
    print(f"  {_dim('Iterations')}  {iterations}")
    print()

    # Metrics
    if metrics:
        mae = metrics.get("MAE", "?")
        mse = metrics.get("MSE", "?")
        norm_mse = metrics.get("norm_MSE", "?")

        mae_color = C.GREEN if isinstance(mae, (int, float)) and mae < 3.0 else C.YELLOW
        print(f"  {_dim('MAE')}         {_c(_b(f'{mae}'), mae_color)}")
        if isinstance(mse, (int, float)):
            print(f"  {_dim('MSE')}         {mse:.4f}")
        if isinstance(norm_mse, (int, float)):
            mse_color = C.GREEN if norm_mse < 1.0 else C.YELLOW
            print(f"  {_dim('norm_MSE')}    {_c(f'{norm_mse:.4f}', mse_color)}  "
                  f"{_dim('(benchmark scale)')}")

    # Normal/Extreme from verdict
    normal = verdict.get("normal_metric", {})
    extreme = verdict.get("extreme_metric", {})
    if normal:
        print(f"  {_dim('Normal')}      MAE={normal.get('MAE', '?')}")
    if extreme:
        print(f"  {_dim('Extreme')}     MAE={extreme.get('MAE', '?')}")

    print()
    if analysis:
        print(f"  {_dim(analysis[:200])}")
    print(f"\n{C.CYAN}{'━' * 56}{C.RESET}\n")
