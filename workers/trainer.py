"""Trainer — 템플릿 기반 학습 실행기 (v2).

v2 변경: normalizer/temporal_mixer/channel_mixer/head 슬롯 지원,
capacity 기반 HP preset, forecasting setting 지원.
"""
from __future__ import annotations

import json
import re


class Trainer:
    """Architect의 JSON config를 받아 결정론적으로 학습을 실행한다."""
    name = "trainer"
    description = "Config 기반 학습 실행 + 표준 결과 출력"

    def __init__(self, cwd: str = "", rules: str = "", benchmark_mode: bool = False):
        self.cwd = cwd
        self.benchmark_mode = benchmark_mode

    def run(self, task: str) -> dict:
        """task에서 config와 데이터 정보를 추출하여 학습 실행."""
        from cballm.blocks.trainer_engine import train_model

        data_path = self._extract_field(task, "DATA_PATH")
        target_col = self._extract_field(task, "TARGET_COL") or "OT"
        pred_len = int(self._extract_field(task, "PREDICTION_LENGTH") or "96")

        model_config = self._extract_model_config(task)

        if not data_path:
            return {
                "worker": self.name,
                "response": "ERROR: DATA_PATH를 찾을 수 없음",
                "code": None,
                "execution_result": "METRICS: {}\nBEST_MODEL: FAILED",
            }

        # 파이프라인 설정 추출
        input_design = model_config.pop("input_design", {})
        preprocessing = model_config.pop("preprocessing", {})
        training = model_config.pop("training", {})
        forecasting_setting = model_config.pop("forecasting_setting", None)
        output_dim = model_config.pop("output_dim", 1)
        recipe_name = model_config.pop("_recipe_name", "custom")

        seq_len = input_design.get("seq_len", 96)
        n_folds = training.get("n_folds", 3)

        # normalizer에 따라 skip_dataset_norm 결정
        normalizer_cfg = model_config.get("normalizer")
        skip_norm = False
        if normalizer_cfg:
            norm_type = normalizer_cfg if isinstance(normalizer_cfg, str) \
                else normalizer_cfg.get("type", "")
            if norm_type in ("RevIN", "RobustScaler"):
                skip_norm = True

        print(f"  Config: {json.dumps(model_config, ensure_ascii=False)[:200]}")
        print(f"  Data: {data_path}, Target: {target_col}, H: {pred_len}, seq_len: {seq_len}")
        print(f"  Setting: {forecasting_setting}, Recipe: {recipe_name}")

        try:
            model_config["preprocessing"] = preprocessing

            result = train_model(
                data_path=data_path,
                target_col=target_col,
                model_config=model_config,
                seq_len=seq_len,
                pred_len=pred_len,
                n_folds=n_folds,
                batch_size=32,
                benchmark_mode=self.benchmark_mode,
            )

            response = result.to_json()
            exec_result = result.to_critic_text()

            # recipe name 추가
            exec_result = f"RECIPE: {recipe_name}\n{exec_result}"

        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            response = f"ERROR: {e}"
            exec_result = f"METRICS: {{}}\nBEST_MODEL: FAILED\n[STDERR]\n{error_msg}"

        return {
            "worker": self.name,
            "response": response,
            "code": None,
            "execution_result": exec_result,
        }

    def _extract_field(self, text: str, field: str) -> str | None:
        match = re.search(rf"{field}\s*=\s*['\"]([^'\"]+)['\"]", text)
        if match:
            return match.group(1)
        match = re.search(rf"{field}\s*=\s*(\S+)", text)
        return match.group(1) if match else None

    def _extract_model_config(self, text: str) -> dict:
        """Architect의 JSON config 추출."""
        from cballm.blocks.builder import list_available_blocks

        config = self._try_parse_json(text)

        if config is None:
            print("  Model config 추출 실패, 기본 config 사용")
            return self._default_config()

        # v1 포맷 감지
        if "backbone" in config and "temporal_mixer" not in config:
            # builder가 v1→v2 변환을 담당하므로 그대로 전달
            pass

        # 블록 이름 검증
        config = self._validate_block_names(config, list_available_blocks())

        return config

    def _try_parse_json(self, text: str) -> dict | None:
        # 전략 1: ```json ... ``` 블록
        match = re.search(r'```json\s*\n(.*?)```', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # 전략 2: 균형 잡힌 중괄호
        for i, ch in enumerate(text):
            if ch == '{':
                depth = 0
                for j in range(i, len(text)):
                    if text[j] == '{':
                        depth += 1
                    elif text[j] == '}':
                        depth -= 1
                    if depth == 0:
                        candidate = text[i:j+1]
                        try:
                            parsed = json.loads(candidate)
                            if isinstance(parsed, dict) and any(
                                k in parsed for k in (
                                    "encoder", "temporal_mixer", "backbone",
                                    "normalizer", "loss",
                                )
                            ):
                                return parsed
                        except json.JSONDecodeError:
                            pass
                        break

        return None

    def _validate_block_names(self, config: dict, available: dict) -> dict:
        """블록 이름이 카탈로그에 있는지 검증."""
        slot_map = {
            "encoder": available.get("encoder", []),
            "temporal_mixer": available.get("temporal_mixer", []),
            "normalizer": available.get("normalizer", []),
            "head": available.get("head", []),
            "loss": available.get("loss", []),
            # v1 호환
            "backbone": available.get("temporal_mixer", []),
        }

        for slot, valid_names in slot_map.items():
            if slot in config and isinstance(config[slot], dict):
                block_type = config[slot].get("type", "")
                if block_type and valid_names and block_type not in valid_names:
                    print(f"  알 수 없는 {slot}: '{block_type}' → fallback")
                    fallback = {
                        "encoder": "LinearProjection",
                        "temporal_mixer": "LinearMix",
                        "backbone": "Linear",
                        "normalizer": "RevIN",
                        "head": "LinearHead",
                        "loss": "MAE",
                    }
                    config[slot] = {"type": fallback.get(slot, "LinearMix")}

        # channel_mixer 검증
        if "channel_mixer" in config and config["channel_mixer"]:
            ch = config["channel_mixer"]
            if isinstance(ch, dict):
                ch_type = ch.get("type", "")
                valid_ch = available.get("channel_mixer", [])
                if ch_type and valid_ch and ch_type not in valid_ch:
                    print(f"  알 수 없는 channel_mixer: '{ch_type}' → 제거")
                    config["channel_mixer"] = None

        # constraint 검증
        if "constraint" in config:
            valid_constraints = []
            for c in config["constraint"]:
                if isinstance(c, dict) and c.get("type") in available.get("constraint", []):
                    valid_constraints.append(c)
                else:
                    print(f"  알 수 없는 constraint: '{c}' → 제거")
            config["constraint"] = valid_constraints

        return config

    @staticmethod
    def _default_config() -> dict:
        """안전한 기본 config (v2)."""
        return {
            "normalizer": {"type": "RevIN"},
            "encoder": {"type": "LinearProjection"},
            "temporal_mixer": {"type": "LinearMix"},
            "channel_mixer": None,
            "head": {"type": "LinearHead"},
            "constraint": [],
            "loss": {"type": "MAE"},
        }
