"""Trainer — 템플릿 기반 학습 실행기. LLM 코드 생성 없이 config → 학습 → 결과."""
from __future__ import annotations

import json
import re


class Trainer:
    """Architect의 JSON config를 받아 결정론적으로 학습을 실행한다.

    LLM 호출 없음. build_model(config) → train loop → 표준 결과 JSON.
    """
    name = "trainer"
    description = "Config 기반 학습 실행 + 표준 결과 출력"

    def __init__(self, cwd: str = "", rules: str = "", benchmark_mode: bool = False):
        self.cwd = cwd
        self.benchmark_mode = benchmark_mode

    def run(self, task: str) -> dict:
        """task에서 config와 데이터 정보를 추출하여 학습 실행."""
        from cballm.blocks.trainer_engine import train_model

        # task에서 필요한 정보 추출
        data_path = self._extract_field(task, "DATA_PATH")
        target_col = self._extract_field(task, "TARGET_COL") or "OT"
        pred_len = int(self._extract_field(task, "PREDICTION_LENGTH") or "96")

        # Architect의 model config JSON 추출
        model_config = self._extract_model_config(task)

        if not data_path:
            return {
                "worker": self.name,
                "response": "ERROR: DATA_PATH를 찾을 수 없음",
                "code": None,
                "execution_result": "METRICS: {}\nBEST_MODEL: FAILED",
            }

        # 파이프라인 설정 추출 (Architect Decision Protocol 결과)
        input_design = model_config.pop("input_design", {})
        preprocessing = model_config.pop("preprocessing", {})
        training = model_config.pop("training", {})

        seq_len = input_design.get("seq_len", 96)
        n_folds = training.get("n_folds", 3)

        print(f"  📦 Config: {json.dumps(model_config, ensure_ascii=False)[:200]}")
        print(f"  📂 Data: {data_path}, Target: {target_col}, H: {pred_len}, seq_len: {seq_len}")

        try:
            # preprocessing을 model_config에 다시 넣어서 train_model이 처리
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
        """task 텍스트에서 필드값 추출."""
        # DATA_PATH = '/path/to/file' 형태
        match = re.search(rf"{field}\s*=\s*['\"]([^'\"]+)['\"]", text)
        if match:
            return match.group(1)
        # DATA_PATH = value (따옴표 없는 경우)
        match = re.search(rf"{field}\s*=\s*(\S+)", text)
        return match.group(1) if match else None

    def _extract_model_config(self, text: str) -> dict:
        """Architect의 JSON config 추출. 없으면 기본 config 반환."""
        from cballm.blocks.builder import list_available_blocks

        # JSON 블록 추출 시도 (여러 전략)
        config = self._try_parse_json(text)

        if config is None:
            print("  ⚠️ Model config 추출 실패, 기본 config 사용")
            return self._default_config()

        # Architect의 전체 설계 JSON에서 블록 config만 추출
        if "models" in config and "encoder" not in config:
            config = self._convert_architect_to_block_config(config)

        # 블록 이름 검증 — hallucinated name 방어
        config = self._validate_block_names(config, list_available_blocks())

        return config

    def _try_parse_json(self, text: str) -> dict | None:
        """여러 전략으로 JSON 추출 시도."""
        import json

        # 전략 1: ```json ... ``` 블록
        match = re.search(r'```json\s*\n(.*?)```', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # 전략 2: 균형 잡힌 중괄호로 JSON 추출
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
                                k in parsed for k in ("encoder", "backbone", "loss", "models")
                            ):
                                return parsed
                        except json.JSONDecodeError:
                            pass
                        break

        return None

    def _validate_block_names(self, config: dict, available: dict) -> dict:
        """블록 이름이 카탈로그에 있는지 검증. 없으면 fallback."""
        slot_map = {
            "encoder": available["encoder"],
            "backbone": available["backbone"],
            "loss": available["loss"],
        }

        for slot, valid_names in slot_map.items():
            if slot in config and isinstance(config[slot], dict):
                block_type = config[slot].get("type", "")
                if block_type not in valid_names:
                    print(f"  ⚠️ 알 수 없는 {slot}: '{block_type}' → fallback")
                    fallback = {"encoder": "Linear", "backbone": "MLP", "loss": "MAE"}
                    config[slot] = {"type": fallback[slot]}

        # regime 검증
        if "regime" in config and config["regime"]:
            if isinstance(config["regime"], dict):
                regime_type = config["regime"].get("type", "")
                if regime_type not in available["regime"]:
                    print(f"  ⚠️ 알 수 없는 regime: '{regime_type}' → 제거")
                    del config["regime"]

        # constraint 검증
        if "constraint" in config:
            valid_constraints = []
            for c in config["constraint"]:
                if isinstance(c, dict) and c.get("type") in available["constraint"]:
                    valid_constraints.append(c)
                else:
                    print(f"  ⚠️ 알 수 없는 constraint: '{c}' → 제거")
            config["constraint"] = valid_constraints

        return config

    def _convert_architect_to_block_config(self, architect_config: dict) -> dict:
        """기존 Architect JSON (models 리스트) → 블록 config 변환."""
        models = architect_config.get("models", [])
        loss = architect_config.get("loss", "MAE")
        regime = architect_config.get("regime_strategy", "none")

        backbone_map = {
            "DLinear": "Linear",
            "Linear": "Linear",
            "PatchTST": "PatchMLP",
            "N-HiTS": "PatchMLP",
        }

        backbone_type = "Linear"  # default
        for m in models:
            if m in backbone_map:
                backbone_type = backbone_map[m]
                break

        config = {
            "encoder": {"type": "Linear"},
            "backbone": {"type": backbone_type},
            "constraint": [],
            "loss": {"type": loss if loss in ["MAE", "MSE", "Huber", "Quantile"] else "MAE"},
        }

        if regime and regime != "none":
            config["regime"] = {"type": "SoftGate", "n_regimes": 2}

        return config

    @staticmethod
    def _default_config() -> dict:
        """안전한 기본 config — MLP backbone, MAE loss."""
        return {
            "encoder": {"type": "Linear"},
            "backbone": {"type": "MLP", "hidden_dim": 256},
            "constraint": [],
            "loss": {"type": "MAE"},
        }
