"""Domain Prior Loader — 도메인 지식을 yaml에서 로드하여 KG에 주입.

domain_priors/ 디렉토리의 yaml 파일을 읽어서
Scout 프로파일과 KG 추천에 반영.

domain prior가 없으면 → 순수 데이터 기반 (기존 동작 유지)
domain prior가 있으면 → 데이터 + 도메인 지식 결합
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DomainPrior:
    """로드된 도메인 prior."""
    domain: str = ""
    description: str = ""
    config_overrides: dict = field(default_factory=dict)
    constraint_blacklist: list[str] = field(default_factory=list)
    constraint_recommend: list[str] = field(default_factory=list)
    loss_override: dict | None = None
    rules: list[str] = field(default_factory=list)  # 원본 텍스트 규칙

    def summary(self) -> str:
        parts = [f"Domain: {self.domain}"]
        if self.constraint_blacklist:
            parts.append(f"  Blacklist: {self.constraint_blacklist}")
        if self.loss_override:
            parts.append(f"  Loss: {self.loss_override}")
        if self.config_overrides:
            parts.append(f"  Overrides: {self.config_overrides}")
        return "\n".join(parts)


def load_domain_prior(prior_path: str | Path | None = None,
                       prior_dir: str | Path | None = None) -> DomainPrior | None:
    """도메인 prior 로드.

    Args:
        prior_path: 특정 yaml 파일 경로 (CLI --prior 인자)
        prior_dir: domain_priors/ 디렉토리 (자동 탐색)

    Returns:
        DomainPrior 또는 None (파일 없으면)
    """
    path = None
    if prior_path:
        path = Path(prior_path)
    elif prior_dir:
        d = Path(prior_dir)
        if d.exists():
            yamls = sorted(d.glob("*.yaml"))
            if yamls:
                path = yamls[0]  # 첫 번째 파일

    if path is None or not path.exists():
        return None

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    prior = DomainPrior(
        domain=data.get("domain", ""),
        description=data.get("description", ""),
    )

    # priors 파싱
    priors = data.get("priors", {})
    for bias_type in ("observational", "inductive", "learning"):
        items = priors.get(bias_type, [])
        for item in items:
            if isinstance(item, dict):
                prior.rules.append(f"[{bias_type}] {item.get('rule', '')}")
                override = item.get("config_override", {})
                if override:
                    # loss override
                    if "loss" in override:
                        prior.loss_override = override["loss"]
                    # constraint blacklist/recommend
                    if "constraint_blacklist" in override:
                        prior.constraint_blacklist.extend(override["constraint_blacklist"])
                    if "constraint_recommend" in override:
                        prior.constraint_recommend.extend(override["constraint_recommend"])
                    # encoder overrides
                    if "encoder" in override:
                        prior.config_overrides["encoder"] = override["encoder"]
            elif isinstance(item, str):
                prior.rules.append(f"[{bias_type}] {item}")

    return prior


def apply_prior_to_config(config: dict, prior: DomainPrior) -> tuple[dict, list[str]]:
    """domain prior를 config에 적용. 적용된 규칙 목록 반환."""
    import copy
    config = copy.deepcopy(config)
    applied = []

    # constraint blacklist
    if prior.constraint_blacklist:
        constraints = config.get("constraint", [])
        before = len(constraints)
        config["constraint"] = [
            c for c in constraints
            if (c.get("type") if isinstance(c, dict) else c) not in prior.constraint_blacklist
        ]
        if len(config["constraint"]) < before:
            applied.append(f"PRIOR: constraint blacklist {prior.constraint_blacklist}")

    # constraint recommend
    for cr in prior.constraint_recommend:
        existing = {(c.get("type") if isinstance(c, dict) else c) for c in config.get("constraint", [])}
        if cr not in existing:
            config.setdefault("constraint", []).append({"type": cr})
            applied.append(f"PRIOR: constraint recommend {cr}")

    # loss override
    if prior.loss_override:
        config["loss"] = prior.loss_override
        applied.append(f"PRIOR: loss → {prior.loss_override}")

    # encoder d_model_min
    enc_override = prior.config_overrides.get("encoder", {})
    d_min = enc_override.get("d_model_min")
    if d_min and isinstance(config.get("encoder"), dict):
        current_d = config["encoder"].get("d_model", 64)
        if current_d < d_min:
            config["encoder"]["d_model"] = d_min
            applied.append(f"PRIOR: d_model {current_d} → {d_min}")

    return config, applied
