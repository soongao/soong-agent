from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from agent_core.config.models import AgentCoreConfig, ModelConfig
from agent_core.config.paths import ResolvedPaths, resolve_runtime_paths
from agent_core.errors import ConfigError


def load_config(path: str | Path) -> AgentCoreConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"config.toml does not exist: {config_path}")
    try:
        data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"failed to parse config.toml: {config_path}", details={"error": str(exc)}) from exc
    except OSError as exc:
        raise ConfigError(f"failed to read config.toml: {config_path}", details={"error": str(exc)}) from exc
    try:
        return AgentCoreConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError("config.toml schema validation failed", details={"errors": exc.errors()}) from exc


def load_runtime_config(
    *,
    home_dir: str | Path | None = None,
    config_path: str | Path | None = None,
    project_dir: str | Path | None = None,
    session_db_path: str | Path | None = None,
) -> tuple[AgentCoreConfig, ResolvedPaths]:
    initial_paths = resolve_runtime_paths(
        home_dir=home_dir,
        config_path=config_path,
        project_dir=project_dir,
        session_db_path=session_db_path,
    )
    config = load_config(initial_paths.config_path)
    paths = resolve_runtime_paths(
        home_dir=initial_paths.home_dir,
        config_path=initial_paths.config_path,
        project_dir=initial_paths.project_dir,
        session_db_path=session_db_path,
        config_session_db_path=config.context.session_db_path,
        plan_dir_template=config.plan.default_dir,
        task_dir_template=config.task.wal_dir,
    )
    return config, paths


def write_required_project_dirs(paths: ResolvedPaths) -> None:
    paths.plan_dir.mkdir(parents=True, exist_ok=True)
    paths.task_dir.mkdir(parents=True, exist_ok=True)
    paths.session_db_path.parent.mkdir(parents=True, exist_ok=True)


def deep_merge_model(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_model(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_model_config(config: AgentCoreConfig, profile: str | dict[str, Any] | None = None) -> ModelConfig:
    base = config.model.model_dump(mode="python")
    override: dict[str, Any] | None = None
    if isinstance(profile, str):
        override = config.model_overrides.get(profile)
    elif isinstance(profile, dict):
        override = profile
    if not override:
        return config.model
    return ModelConfig.model_validate(deep_merge_model(base, override))
