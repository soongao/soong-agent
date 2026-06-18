from __future__ import annotations

from pathlib import Path

from agent_core.assets.loader import read_asset
from agent_core.config.loader import load_config
from agent_core.config.paths import resolve_home_dir, resolve_project_dir as resolve_core_project_dir
from agent_core.errors import ConfigError


class HubConfigBootstrapError(RuntimeError):
    pass


class HubConfigValidationError(RuntimeError):
    pass


def bootstrap_default_config(*, home_dir: str | Path | None = None) -> Path | None:
    resolved_home = resolve_home_dir(home_dir)
    config_path = resolved_home / "config.toml"
    if config_path.exists():
        return None
    resolved_home.mkdir(parents=True, exist_ok=True)
    config_path.write_text(read_asset("template.config.default"), encoding="utf-8")
    return config_path


def validate_config(*, home_dir: str | Path | None = None) -> dict:
    resolved_home = resolve_home_dir(home_dir)
    config_path = resolved_home / "config.toml"
    try:
        config = load_config(config_path)
    except ConfigError as exc:
        raise HubConfigValidationError(str(exc)) from exc
    return {
        "config_path": str(config_path),
        "provider": config.model.provider,
        "model": config.model.name,
        "base_url": config.model.base_url,
    }


def hub_db_path(*, home_dir: str | Path | None = None) -> Path:
    return resolve_home_dir(home_dir) / "hub" / "hub.db"


def resolve_project_dir(path: str | Path | None = None) -> Path:
    return resolve_core_project_dir(path)
