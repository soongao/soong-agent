from __future__ import annotations

from pathlib import Path

from agent_core.assets.loader import read_asset
from agent_core.config.paths import resolve_home_dir


def ensure_default_config() -> Path | None:
    home_dir = resolve_home_dir()
    config_path = home_dir / "config.toml"
    if config_path.exists():
        return None
    home_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(read_asset("template.config.default"), encoding="utf-8")
    return config_path
