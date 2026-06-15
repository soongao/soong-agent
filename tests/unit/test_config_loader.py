from __future__ import annotations

from pathlib import Path

import pytest

from agent_core.config import load_runtime_config
from agent_core.config.loader import resolve_model_config
from agent_core.errors import ConfigError
from tests.conftest import write_config


def test_missing_config_fails(isolated_dirs) -> None:
    _home, project = isolated_dirs
    with pytest.raises(ConfigError):
        load_runtime_config(project_dir=project)


def test_project_file_resolves_parent(isolated_dirs) -> None:
    home, project = isolated_dirs
    write_config(home)
    file_path = project / "src" / "a.py"
    file_path.parent.mkdir()
    file_path.write_text("print('x')\n", encoding="utf-8")
    _config, paths = load_runtime_config(project_dir=file_path)
    assert paths.project_dir == file_path.parent.resolve()


def test_project_config_ignored(isolated_dirs) -> None:
    home, project = isolated_dirs
    write_config(home)
    project_config = project / ".soong-agent" / "config.toml"
    project_config.parent.mkdir()
    project_config.write_text("[model]\nprovider='ollama'\nname='ignored'\n", encoding="utf-8")
    config, _paths = load_runtime_config(project_dir=project)
    assert config.model.provider == "fake"


def test_resolve_model_config_named_and_inline_profile(isolated_dirs) -> None:
    home, project = isolated_dirs
    path = write_config(home)
    path.write_text(
        path.read_text(encoding="utf-8")
        + "\n[model_overrides.compact]\nname = \"compact-model\"\nmax_output_tokens = 128\n\n",
        encoding="utf-8",
    )
    config, _paths = load_runtime_config(project_dir=project)
    named = resolve_model_config(config, "compact")
    inline = resolve_model_config(config, {"name": "inline-model", "temperature": 0.0})
    assert named.name == "compact-model"
    assert named.provider == config.model.provider
    assert named.max_output_tokens == 128
    assert inline.name == "inline-model"
    assert inline.temperature == 0.0
    assert inline.provider == config.model.provider
