from __future__ import annotations

from pathlib import Path

import pytest

from agent_core.config import load_runtime_config
from agent_core.config.paths import resolve_home_dir
from agent_core.config.loader import resolve_model_config
from agent_core.errors import ConfigError
from tests.conftest import write_config


def test_missing_config_fails(isolated_dirs) -> None:
    _home, project = isolated_dirs
    with pytest.raises(ConfigError):
        load_runtime_config(project_dir=project)


def test_isolated_dirs_use_soong_agent_test_runs_layout(isolated_dirs, monkeypatch) -> None:
    home, project = isolated_dirs
    # The exact pytest temp prefix is not contractual, but the user-facing layout is.
    assert home.name == "home"
    assert project.name == "project"
    assert home.parent == project.parent
    assert home.parent.parent.name == "test-runs"
    assert home.parent.parent.parent.name == ".soong-agent"
    assert str(home).startswith(str(Path.home()))
    assert str(project).startswith(str(Path.home()))


def test_isolated_dirs_do_not_create_non_test_run_soong_agent_entries(isolated_dirs) -> None:
    home, project = isolated_dirs
    soong_home = home.parent.parent.parent
    entries = {entry.name for entry in soong_home.iterdir()}

    assert entries == {"test-runs"}
    assert home.is_relative_to(soong_home / "test-runs")
    assert project.is_relative_to(soong_home / "test-runs")


def test_invalid_toml_fails_with_config_path(isolated_dirs) -> None:
    home, project = isolated_dirs
    config_path = home / "config.toml"
    config_path.write_text("[model\n", encoding="utf-8")
    with pytest.raises(ConfigError) as exc_info:
        load_runtime_config(project_dir=project)
    assert str(config_path) in exc_info.value.message


def test_schema_invalid_fails_without_project_dirs(isolated_dirs) -> None:
    home, project = isolated_dirs
    (home / "config.toml").write_text("[model]\nname = 'missing-provider'\n", encoding="utf-8")
    with pytest.raises(ConfigError) as exc_info:
        load_runtime_config(project_dir=project)
    assert exc_info.value.code.value == "config_error"
    assert not (project / ".soong-agent").exists()


def test_invalid_network_policy_default_fails_schema_validation(isolated_dirs) -> None:
    home, project = isolated_dirs
    path = write_config(home)
    path.write_text(
        path.read_text(encoding="utf-8")
        + """

[permissions.network_policy]
default = "maybe"
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError) as exc_info:
        load_runtime_config(project_dir=project)

    assert exc_info.value.code.value == "config_error"
    assert "network_policy" in str(exc_info.value.details["errors"])


def test_invalid_runtime_max_turns_fails_schema_validation(isolated_dirs) -> None:
    home, project = isolated_dirs
    path = write_config(home)
    path.write_text(path.read_text(encoding="utf-8").replace("max_turns = 128", "max_turns = 0"), encoding="utf-8")

    with pytest.raises(ConfigError) as exc_info:
        load_runtime_config(project_dir=project)

    assert exc_info.value.code.value == "config_error"
    assert "max_turns" in str(exc_info.value.details["errors"])


def test_invalid_tool_override_schema_fails_validation(isolated_dirs) -> None:
    home, project = isolated_dirs
    path = write_config(home)
    path.write_text(
        path.read_text(encoding="utf-8")
        + """

[tools.overrides."code.read_file"]
permission = "execute"
tags = ["dangerous"]
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError) as exc_info:
        load_runtime_config(project_dir=project)

    assert exc_info.value.code.value == "config_error"
    assert "tools" in str(exc_info.value.details["errors"])


def test_invalid_mcp_tool_override_schema_fails_validation(isolated_dirs) -> None:
    home, project = isolated_dirs
    path = write_config(home)
    path.write_text(
        path.read_text(encoding="utf-8")
        + """

[tools.mcp.tool_overrides."mcp.local.echo"]
permission = "execute"
description = "bad"
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError) as exc_info:
        load_runtime_config(project_dir=project)

    assert exc_info.value.code.value == "config_error"
    assert "tool_overrides" in str(exc_info.value.details["errors"])


def test_missing_project_path_fails_without_project_dirs(isolated_dirs) -> None:
    home, project = isolated_dirs
    write_config(home)
    missing = project / "missing"
    with pytest.raises(ConfigError):
        load_runtime_config(project_dir=missing)
    assert not (project / ".soong-agent").exists()


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
    assert config.model.provider == "ollama"
    assert config.model.name == "gemma4"


def test_explicit_home_overrides_environment(isolated_dirs, tmp_path, monkeypatch) -> None:
    env_home, project = isolated_dirs
    explicit_home = tmp_path / "explicit-home"
    explicit_home.mkdir()
    write_config(env_home, model_name="env-model")
    write_config(explicit_home, model_name="explicit-model")
    monkeypatch.setenv("SOONG_AGENT_HOME", str(env_home))

    config, paths = load_runtime_config(home_dir=explicit_home, project_dir=project)

    assert paths.home_dir == explicit_home.resolve()
    assert config.model.provider == "ollama"
    assert config.model.name == "explicit-model"


def test_environment_home_is_used_when_home_dir_omitted(isolated_dirs, tmp_path, monkeypatch) -> None:
    env_home, project = isolated_dirs
    other_home = tmp_path / "other-home"
    other_home.mkdir()
    write_config(env_home, model_name="env-model")
    write_config(other_home, model_name="other-model")
    monkeypatch.setenv("SOONG_AGENT_HOME", str(env_home))

    config, paths = load_runtime_config(project_dir=project)

    assert paths.home_dir == env_home.resolve()
    assert config.model.provider == "ollama"
    assert config.model.name == "env-model"


def test_default_home_uses_dot_soong_agent(monkeypatch, tmp_path) -> None:
    fake_user_home = tmp_path / "user"
    fake_user_home.mkdir()
    monkeypatch.delenv("SOONG_AGENT_HOME", raising=False)
    monkeypatch.setenv("HOME", str(fake_user_home))

    assert resolve_home_dir() == (fake_user_home / ".soong-agent").resolve()


def test_project_dir_defaults_to_current_working_directory(isolated_dirs, monkeypatch) -> None:
    home, project = isolated_dirs
    write_config(home)
    monkeypatch.chdir(project)

    _config, paths = load_runtime_config()

    assert paths.project_dir == project.resolve()


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
