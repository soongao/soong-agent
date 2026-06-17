from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from agent_core.errors import ConfigError


@dataclass(frozen=True)
class ResolvedPaths:
    home_dir: Path
    config_path: Path
    project_dir: Path
    session_db_path: Path
    plan_dir: Path
    task_dir: Path


def resolve_home_dir(home_dir: str | Path | None = None) -> Path:
    if home_dir is not None:
        return Path(home_dir).expanduser().resolve()
    env_home = os.environ.get("SOONG_AGENT_HOME")
    if env_home:
        return Path(env_home).expanduser().resolve()
    return Path("~/.soong-agent").expanduser().resolve()


def resolve_project_dir(project_dir: str | Path | None = None) -> Path:
    candidate = Path.cwd() if project_dir is None else Path(project_dir).expanduser()
    if not candidate.exists():
        raise ConfigError(f"project path does not exist: {candidate}")
    if candidate.is_file():
        candidate = candidate.parent
    if not candidate.is_dir():
        raise ConfigError(f"project path is not a directory: {candidate}")
    return candidate.resolve()


def expand_config_path(value: str, *, home_dir: Path, project_dir: Path) -> Path:
    expanded = value.replace("${SOONG_AGENT_HOME}", str(home_dir)).replace("<project>", str(project_dir))
    return Path(expanded).expanduser()


def resolve_runtime_paths(
    *,
    home_dir: str | Path | None,
    config_path: str | Path | None,
    project_dir: str | Path | None,
    session_db_path: str | Path | None,
    config_session_db_path: str = "${SOONG_AGENT_HOME}/sessions.sqlite",
    plan_dir_template: str = "<project>/.soong-agent/plans",
    task_dir_template: str = "<project>/.soong-agent/tasks",
) -> ResolvedPaths:
    resolved_home = resolve_home_dir(home_dir)
    resolved_project = resolve_project_dir(project_dir)
    resolved_config = (
        Path(config_path).expanduser().resolve()
        if config_path is not None
        else (resolved_home / "config.toml").resolve()
    )
    resolved_session_db = (
        Path(session_db_path).expanduser()
        if session_db_path is not None
        else expand_config_path(config_session_db_path, home_dir=resolved_home, project_dir=resolved_project)
    ).resolve()
    plan_dir = expand_config_path(plan_dir_template, home_dir=resolved_home, project_dir=resolved_project).resolve()
    task_dir = expand_config_path(task_dir_template, home_dir=resolved_home, project_dir=resolved_project).resolve()
    return ResolvedPaths(
        home_dir=resolved_home,
        config_path=resolved_config,
        project_dir=resolved_project,
        session_db_path=resolved_session_db,
        plan_dir=plan_dir,
        task_dir=task_dir,
    )
