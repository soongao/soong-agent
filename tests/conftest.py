from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def isolated_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / "home"
    project = tmp_path / "project"
    home.mkdir()
    project.mkdir()
    monkeypatch.setenv("SOONG_AGENT_HOME", str(home))
    return home, project


def write_config(
    home: Path,
    *,
    provider: str = "fake",
    worker_pool: bool = False,
    disabled_tools: list[str] | None = None,
    tool_overrides: dict[str, dict] | None = None,
) -> Path:
    path = home / "config.toml"
    path.write_text(
        f"""
[runtime]
cancel_timeout_ms = 1000

[model]
provider = "{provider}"
base_url = ""
api_key_env = ""
name = "fake-model"
context_window = 8192
max_output_tokens = 1024
temperature = 0.0
timeout_ms = 1000

[context]
session_db_path = "${{SOONG_AGENT_HOME}}/sessions.sqlite"

[permissions]
readonly_default = "allow"
write_without_callback = "deny"
remember_scope = "session"
allow_for_session_enabled = true

[tools]
declarative_enabled = true
disabled = {json.dumps(disabled_tools or [])}
allowed_write_roots = []
allow_tmp_write = false
default_timeout_ms = 1000
max_timeout_ms = 2000
env_allowlist = ["PATH", "HOME", "TMPDIR"]
stdout_limit_bytes = 64
stderr_limit_bytes = 64
sensitive_paths = ["~/.ssh", "~/.gnupg", "~/.aws", "~/.config/gcloud", "*.pem", "*.key", ".env", ".env.*"]
""".strip()
        + _tool_overrides_toml(tool_overrides or {})
        + (
            """

[[agents.worker_pools]]
pool_id = "default"

[[agents.worker_pools.workers]]
worker_id = "worker1"
agent_definition_id = "default_worker_agent"
allowed_tools = ["agent.task_get", "agent.task_query_steps", "agent.task_claim_step", "agent.task_update_step", "code.read_file"]
"""
            if worker_pool
            else ""
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _tool_overrides_toml(overrides: dict[str, dict]) -> str:
    if not overrides:
        return ""
    lines = []
    for tool_name, values in overrides.items():
        lines.append("")
        lines.append(f'[tools.overrides."{tool_name}"]')
        for key, value in values.items():
            lines.append(f"{key} = {json.dumps(value)}")
    return "\n" + "\n".join(lines)


@pytest.fixture
def config_file(isolated_dirs):
    home, _project = isolated_dirs
    return write_config(home)
