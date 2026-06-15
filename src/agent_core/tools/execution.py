from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_core.artifacts import ArtifactManager
from agent_core.config.models import AgentCoreConfig
from agent_core.permissions import PermissionCallback, PermissionSessionCache
from agent_core.types.tools import ToolDefinition


ToolHandler = Callable[["ToolExecutionContext", dict[str, Any]], Awaitable[Any]]


@dataclass
class ToolExecutionContext:
    session_id: str
    run_id: str
    agent_id: str
    agent_role: str
    project_dir: Path
    home_dir: Path
    config: AgentCoreConfig
    artifact_manager: ArtifactManager
    permission_callback: PermissionCallback | None = None
    permission_cache: PermissionSessionCache | None = None
    parent_agent_id: str | None = None
    parent_run_id: str | None = None
    cwd: Path | None = None
    debug: bool = False
    allowed_tool_names: set[str] | None = None
    effective_tool_definitions: dict[str, ToolDefinition] | None = None
    services: dict[str, Any] | None = None
    hooks: list[dict[str, Any]] | None = None

    @property
    def effective_cwd(self) -> Path:
        return self.cwd or self.project_dir

    def service(self, name: str) -> Any:
        if not self.services or name not in self.services:
            raise KeyError(f"runtime service not available: {name}")
        return self.services[name]
