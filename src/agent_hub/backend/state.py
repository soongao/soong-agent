from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_hub.backend.database import HubDatabase
from agent_hub.backend.errors import raise_hub_error
from agent_hub.backend.events import HubEventHub
from agent_hub.backend.permissions import PermissionBridge
from agent_hub.backend.runtime import HubRuntimeBridge


@dataclass
class HubAppState:
    home_dir: Path
    project_dir: Path
    config_path: Path
    db: HubDatabase
    event_hub: HubEventHub
    permission_bridge: PermissionBridge
    runtime_bridge: HubRuntimeBridge | None
    config_info: dict[str, Any]
    startup_error: dict[str, Any] | None = None


def require_runtime_bridge(state: HubAppState) -> HubRuntimeBridge:
    if state.runtime_bridge is not None and state.startup_error is None:
        return state.runtime_bridge
    error = state.startup_error or {"code": "core_start_failed", "message": "core runtime is not available", "details": {}}
    raise_hub_error(503, str(error.get("code") or "core_start_failed"), str(error.get("message") or "core runtime is not available"), error.get("details") or {})
