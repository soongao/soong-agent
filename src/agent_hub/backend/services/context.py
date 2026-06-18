from __future__ import annotations

from agent_core.context.instructions import build_auto_instruction_entries
from agent_core.context.skills import build_skill_catalog
from agent_hub.backend.state import HubAppState


def context_status(state: HubAppState) -> dict:
    instructions = build_auto_instruction_entries(home_dir=state.home_dir, project_dir=state.project_dir)
    skills = build_skill_catalog(state.home_dir)
    return {
        "auto_instruction_paths": [str(entry.path) for entry in instructions],
        "skill_count": len(skills),
        "skills": [{"name": skill["name"], "description": skill.get("description") or ""} for skill in skills],
    }


def health_status(state: HubAppState) -> dict:
    if state.startup_error is not None or state.runtime_bridge is None:
        return failed_status(state)
    runtime = state.runtime_bridge.runtime
    return {
        "ok": True,
        "status": "ready",
        "config_path": str(state.config_path),
        "provider": state.config_info.get("provider"),
        "model": state.config_info.get("model"),
        "base_url": state.config_info.get("base_url"),
        "core_started": runtime._started,
        "hub_db_path": str(state.db.path),
        "project_dir": str(state.project_dir),
        "context": context_status(state),
        "warnings": [],
    }


def config_status(state: HubAppState) -> dict:
    if state.startup_error is not None or state.runtime_bridge is None:
        return failed_status(state)
    runtime = state.runtime_bridge.runtime
    return {
        "ok": True,
        "status": "ready",
        "config_path": str(state.config_path),
        "provider": state.config_info.get("provider"),
        "model": state.config_info.get("model"),
        "base_url": state.config_info.get("base_url"),
        "core_started": runtime._started,
        "project_dir": str(state.project_dir),
        "context": context_status(state),
        "warnings": [],
    }


def failed_status(state: HubAppState) -> dict:
    error = state.startup_error or {"code": "core_start_failed", "message": "core runtime is not available", "details": {}}
    return {
        "ok": False,
        "status": "core_failed",
        "config_path": str(state.config_path),
        "provider": state.config_info.get("provider"),
        "model": state.config_info.get("model"),
        "base_url": state.config_info.get("base_url"),
        "core_started": False,
        "hub_db_path": str(state.db.path),
        "project_dir": str(state.project_dir),
        "context": context_status(state),
        "warnings": [],
        "error": error,
    }

