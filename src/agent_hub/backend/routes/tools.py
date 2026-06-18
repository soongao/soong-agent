from __future__ import annotations

from fastapi import APIRouter, Request

from agent_hub.backend.state import HubAppState, require_runtime_bridge

router = APIRouter(prefix="/tools")


@router.get("")
async def list_tools(request: Request) -> dict:
    state: HubAppState = request.app.state.hub
    runtime = require_runtime_bridge(state).runtime
    return {
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "permission": tool.permission,
                "tags": sorted(tool.tags),
                "enabled": True,
            }
            for tool in runtime._effective_tools(agent_role="orchestrator")
        ]
    }
