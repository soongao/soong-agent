from __future__ import annotations

from fastapi import APIRouter, Request

from agent_hub.backend.models import PermissionDecisionRequest
from agent_hub.backend.state import HubAppState

router = APIRouter(prefix="/permissions")


@router.post("/{permission_request_id}/decision")
async def decide(permission_request_id: str, payload: PermissionDecisionRequest, request: Request) -> dict:
    state: HubAppState = request.app.state.hub
    return await state.permission_bridge.decide(permission_request_id, payload.decision)

