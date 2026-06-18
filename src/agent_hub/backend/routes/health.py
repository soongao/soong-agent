from __future__ import annotations

from fastapi import APIRouter, Request

from agent_hub.backend.services.context import config_status as build_config_status
from agent_hub.backend.services.context import health_status
from agent_hub.backend.state import HubAppState

router = APIRouter()


@router.get("/health")
async def health(request: Request) -> dict:
    state: HubAppState = request.app.state.hub
    return health_status(state)


@router.get("/config/status")
async def config_status(request: Request) -> dict:
    state: HubAppState = request.app.state.hub
    return build_config_status(state)
