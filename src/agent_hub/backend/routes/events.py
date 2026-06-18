from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from agent_hub.backend.events import sse_encode
from agent_hub.backend.state import HubAppState

router = APIRouter()


@router.get("/events")
async def events(request: Request, conversation_id: str | None = None) -> StreamingResponse:
    state: HubAppState = request.app.state.hub

    async def stream():
        async for event in state.event_hub.subscribe(conversation_id):
            yield sse_encode(event)

    return StreamingResponse(stream(), media_type="text/event-stream")

