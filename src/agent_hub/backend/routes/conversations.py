from __future__ import annotations

from fastapi import APIRouter, Request

from agent_hub.backend.models import (
    BranchRequest,
    BranchableNodeResponse,
    ConversationCancelRequest,
    ConversationCreateRequest,
    ConversationListResponse,
    ConversationWorkerAddRequest,
    ConversationWorkerListResponse,
    ForkRequest,
    MessageListResponse,
    MessageSendRequest,
    MessageSendResponse,
    SkillLoadRequest,
)
from agent_hub.backend.services.workers import redact_worker_payload, worker_runtime_status
from agent_hub.backend.services.conversations import fork_conversation, get_active_conversation, switch_branch, validate_skill_load_name
from agent_hub.backend.state import HubAppState, require_runtime_bridge

router = APIRouter(prefix="/conversations")


@router.post("")
async def create_conversation(payload: ConversationCreateRequest, request: Request):
    state: HubAppState = request.app.state.hub
    runtime_bridge = require_runtime_bridge(state)
    return await runtime_bridge.create_conversation(title=payload.title)


@router.get("")
async def list_conversations(request: Request) -> ConversationListResponse:
    state: HubAppState = request.app.state.hub
    return ConversationListResponse(conversations=await state.db.list_conversations())


@router.get("/{conversation_id}")
async def get_conversation(conversation_id: str, request: Request):
    state: HubAppState = request.app.state.hub
    return await get_active_conversation(state, conversation_id)


@router.delete("/{conversation_id}")
async def delete_conversation(conversation_id: str, request: Request):
    state: HubAppState = request.app.state.hub
    runtime_bridge = require_runtime_bridge(state)
    conversation = await get_active_conversation(state, conversation_id)
    return await runtime_bridge.delete_conversation(conversation)


@router.get("/{conversation_id}/messages")
async def list_messages(conversation_id: str, request: Request, limit: int = 100) -> MessageListResponse:
    state: HubAppState = request.app.state.hub
    await get_active_conversation(state, conversation_id)
    return MessageListResponse(messages=await state.db.list_messages(conversation_id, limit=limit))


@router.get("/{conversation_id}/workers")
async def list_conversation_workers(conversation_id: str, request: Request) -> ConversationWorkerListResponse:
    state: HubAppState = request.app.state.hub
    runtime = require_runtime_bridge(state).runtime
    conversation = await get_active_conversation(state, conversation_id)
    worker_ids = set(await state.db.list_conversation_worker_ids(conversation.conversation_id))
    workers = [
        redact_worker_payload(worker.model_dump(mode="json") | worker_runtime_status(runtime, worker))
        for worker in await runtime.list_worker_configs(include_disabled=True, include_deleted=False)
        if worker.worker_id in worker_ids
    ]
    return ConversationWorkerListResponse(workers=workers)


@router.post("/{conversation_id}/workers")
async def add_conversation_worker(conversation_id: str, payload: ConversationWorkerAddRequest, request: Request) -> dict:
    state: HubAppState = request.app.state.hub
    runtime = require_runtime_bridge(state).runtime
    conversation = await get_active_conversation(state, conversation_id)
    worker = await runtime.get_worker_config(payload.worker_id)
    if worker is None or worker.deleted_at is not None:
        from agent_hub.backend.errors import raise_hub_error

        raise_hub_error(404, "worker_not_found", f"Worker not found: {payload.worker_id}")
    await state.db.add_conversation_worker(conversation.conversation_id, worker.worker_id)
    return redact_worker_payload(worker.model_dump(mode="json") | worker_runtime_status(runtime, worker))


@router.delete("/{conversation_id}/workers/{worker_id}")
async def remove_conversation_worker(conversation_id: str, worker_id: str, request: Request) -> dict:
    state: HubAppState = request.app.state.hub
    require_runtime_bridge(state)
    conversation = await get_active_conversation(state, conversation_id)
    await state.db.remove_conversation_worker(conversation.conversation_id, worker_id)
    return {"conversation_id": conversation.conversation_id, "worker_id": worker_id, "removed": True}


@router.post("/{conversation_id}/messages")
async def send_message(conversation_id: str, payload: MessageSendRequest, request: Request) -> MessageSendResponse:
    state: HubAppState = request.app.state.hub
    runtime_bridge = require_runtime_bridge(state)
    conversation = await get_active_conversation(state, conversation_id)
    message, run_id, status = await runtime_bridge.send_message(conversation, payload.text)
    return MessageSendResponse(
        message_id=message.message_id,
        conversation_id=conversation_id,
        core_session_id=conversation.core_session_id,
        core_run_id=run_id,
        status=status,
    )


@router.post("/{conversation_id}/skills/{skill_name}/load")
async def load_skill(conversation_id: str, skill_name: str, payload: SkillLoadRequest, request: Request) -> dict:
    state: HubAppState = request.app.state.hub
    runtime_bridge = require_runtime_bridge(state)
    conversation = await get_active_conversation(state, conversation_id)
    requested_name = validate_skill_load_name(path_name=skill_name, payload_name=payload.name)
    return await runtime_bridge.load_skill_for_conversation(conversation, requested_name)


@router.post("/{conversation_id}/cancel")
async def cancel_conversation(conversation_id: str, payload: ConversationCancelRequest, request: Request) -> dict:
    state: HubAppState = request.app.state.hub
    runtime_bridge = require_runtime_bridge(state)
    conversation = await get_active_conversation(state, conversation_id)
    return await runtime_bridge.cancel_conversation_run(
        conversation,
        core_run_id=payload.core_run_id,
        queue_id=payload.queue_id,
    )


@router.get("/{conversation_id}/branchable-nodes")
async def branchable_nodes(conversation_id: str, request: Request) -> BranchableNodeResponse:
    state: HubAppState = request.app.state.hub
    runtime_bridge = require_runtime_bridge(state)
    conversation = await get_active_conversation(state, conversation_id)
    nodes = await runtime_bridge.runtime.list_branchable_nodes(conversation.core_session_id)
    return BranchableNodeResponse(
        nodes=[
            {
                "core_node_id": node.node_id,
                "preview": node.content_preview,
                "created_at": node.created_at,
                "active": node.active,
            }
            for node in nodes
        ]
    )


@router.post("/{conversation_id}/branch")
async def branch(conversation_id: str, payload: BranchRequest, request: Request) -> dict:
    state: HubAppState = request.app.state.hub
    require_runtime_bridge(state)
    conversation = await get_active_conversation(state, conversation_id)
    return await switch_branch(state, conversation, payload.core_node_id)


@router.post("/{conversation_id}/fork")
async def fork(conversation_id: str, payload: ForkRequest, request: Request) -> dict:
    state: HubAppState = request.app.state.hub
    require_runtime_bridge(state)
    conversation = await get_active_conversation(state, conversation_id)
    return await fork_conversation(state, conversation, core_node_id=payload.core_node_id, title=payload.title)
