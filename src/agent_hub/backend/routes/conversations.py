from __future__ import annotations

from fastapi import APIRouter, Request

from agent_hub.backend.models import (
    BranchRequest,
    BranchableNodeResponse,
    ConversationCancelRequest,
    ConversationCreateRequest,
    ConversationListResponse,
    ForkRequest,
    MessageListResponse,
    MessageSendRequest,
    MessageSendResponse,
    SkillLoadRequest,
)
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
