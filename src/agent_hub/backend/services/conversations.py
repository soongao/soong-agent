from __future__ import annotations

from agent_core.errors.codes import ErrorCode
from agent_hub.backend.errors import raise_hub_error
from agent_hub.backend.models import ConversationView
from agent_hub.backend.state import HubAppState


def raise_conversation_not_found(conversation_id: str) -> None:
    raise_hub_error(404, "conversation_not_found", f"Conversation not found: {conversation_id}")


async def get_active_conversation(state: HubAppState, conversation_id: str) -> ConversationView:
    conversation = await state.db.get_conversation(conversation_id)
    if conversation is None or conversation.status == "deleted":
        raise_conversation_not_found(conversation_id)
    return conversation


async def switch_branch(state: HubAppState, conversation: ConversationView, core_node_id: str) -> dict:
    assert state.runtime_bridge is not None
    result = await state.runtime_bridge.runtime.switch_node(conversation.core_session_id, core_node_id)
    if result.error is not None:
        raise_hub_error(
            409 if result.error.code == ErrorCode.SESSION_ACTIVE else 400,
            result.error.code.value,
            result.error.message,
            result.error.details,
        )
    await state.db.update_conversation(conversation.conversation_id, active_core_node_id=core_node_id)
    await state.event_hub.publish("conversation_updated", conversation_id=conversation.conversation_id, payload={"active_core_node_id": core_node_id})
    return result.model_dump(mode="json")


async def fork_conversation(state: HubAppState, conversation: ConversationView, *, core_node_id: str, title: str | None = None) -> dict:
    assert state.runtime_bridge is not None
    forked = await state.runtime_bridge.runtime.fork_session(
        conversation.core_session_id,
        node_id=core_node_id,
        mode="orchestrator",
    )
    if forked.error is not None or not forked.session_id:
        if forked.error is not None:
            raise_hub_error(
                409 if forked.error.code == ErrorCode.SESSION_ACTIVE else 400,
                forked.error.code.value,
                forked.error.message,
                forked.error.details,
            )
        raise_hub_error(400, "internal_error", "Fork did not create a session.")
    new_conversation = await state.db.create_conversation(
        core_session_id=forked.session_id,
        title=title or f"Fork from {core_node_id[:8]}",
    )
    state.permission_bridge.bind_session(core_session_id=forked.session_id, conversation_id=new_conversation.conversation_id)
    await state.event_hub.publish("conversation_created", conversation_id=new_conversation.conversation_id, payload=new_conversation.model_dump(mode="json"))
    return {
        "conversation_id": new_conversation.conversation_id,
        "core_session_id": new_conversation.core_session_id,
        "fork": forked.model_dump(mode="json"),
    }


def validate_skill_load_name(*, path_name: str, payload_name: str | None) -> str:
    requested_name = payload_name.strip() if payload_name else path_name
    if requested_name != path_name:
        raise_hub_error(400, "validation_error", "Skill name path and payload must match.")
    return requested_name

