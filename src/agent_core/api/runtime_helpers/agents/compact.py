from __future__ import annotations

from typing import Any

from agent_core.api.runtime_helpers.memory import _compact_input
from agent_core.compact import DEFAULT_COMPACT_AGENT_ID
from agent_core.config.loader import resolve_model_config
from agent_core.errors import AgentCoreError
from agent_core.errors.codes import ErrorCode
from agent_core.events import make_event
from agent_core.providers import ModelMessage, ModelRequest, SystemBlock
from agent_core.providers.base import ModelRole
from agent_core.storage import new_id
from agent_core.types import ErrorPayload, RunStatus, TextBlock

from agent_core.api.runtime_helpers.model import _ensure_provider_supports_request


async def run_compact_agent(
    runtime: Any,
    *,
    session_id: str,
    source_node_ids: list[str] | None = None,
    reason: str = "manual",
    first_kept_node_id: str | None = None,
) -> dict[str, Any]:
    await runtime._ensure_started()
    assert runtime.store and runtime.paths and runtime.config
    definition = runtime.agent_definitions.get(DEFAULT_COMPACT_AGENT_ID)
    if definition is None:
        raise AgentCoreError(ErrorCode.INVALID_AGENT_DEFINITION, "default compact agent missing")
    active_node_id = await runtime.store.active_node_id(session_id)
    active_node = await runtime.store.get_node(session_id, active_node_id) if active_node_id else None
    session = await runtime.store.get_session(session_id)
    parent_agent_id = active_node.agent_id if active_node is not None else (str(session["root_agent_id"]) if session else None)
    parent_run_id = active_node.run_id if active_node is not None else None
    agent_id = new_id("agent_compact")
    run_id = new_id("run_compact")
    await runtime.store.ensure_agent(
        agent_id=agent_id,
        session_id=session_id,
        agent_type="fork",
        status="running",
        parent_agent_id=parent_agent_id,
        created_by_run_id=parent_run_id,
        fork_from_node_id=active_node_id,
        metadata={"purpose": "compact", "reason": reason, "agent_definition_id": DEFAULT_COMPACT_AGENT_ID},
    )
    await runtime.store.create_run(run_id=run_id, session_id=session_id, agent_id=agent_id, status=RunStatus.RUNNING.value)
    replay = await runtime.replay_session(session_id)
    selected = [node for node in replay.nodes if not source_node_ids or node.node_id in set(source_node_ids)]
    compact_input = _compact_input(selected)
    await runtime.store.add_event(
        make_event(
            session_id=session_id,
            agent_id=agent_id,
            run_id=run_id,
            event_type="compact_started",
            payload={
                "reason": reason,
                "source_node_ids": source_node_ids or [node.node_id for node in selected],
                "active_node_id": active_node_id,
            },
        )
    )
    input_node = await runtime.store.add_node(
        session_id=session_id,
        parent_id=active_node_id,
        agent_id=agent_id,
        run_id=run_id,
        role="user",
        node_type="compact_input",
        content=[TextBlock(text=compact_input)],
        metadata={"purpose": "compact", "reason": reason, "source_node_ids": source_node_ids or [node.node_id for node in selected]},
        make_active=False,
    )
    model_config = resolve_model_config(runtime.config, runtime.config.compact.model_profile)
    request = ModelRequest(
        model=model_config.name,
        system=[
            SystemBlock(
                block_id=f"agent_definition.{DEFAULT_COMPACT_AGENT_ID}",
                source="agent_definition",
                content=definition.body,
                priority=900,
                dynamic=True,
                metadata={"agent_definition_id": DEFAULT_COMPACT_AGENT_ID},
            )
        ],
        messages=[ModelMessage(role=ModelRole.USER, content=[TextBlock(text=compact_input)], node_type="compact_input")],
        tools=[],
        temperature=model_config.temperature,
        max_output_tokens=min(model_config.max_output_tokens, runtime.config.compact.max_summary_tokens),
        metadata={"session_id": session_id, "run_id": run_id, "purpose": "compact"},
    )
    provider = runtime._provider_for_model(model_config)
    _ensure_provider_supports_request(provider, request)
    delta_parts: list[str] = []
    final_parts: list[str] = []
    error_payload: ErrorPayload | None = None
    async for model_event in provider.stream(request):
        if model_event.event_type == "model_text_delta" and model_event.text_delta:
            delta_parts.append(model_event.text_delta)
        elif model_event.event_type == "model_failed":
            error_payload = model_event.error or ErrorPayload(code=ErrorCode.PROVIDER_ERROR, message="compact provider failed")
            break
        elif model_event.event_type == "model_completed":
            for block in model_event.content:
                if getattr(block, "type", None) == "text":
                    final_parts.append(getattr(block, "text", ""))
            break
    if error_payload is not None:
        await runtime.store.update_run(
            run_id=run_id,
            status=RunStatus.FAILED.value,
            start_node_id=input_node.node_id,
            end_reason="failed",
            error=error_payload.model_dump(mode="json"),
        )
        await runtime.store.add_event(
            make_event(
                session_id=session_id,
                agent_id=agent_id,
                run_id=run_id,
                event_type="compact_failed",
                level="error",
                node_id=input_node.node_id,
                payload=error_payload.model_dump(mode="json"),
            )
        )
        raise AgentCoreError(error_payload.code, error_payload.message, details=error_payload.details)
    summary = ("".join(final_parts) if final_parts else "".join(delta_parts)).strip()
    active_now = await runtime.store.active_node_id(session_id)
    stale = active_now != active_node_id
    compaction_node = None
    if not stale:
        kept_node_id = first_kept_node_id if first_kept_node_id is not None else active_node_id
        compaction_node = await runtime.store.add_node(
            session_id=session_id,
            parent_id=active_node_id,
            agent_id=agent_id,
            run_id=run_id,
            role="assistant",
            node_type="compaction",
            content=[TextBlock(text=summary)],
            metadata={
                "purpose": "compact",
                "first_kept_node_id": kept_node_id,
                "source_node_ids": source_node_ids or [node.node_id for node in selected],
                "details": {"reason": reason, "stale": stale},
            },
            make_active=True,
        )
    await runtime.store.update_run(
        run_id=run_id,
        status=RunStatus.COMPLETED.value,
        start_node_id=input_node.node_id,
        end_node_id=compaction_node.node_id if compaction_node is not None else input_node.node_id,
        end_reason="completed",
    )
    await runtime.store.update_agent(
        agent_id=agent_id,
        status=RunStatus.COMPLETED.value,
        result={
            "compact_run_id": run_id,
            "compaction_node_id": compaction_node.node_id if compaction_node is not None else None,
            "stale": stale,
            "summary_length": len(summary),
        },
    )
    await runtime.store.add_event(
        make_event(
            session_id=session_id,
            agent_id=agent_id,
            run_id=run_id,
            event_type="compact_completed",
            node_id=compaction_node.node_id if compaction_node is not None else input_node.node_id,
            payload={"stale": stale, "summary_length": len(summary), "reason": reason},
        )
    )
    return {
        "compact_run_id": run_id,
        "compact_agent_id": agent_id,
        "summary": summary,
        "stale": stale,
        "compaction_node_id": compaction_node.node_id if compaction_node is not None else None,
    }
