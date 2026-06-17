from __future__ import annotations

import hashlib
import json
from typing import Any

from agent_core.artifacts.redaction import redact_value
from agent_core.providers import ModelMessage
from agent_core.providers.base import ModelRole
from agent_core.tasks.service import TaskService
from agent_core.types import Node, RuntimeEvent, SessionInfo, TextBlock


def _summary_from_step(step: dict[str, Any]) -> dict[str, Any]:
    return {
        "claimed_step_id": step.get("step_id"),
        "step_status": step.get("status"),
        "step_result_summary": step.get("result_summary"),
        "no_step_claimed": False,
    }


def _tool_event_payload(name: str, result: Any) -> dict[str, Any]:
    payload = {"name": name, "is_error": bool(getattr(result, "is_error", False))}
    error = getattr(result, "error", None)
    if error is not None:
        payload["error"] = error.model_dump(mode="json") if hasattr(error, "model_dump") else error
    return payload


def _content_has_text(content: list[Any]) -> bool:
    for block in content:
        if getattr(block, "type", None) == "text" and str(getattr(block, "text", "")).strip():
            return True
    return False


def _last_message_is_tool_result(messages: list[ModelMessage]) -> bool:
    if not messages:
        return False
    return messages[-1].role == ModelRole.TOOL


def _model_request_views_from_events(events: list[RuntimeEvent], *, run_id: str | None) -> list[dict[str, Any]]:
    views: list[dict[str, Any]] = []
    for event in events:
        if event.event_type != "context_built":
            continue
        if run_id is not None and event.run_id != run_id:
            continue
        payload = dict(event.payload)
        system_blocks = list(payload.get("system_blocks") or [])
        views.append(
            {
                "run_id": event.run_id,
                "agent_id": event.agent_id,
                "event_id": event.event_id,
                "seq": event.seq,
                "run_seq": event.run_seq,
                "model": payload.get("model"),
                "message_count": payload.get("messages"),
                "tool_count": payload.get("tools"),
                "tool_names": list(payload.get("tool_names") or []),
                "system_blocks": system_blocks,
                "retained_node_ids": list(payload.get("retained_node_ids") or []),
                "trimmed_node_ids": list(payload.get("trimmed_node_ids") or []),
                "synthetic_messages": list(payload.get("synthetic_messages") or []),
                "estimated_input_tokens": payload.get("estimated_input_tokens"),
                "too_long": bool(payload.get("too_long", False)),
            }
        )
    return views


def _session_info_from_row(row: dict[str, Any]) -> SessionInfo:
    try:
        metadata = json.loads(row.get("metadata_json") or "{}")
    except json.JSONDecodeError:
        metadata = {}
    return SessionInfo(
        session_id=str(row.get("session_id") or ""),
        cwd=str(row.get("cwd") or ""),
        root_agent_id=str(row.get("root_agent_id") or ""),
        active_node_id=row.get("active_node_id"),
        parent_session_id=row.get("parent_session_id"),
        status=str(row.get("status") or ""),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
        metadata=metadata,
    )


def _node_content_preview(node: Node, limit: int = 120) -> str:
    parts: list[str] = []
    for block in node.content:
        text = getattr(block, "text", None)
        if text:
            parts.append(str(text))
            continue
        data = getattr(block, "data", None)
        if data is not None:
            parts.append(json.dumps(data, ensure_ascii=False))
    preview = " ".join(part.strip() for part in parts if part.strip())
    preview = " ".join(preview.split())
    if len(preview) > limit:
        return preview[: max(limit - 3, 0)] + "..."
    return preview


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _synthetic_context_nodes_from_tool_results(tool_results: list[Any]) -> list[dict[str, Any]]:
    synthetic: list[dict[str, Any]] = []
    allowed_node_types = {"plan_instruction", "task_instruction", "skill_context", "memory_context"}
    for result in tool_results:
        if getattr(result, "is_error", False):
            continue
        for block in getattr(result, "content", []) or []:
            if getattr(block, "type", None) != "json":
                continue
            data = getattr(block, "data", None)
            if not isinstance(data, dict):
                continue
            node_type = data.get("node_type")
            if node_type not in allowed_node_types:
                continue
            if data.get("already_loaded") is True or data.get("already_recalled") is True:
                continue
            content = data.get("content")
            if not isinstance(content, str) or not content:
                continue
            text = content if node_type in {"skill_context", "memory_context"} else _synthetic_context_text(node_type=str(node_type), data=data)
            metadata = {
                "synthetic": True,
                "source": getattr(result, "tool_name", None),
                "tool_call_id": getattr(result, "tool_call_id", None),
            }
            if isinstance(data.get("metadata"), dict):
                metadata.update(data["metadata"])
            for key in ("goal", "suggested_dir", "name", "path", "hash", "query", "template_id", "template_version"):
                if data.get(key) is not None:
                    metadata[key] = data[key]
            synthetic.append({"node_type": str(node_type), "text": text, "metadata": metadata})
    return synthetic


def _synthetic_context_text(*, node_type: str, data: dict[str, Any]) -> str:
    tag = node_type.replace("_", "-")
    lines = [f"<{tag}>"]
    if data.get("goal"):
        lines.extend(["Goal:", str(data["goal"]), ""])
    if data.get("suggested_dir"):
        lines.extend(["Suggested directory:", str(data["suggested_dir"]), ""])
    lines.append(str(data["content"]))
    lines.append(f"</{tag}>")
    return "\n".join(lines)


def _task_board_context_message(task_service: TaskService, session_id: str) -> ModelMessage | None:
    summaries = task_service.active_task_summaries(session_id)
    if not summaries:
        return None
    lines = ["<task_board>"]
    for task in summaries:
        lines.append(f"Task {task['task_id']} [{task['status']}]: {task.get('title') or ''}")
        if task.get("summary"):
            lines.append(str(task["summary"]))
        for step in task.get("steps") or []:
            lines.append(
                "- "
                f"{step['step_id']} [{step['status']}] "
                f"{step.get('title') or ''}; "
                f"worker_pool={step.get('worker_pool_id') or ''}; "
                f"claimed_by={step.get('claimed_by_agent_id') or ''}; "
                f"lease_expires_at={step.get('lease_expires_at') or ''}; "
                f"result={step.get('result_summary') or ''}; "
                f"artifacts={','.join(step.get('artifact_ids') or [])}"
            )
        lines.append("")
    lines.append("</task_board>")
    return ModelMessage(
        role=ModelRole.USER,
        content=[TextBlock(text="\n".join(lines))],
        node_type="task_board",
        metadata={"synthetic": True, "source": "task_board", "task_count": len(summaries)},
    )


def _redact_replay_payload(
    nodes: list[Node],
    events: list[RuntimeEvent],
    artifacts: list[dict[str, Any]],
) -> tuple[list[Node], list[RuntimeEvent], list[dict[str, Any]]]:
    redacted_nodes = [
        node.model_copy(
            update={
                "content": redact_value(node.content),
                "metadata": redact_value(node.metadata),
            }
        )
        for node in nodes
    ]
    redacted_events = [
        event.model_copy(update={"payload": redact_value(event.payload)})
        for event in events
    ]
    redacted_artifacts = [redact_value(artifact) for artifact in artifacts]
    return redacted_nodes, redacted_events, redacted_artifacts
