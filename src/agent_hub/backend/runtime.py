from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_core import AgentRuntime
from agent_core.providers import ProviderRegistry
from agent_core.storage import new_id
from agent_core.types import Node, RunMode, RunStatus, RuntimeEvent
from agent_hub.backend.database import HubDatabase
from agent_hub.backend.errors import raise_hub_error
from agent_hub.backend.events import HubEventHub
from agent_hub.backend.models import ConversationView, MessageView
from agent_hub.backend.permissions import PermissionBridge
from agent_hub.backend.services.workers import redact_worker_payload

logger = logging.getLogger(__name__)


@dataclass
class ParsedMention:
    target_type: str
    target_id: str | None
    target_name: str | None
    display_text: str
    directives: dict[str, Any] | None


class HubRuntimeBridge:
    def __init__(
        self,
        *,
        db: HubDatabase,
        events: HubEventHub,
        permission_bridge: PermissionBridge,
        project_dir: Path,
        home_dir: Path | None = None,
        provider_registry: ProviderRegistry | None = None,
    ) -> None:
        self.db = db
        self.events = events
        self.permission_bridge = permission_bridge
        self.project_dir = project_dir
        self.runtime = AgentRuntime(
            project_dir=project_dir,
            home_dir=home_dir,
            permission_callback=permission_bridge.permission_callback,
            provider_registry=provider_registry,
        )
        self._run_tasks: dict[str, asyncio.Task[Any]] = {}

    async def start(self) -> None:
        await self.runtime._ensure_started()

    async def close(self) -> None:
        handles = list(self.runtime._session_active.values())  # type: ignore[attr-defined]
        for queue in list(self.runtime._session_queues.values()):  # type: ignore[attr-defined]
            handles.extend(list(queue))
        for handle in handles:
            await handle.cancel()
        for queue_item in self.runtime.list_worker_queue():
            self.runtime.cancel_worker_queue_item(queue_item.queue_id)
        for task in list(self._run_tasks.values()):
            task.cancel()
        if self._run_tasks:
            await asyncio.gather(*self._run_tasks.values(), return_exceptions=True)
        await self.runtime.close()

    async def create_conversation(self, *, title: str = "New conversation") -> ConversationView:
        core_session_id = new_id("sess")
        conversation = await self.db.create_conversation(core_session_id=core_session_id, title=title)
        assert self.runtime.paths
        await self.runtime._ensure_started()
        await self.runtime.store.ensure_session(  # type: ignore[union-attr]
            session_id=core_session_id,
            cwd=str(self.project_dir),
            root_agent_id=self.runtime._root_agent_id(session_id=core_session_id, mode=RunMode.ORCHESTRATOR),
        )
        self.permission_bridge.bind_session(core_session_id=core_session_id, conversation_id=conversation.conversation_id)
        await self.events.publish("conversation_created", conversation_id=conversation.conversation_id, payload=conversation.model_dump(mode="json"))
        logger.info("conversation created conversation_id=%s core_session_id=%s", conversation.conversation_id, core_session_id)
        return conversation

    async def send_message(self, conversation: ConversationView, text: str) -> tuple[MessageView, str, str]:
        parsed = self._parse_mention(conversation.core_session_id, text)
        title_updates: dict[str, str] = {}
        if conversation.title == "New conversation":
            title_updates["title"] = _title_from_text(parsed.display_text)
        user_message = await self.db.create_message(
            conversation_id=conversation.conversation_id,
            sender_type="user",
            sender_id="user",
            sender_name="You",
            target_type=parsed.target_type,
            target_id=parsed.target_id,
            original_text=text,
            display_text=parsed.display_text,
            status="completed",
            core_session_id=conversation.core_session_id,
            metadata={"target_name": parsed.target_name},
        )
        await self.db.update_conversation(
            conversation.conversation_id,
            last_message_preview=parsed.display_text[:160],
            **title_updates,
        )
        await self.events.publish("message_created", conversation_id=conversation.conversation_id, payload=user_message.model_dump(mode="json"))
        handle = await self.runtime.start(
            parsed.display_text,
            session_id=conversation.core_session_id,
            mode="orchestrator",
            directives=parsed.directives,
        )
        logger.info(
            "run started conversation_id=%s core_session_id=%s run_id=%s status=%s target_type=%s target_id=%s",
            conversation.conversation_id,
            conversation.core_session_id,
            handle.run_id,
            handle.status.value,
            parsed.target_type,
            parsed.target_id,
        )
        user_message = await self.db.update_message(user_message.message_id, core_run_id=handle.run_id) or user_message
        await self.events.publish("message_updated", conversation_id=conversation.conversation_id, payload=user_message.model_dump(mode="json"))
        orchestrator = await self.db.create_message(
            conversation_id=conversation.conversation_id,
            sender_type="orchestrator",
            sender_id=handle.agent_id,
            sender_name="Orchestrator",
            target_type="none",
            original_text="",
            display_text="Dispatching worker..." if parsed.target_type == "worker" else "",
            status="queued" if handle.status == RunStatus.QUEUED else "running",
            core_session_id=conversation.core_session_id,
            core_run_id=handle.run_id,
            metadata={"target_worker_id": parsed.target_id, "target_worker_name": parsed.target_name} if parsed.target_type == "worker" else {},
        )
        await self.events.publish("message_created", conversation_id=conversation.conversation_id, payload=orchestrator.model_dump(mode="json"))
        self._run_tasks[handle.run_id] = asyncio.create_task(
            self._consume_run_events(
                conversation_id=conversation.conversation_id,
                orchestrator_message_id=orchestrator.message_id,
                run_id=handle.run_id,
                handle=handle,
            )
        )
        return user_message, handle.run_id, orchestrator.status

    async def cancel_conversation_run(
        self,
        conversation: ConversationView,
        *,
        core_run_id: str | None = None,
        queue_id: str | None = None,
    ) -> dict[str, Any]:
        if queue_id:
            cancelled = self.runtime.cancel_worker_queue_item(queue_id)
            logger.info(
                "worker queue cancel requested conversation_id=%s queue_id=%s cancelled=%s",
                conversation.conversation_id,
                queue_id,
                cancelled,
            )
            if cancelled:
                await self.events.publish(
                    "worker_cancelled",
                    conversation_id=conversation.conversation_id,
                    payload={"queue_id": queue_id},
                )
            return {"queue_id": queue_id, "cancelled": cancelled}
        if not core_run_id:
            return {"cancelled": False, "reason": "missing core_run_id or queue_id"}
        handle = self._find_run_handle(core_run_id)
        if handle is None:
            return {"core_run_id": core_run_id, "cancelled": False, "reason": "run not active or queued"}
        result = await handle.cancel()
        logger.info(
            "run cancel requested conversation_id=%s core_run_id=%s status=%s",
            conversation.conversation_id,
            core_run_id,
            result.status.value,
        )
        await self.db.update_messages_by_run(core_run_id, status=result.status.value)
        await self.events.publish(
            "run_cancelled",
            conversation_id=conversation.conversation_id,
            payload={"core_run_id": core_run_id, **result.model_dump(mode="json")},
        )
        return result.model_dump(mode="json")

    async def delete_conversation(self, conversation: ConversationView) -> ConversationView:
        for queue_item in self.runtime.list_worker_queue():
            if queue_item.session_id == conversation.core_session_id:
                self.runtime.cancel_worker_queue_item(queue_item.queue_id)
        active_handle = self.runtime._session_active.get(conversation.core_session_id)  # type: ignore[attr-defined]
        if active_handle is not None:
            await active_handle.cancel()
        for queued_handle in list(self.runtime._session_queues.get(conversation.core_session_id, [])):  # type: ignore[attr-defined]
            await queued_handle.cancel()
        deleted = await self.db.soft_delete_conversation(conversation.conversation_id)
        if deleted is None:
            raise_hub_error(404, "conversation_not_found", f"Conversation not found: {conversation.conversation_id}")
        await self.events.publish(
            "conversation_deleted",
            conversation_id=conversation.conversation_id,
            payload=deleted.model_dump(mode="json"),
        )
        logger.info("conversation deleted conversation_id=%s core_session_id=%s", conversation.conversation_id, conversation.core_session_id)
        return deleted

    async def load_skill_for_conversation(self, conversation: ConversationView, skill_name: str) -> dict[str, Any]:
        result = await self.runtime.load_skill(
            conversation.core_session_id,
            skill_name,
            mode="orchestrator",
        )
        if result.error is not None:
            status_code = 404 if result.error.code.value == "skill_not_found" else 409
            raise_hub_error(status_code, result.error.code.value, result.error.message, result.error.details)
        logger.info(
            "skill loaded conversation_id=%s core_session_id=%s skill=%s already_loaded=%s",
            conversation.conversation_id,
            conversation.core_session_id,
            skill_name,
            result.already_loaded,
        )
        return result.model_dump(mode="json")

    def _find_run_handle(self, core_run_id: str):
        active = self.runtime._session_active  # type: ignore[attr-defined]
        for handle in active.values():
            if handle.run_id == core_run_id:
                return handle
        queues = self.runtime._session_queues  # type: ignore[attr-defined]
        for queue in queues.values():
            for handle in queue:
                if handle.run_id == core_run_id:
                    return handle
        return None

    def _parse_mention(self, core_session_id: str, text: str) -> ParsedMention:
        stripped = text.strip()
        if not stripped:
            raise_hub_error(400, "validation_error", "Message text is required.")
        if not stripped.startswith("@"):
            return ParsedMention("orchestrator", "orchestrator", "Orchestrator", text, None)
        mention, _, rest = stripped.partition(" ")
        name = mention[1:]
        display_text = rest.strip()
        if name.lower() == "orchestrator":
            if not display_text:
                raise_hub_error(400, "validation_error", "Message text is required after @Orchestrator.")
            return ParsedMention("orchestrator", "orchestrator", "Orchestrator", display_text, None)
        if not display_text:
            raise_hub_error(400, "validation_error", f"Message text is required after @{name}.")
        resolution = self.runtime.resolve_worker_mention(name, session_id=core_session_id)
        if not resolution.resolved:
            status_code = 409 if resolution.error_code in {"worker_ambiguous", "worker_disabled", "worker_deleted"} else 404
            raise_hub_error(
                status_code,
                resolution.error_code or "worker_not_found",
                resolution.error_message or f"Worker mention could not be resolved: @{name}",
                {"resolution": resolution.model_dump(mode="json")},
            )
        return ParsedMention(
            "worker",
            resolution.worker_id,
            resolution.name or resolution.worker_id,
            display_text,
            {"mentioned_worker": resolution.to_directive().model_dump(mode="json", exclude_none=True)},
        )

    async def _consume_run_events(self, *, conversation_id: str, orchestrator_message_id: str, run_id: str, handle) -> None:
        text_parts: list[str] = []
        try:
            async for event in handle.events():
                await self._map_core_event(
                    conversation_id=conversation_id,
                    orchestrator_message_id=orchestrator_message_id,
                    event=event,
                    text_parts=text_parts,
                )
        finally:
            self._run_tasks.pop(run_id, None)

    async def _map_core_event(
        self,
        *,
        conversation_id: str,
        orchestrator_message_id: str,
        event: RuntimeEvent,
        text_parts: list[str],
    ) -> None:
        if event.event_type in {"assistant_delta", "model_text_delta"}:
            delta = str(event.payload.get("text") or "")
            if delta:
                text_parts.append(delta)
                message = await self.db.update_message(orchestrator_message_id, display_text="".join(text_parts), status="running")
                await self.events.publish(
                    "message_delta",
                    conversation_id=conversation_id,
                    payload={"message_id": orchestrator_message_id, "delta": delta, "message": message.model_dump(mode="json") if message else None},
                )
        elif event.event_type == "run_dequeued":
            logger.info("run dequeued session_id=%s run_id=%s", event.session_id, event.run_id)
            message = await self.db.update_message(orchestrator_message_id, status="running")
            await self.events.publish("run_started", conversation_id=conversation_id, payload=message.model_dump(mode="json") if message else {})
        elif event.event_type == "loop_started":
            logger.info("run loop started session_id=%s run_id=%s", event.session_id, event.run_id)
            message = await self.db.update_message(orchestrator_message_id, status="running")
            await self.events.publish("run_started", conversation_id=conversation_id, payload=message.model_dump(mode="json") if message else {})
        elif event.event_type == "message_created" and event.payload.get("role") == "user":
            user_message = await self.db.latest_message_for_run(
                conversation_id=conversation_id,
                core_run_id=event.run_id,
                sender_type="user",
            )
            if user_message is not None:
                message = await self.db.update_message(user_message.message_id, core_node_id=event.node_id)
                await self.events.publish(
                    "message_updated",
                    conversation_id=conversation_id,
                    payload=message.model_dump(mode="json") if message else {},
                )
        elif event.event_type == "model_completed":
            message = await self._update_orchestrator_from_node(
                message_id=orchestrator_message_id,
                session_id=event.session_id,
                node_id=event.node_id,
                status="running",
                existing_text="".join(text_parts),
            )
            await self.events.publish("message_updated", conversation_id=conversation_id, payload=message.model_dump(mode="json") if message else {})
        elif event.event_type == "worker_run_started":
            logger.info(
                "worker run started session_id=%s parent_run_id=%s child_run_id=%s worker_id=%s task_id=%s",
                event.session_id,
                event.payload.get("parent_run_id") or event.run_id,
                event.payload.get("child_run_id") or event.run_id,
                event.payload.get("worker_id"),
                event.payload.get("task_id"),
            )
            parent_run_id = str(event.payload.get("parent_run_id") or event.run_id or "")
            worker_display = await self._worker_display(event.payload.get("worker_id"))
            queued_message = await self.db.latest_queued_worker_message(
                conversation_id=conversation_id,
                core_run_id=parent_run_id,
                task_id=str(event.payload.get("task_id") or ""),
                worker_id=str(event.payload.get("worker_id") or ""),
            )
            event_type = "message_created"
            if queued_message is None:
                message = await self.db.create_message(
                    conversation_id=conversation_id,
                    sender_type="worker",
                    sender_id=event.payload.get("worker_agent_id"),
                    sender_name=worker_display["sender_name"],
                    target_type="none",
                    original_text="",
                    display_text=str(event.payload.get("task_id") or "Worker task started"),
                    status="running",
                    core_session_id=event.session_id,
                    core_run_id=parent_run_id,
                    core_node_id=event.node_id,
                    child_run_id=str(event.payload.get("child_run_id") or event.run_id or ""),
                    task_id=event.payload.get("task_id"),
                    worker_id=event.payload.get("worker_id"),
                    metadata=worker_display["metadata"],
                )
            else:
                event_type = "message_updated"
                message = await self.db.update_message(
                    queued_message.message_id,
                    display_text=str(event.payload.get("task_id") or "Worker task started"),
                    status="running",
                    core_node_id=event.node_id,
                    child_run_id=str(event.payload.get("child_run_id") or event.run_id or ""),
                )
            await self.events.publish(event_type, conversation_id=conversation_id, payload=message.model_dump(mode="json") if message else {})
            await self.events.publish("worker_started", conversation_id=conversation_id, payload=event.payload | {"message": message.model_dump(mode="json")})
        elif event.event_type == "worker_run_completed":
            logger.info(
                "worker run completed session_id=%s child_run_id=%s worker_id=%s task_id=%s",
                event.session_id,
                event.payload.get("child_run_id") or event.run_id,
                event.payload.get("worker_id"),
                event.payload.get("task_id"),
            )
            worker_id = str(event.payload.get("worker_id") or "")
            child_run_id = str(event.payload.get("child_run_id") or event.run_id or "")
            display_text = _worker_summary_text(event.payload)
            message = await self.db.latest_worker_message(
                conversation_id=conversation_id,
                child_run_id=child_run_id,
                worker_id=worker_id,
            )
            if message is None:
                worker_display = await self._worker_display(worker_id)
                message = await self.db.create_message(
                    conversation_id=conversation_id,
                    sender_type="worker",
                    sender_id=event.payload.get("worker_agent_id"),
                    sender_name=worker_display["sender_name"],
                    target_type="none",
                    original_text="",
                    display_text=display_text,
                    status="completed",
                    core_session_id=event.session_id,
                    core_run_id=str(event.payload.get("parent_run_id") or ""),
                    core_node_id=event.node_id,
                    child_run_id=child_run_id,
                    task_id=event.payload.get("task_id"),
                    worker_id=worker_id,
                    metadata=worker_display["metadata"],
                )
            else:
                message = await self.db.update_message(
                    message.message_id,
                    display_text=display_text,
                    status="completed",
                    core_node_id=event.node_id,
                )
            payload = message.model_dump(mode="json") if message else {}
            await self.events.publish("message_updated", conversation_id=conversation_id, payload=payload)
            await self.events.publish("worker_completed", conversation_id=conversation_id, payload=event.payload | {"message": payload})
        elif event.event_type in {"worker_run_failed", "worker_run_cancelled"}:
            logger.info(
                "%s session_id=%s child_run_id=%s worker_id=%s task_id=%s",
                event.event_type,
                event.session_id,
                event.payload.get("child_run_id") or event.run_id,
                event.payload.get("worker_id"),
                event.payload.get("task_id"),
            )
            child_run_id = str(event.payload.get("child_run_id") or event.run_id or "")
            message = await self.db.latest_worker_message(
                conversation_id=conversation_id,
                child_run_id=child_run_id,
                worker_id=str(event.payload.get("worker_id") or ""),
            )
            status = "cancelled" if event.event_type == "worker_run_cancelled" else "failed"
            text = str(event.payload.get("message") or event.payload.get("task_id") or status)
            if message is not None:
                message = await self.db.update_message(message.message_id, display_text=text, status=status, core_node_id=event.node_id)
                payload = message.model_dump(mode="json") if message else {}
                await self.events.publish("message_updated", conversation_id=conversation_id, payload=payload)
                await self.events.publish(f"worker_{status}", conversation_id=conversation_id, payload=event.payload | {"message": payload})
        elif event.event_type == "run_completed":
            logger.info("run completed session_id=%s run_id=%s", event.session_id, event.run_id)
            message = await self._update_orchestrator_from_node(
                message_id=orchestrator_message_id,
                session_id=event.session_id,
                node_id=event.node_id,
                status="completed",
                existing_text="".join(text_parts),
            )
            await self.events.publish("run_completed", conversation_id=conversation_id, payload=message.model_dump(mode="json") if message else {})
        elif event.event_type == "loop_failed":
            logger.info("run failed session_id=%s run_id=%s", event.session_id, event.run_id)
            message = await self.db.update_message(orchestrator_message_id, status="failed")
            await self.events.publish("run_failed", conversation_id=conversation_id, payload={"message": message.model_dump(mode="json") if message else {}, "error": event.payload})
        elif event.event_type == "run_cancelled":
            logger.info("run cancelled session_id=%s run_id=%s", event.session_id, event.run_id)
            message = await self.db.update_message(orchestrator_message_id, status="cancelled")
            await self.events.publish("run_cancelled", conversation_id=conversation_id, payload=message.model_dump(mode="json") if message else {})
        elif event.event_type == "permission_failed":
            logger.info("permission failed session_id=%s run_id=%s tool=%s", event.session_id, event.run_id, event.payload.get("name"))
            error = event.payload.get("error") if isinstance(event.payload.get("error"), dict) else {}
            message_text = str(error.get("message") or "Permission check failed")
            system_message = await self.db.create_message(
                conversation_id=conversation_id,
                sender_type="system",
                sender_id="permission",
                sender_name="Permission",
                target_type="none",
                original_text="",
                display_text=f"{event.payload.get('name') or 'tool'}: {message_text}",
                status="failed",
                core_session_id=event.session_id,
                core_run_id=event.run_id,
                metadata={"core_event": event.model_dump(mode="json")},
            )
            await self.events.publish("message_created", conversation_id=conversation_id, payload=system_message.model_dump(mode="json"))
            await self.events.publish("permission_failed", conversation_id=conversation_id, payload={"message": system_message.model_dump(mode="json"), "error": event.payload})
        elif event.event_type == "worker_queued":
            logger.info(
                "worker queued session_id=%s parent_run_id=%s worker_id=%s task_id=%s queue_id=%s",
                event.session_id,
                event.payload.get("parent_run_id") or event.run_id,
                event.payload.get("worker_id"),
                event.payload.get("task_id"),
                event.payload.get("queue_id"),
            )
            worker_display = await self._worker_display(event.payload.get("worker_id"))
            message = await self.db.create_message(
                conversation_id=conversation_id,
                sender_type="worker",
                sender_id=event.payload.get("worker_agent_id"),
                sender_name=worker_display["sender_name"],
                target_type="none",
                original_text="",
                display_text=f"Queued task {event.payload.get('task_id') or ''}".strip(),
                status="queued",
                core_session_id=event.session_id,
                core_run_id=str(event.payload.get("parent_run_id") or event.run_id or ""),
                task_id=event.payload.get("task_id"),
                worker_id=event.payload.get("worker_id"),
                queue_id=event.payload.get("queue_id"),
                metadata=worker_display["metadata"],
            )
            await self.events.publish("message_created", conversation_id=conversation_id, payload=message.model_dump(mode="json"))
            await self.events.publish("worker_queued", conversation_id=conversation_id, payload=event.payload | {"message": message.model_dump(mode="json")})

    async def _worker_display(self, worker_id: Any) -> dict[str, Any]:
        normalized = str(worker_id or "")
        if not normalized:
            return {"sender_name": "Worker", "metadata": {}}
        worker = await self.runtime.get_worker_config(normalized)
        if worker is None:
            return {"sender_name": normalized, "metadata": {"worker_id": normalized}}
        worker_json = redact_worker_payload(worker.model_dump(mode="json"))
        snapshot_id = await self.db.snapshot_worker(normalized, worker_json)
        return {
            "sender_name": worker.name or normalized,
            "metadata": {
                "worker_snapshot_id": snapshot_id,
                "worker_snapshot": worker_json,
            },
        }

    async def _update_orchestrator_from_node(
        self,
        *,
        message_id: str,
        session_id: str,
        node_id: str | None,
        status: str,
        existing_text: str,
    ) -> MessageView | None:
        updates: dict[str, Any] = {"status": status}
        if node_id:
            updates["core_node_id"] = node_id
            if not existing_text.strip():
                assert self.runtime.store
                node = await self.runtime.store.get_node(session_id, node_id)
                if node is not None:
                    text = _node_text(node)
                    if text:
                        updates["display_text"] = text
        return await self.db.update_message(message_id, **updates)


def _title_from_text(text: str) -> str:
    stripped = " ".join(text.split())
    return (stripped[:60] or "New conversation")


def _node_text(node: Node) -> str:
    parts: list[str] = []
    for block in node.content:
        text = getattr(block, "text", None)
        if text:
            parts.append(str(text))
    return "\n".join(parts).strip()


def _worker_summary_text(payload: dict[str, Any]) -> str:
    if payload.get("step_result_summary"):
        return str(payload["step_result_summary"])
    summary = payload.get("summary")
    if isinstance(summary, dict):
        for key in ("step_result_summary", "result_summary", "summary"):
            if summary.get(key):
                return str(summary[key])
    task_id = payload.get("task_id")
    return f"Worker completed {task_id}" if task_id else "Worker completed"
