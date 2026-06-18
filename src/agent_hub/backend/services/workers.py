from __future__ import annotations

from typing import Any

from agent_core.types import WorkerConfigUpdate


def worker_runtime_status(runtime, worker: Any) -> dict[str, Any]:
    runtime_state = None
    if runtime.worker_runtime is not None:
        runtime_states = {worker.worker_id: worker for worker in runtime.worker_runtime.list_workers()}
        runtime_state = runtime_states.get(worker.worker_id)
    status = runtime_state.status if runtime_state is not None else "disabled"
    if worker.deleted_at is not None:
        status = "deleted"
    elif not worker.enabled:
        status = "disabled"
    return {
        "status": status,
        "queue_length": len(runtime.list_worker_queue(worker.worker_id)),
        "current_task_id": runtime_state.current_task_id if runtime_state is not None else None,
        "current_run_id": runtime_state.current_run_id if runtime_state is not None else None,
        "current_step_id": runtime_state.current_step_id if runtime_state is not None else None,
    }


def redact_worker_payload(worker: dict[str, Any]) -> dict[str, Any]:
    redacted = dict(worker)
    model = redacted.get("model")
    if isinstance(model, dict) and model.get("api_key"):
        redacted["model"] = {**model, "api_key": "***"}
    return redacted


async def restore_redacted_api_key(runtime, worker_id: str, payload: WorkerConfigUpdate) -> WorkerConfigUpdate:
    if payload.model is None or payload.model.api_key != "***":
        return payload
    existing = await runtime.get_worker_config(worker_id)
    existing_api_key = None
    if existing is not None and isinstance(existing.model, dict):
        existing_api_key = existing.model.get("api_key")
    model = payload.model.model_copy(update={"api_key": existing_api_key})
    return payload.model_copy(update={"model": model})


async def cancel_worker_queue_item(state, *, worker_id: str, queue_id: str) -> dict[str, Any]:
    runtime = state.runtime_bridge.runtime
    queue_items = runtime.list_worker_queue(worker_id)
    item = next((queue_item for queue_item in queue_items if queue_item.queue_id == queue_id), None)
    cancelled = runtime.cancel_worker_queue_item(queue_id)
    if not cancelled:
        from agent_hub.backend.errors import raise_hub_error

        raise_hub_error(404, "run_not_found", f"Worker queue item not found: {queue_id}")
    conversation_id = None
    if item is not None:
        conversation = await state.db.get_conversation_by_core_session(item.session_id)
        conversation_id = conversation.conversation_id if conversation is not None else None
        if conversation_id is not None:
            messages = await state.db.update_messages_by_queue_id(queue_id, status="cancelled")
            for message in messages:
                await state.event_hub.publish("message_updated", conversation_id=conversation_id, payload=message.model_dump(mode="json"))
    await state.event_hub.publish(
        "worker_cancelled",
        conversation_id=conversation_id,
        payload={"worker_id": worker_id, "queue_id": queue_id},
    )
    return {"worker_id": worker_id, "queue_id": queue_id, "cancelled": True}

