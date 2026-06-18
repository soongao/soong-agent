from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request

from agent_core.types import WorkerConfigCreate, WorkerConfigUpdate
from agent_hub.backend.errors import raise_hub_error
from agent_hub.backend.models import WorkerListResponse
from agent_hub.backend.services.workers import cancel_worker_queue_item as cancel_worker_queue_item_service
from agent_hub.backend.services.workers import redact_worker_payload, restore_redacted_api_key, worker_runtime_status
from agent_hub.backend.state import HubAppState, require_runtime_bridge

router = APIRouter(prefix="/workers")
logger = logging.getLogger(__name__)


@router.get("")
async def list_workers(request: Request, include_disabled: bool = True, include_deleted: bool = False) -> WorkerListResponse:
    state: HubAppState = request.app.state.hub
    runtime = require_runtime_bridge(state).runtime
    workers = await runtime.list_worker_configs(include_disabled=include_disabled, include_deleted=include_deleted)
    return WorkerListResponse(
        workers=[
            redact_worker_payload(worker.model_dump(mode="json") | worker_runtime_status(runtime, worker))
            for worker in workers
        ]
    )


@router.get("/{worker_id}")
async def get_worker(worker_id: str, request: Request) -> dict[str, Any]:
    state: HubAppState = request.app.state.hub
    runtime = require_runtime_bridge(state).runtime
    worker = await runtime.get_worker_config(worker_id)
    if worker is None or worker.deleted_at is not None:
        raise_hub_error(404, "worker_not_found", f"Worker not found: {worker_id}")
    return redact_worker_payload(worker.model_dump(mode="json") | worker_runtime_status(runtime, worker))


@router.post("")
async def create_worker(payload: WorkerConfigCreate, request: Request) -> dict[str, Any]:
    state: HubAppState = request.app.state.hub
    runtime = require_runtime_bridge(state).runtime
    worker = await runtime.create_worker_config(payload)
    logger.info("worker created worker_id=%s source=%s enabled=%s", worker.worker_id, worker.source, worker.enabled)
    return redact_worker_payload(worker.model_dump(mode="json"))


@router.patch("/{worker_id}")
async def update_worker(worker_id: str, payload: WorkerConfigUpdate, request: Request) -> dict[str, Any]:
    state: HubAppState = request.app.state.hub
    runtime = require_runtime_bridge(state).runtime
    payload = await restore_redacted_api_key(runtime, worker_id, payload)
    worker = await runtime.update_worker_config(worker_id, payload)
    logger.info("worker updated worker_id=%s enabled=%s", worker.worker_id, worker.enabled)
    return redact_worker_payload(worker.model_dump(mode="json"))


@router.post("/{worker_id}/enable")
async def enable_worker(worker_id: str, request: Request) -> dict[str, Any]:
    state: HubAppState = request.app.state.hub
    runtime = require_runtime_bridge(state).runtime
    worker = await runtime.enable_worker_config(worker_id)
    logger.info("worker enabled worker_id=%s", worker.worker_id)
    return redact_worker_payload(worker.model_dump(mode="json"))


@router.post("/{worker_id}/disable")
async def disable_worker(worker_id: str, request: Request) -> dict[str, Any]:
    state: HubAppState = request.app.state.hub
    runtime = require_runtime_bridge(state).runtime
    worker = await runtime.disable_worker_config(worker_id)
    logger.info("worker disabled worker_id=%s", worker.worker_id)
    return redact_worker_payload(worker.model_dump(mode="json"))


@router.delete("/{worker_id}")
async def delete_worker(worker_id: str, request: Request) -> dict[str, Any]:
    state: HubAppState = request.app.state.hub
    runtime = require_runtime_bridge(state).runtime
    worker = await runtime.soft_delete_worker_config(worker_id)
    logger.info("worker soft deleted worker_id=%s", worker.worker_id)
    return redact_worker_payload(worker.model_dump(mode="json"))


@router.get("/{worker_id}/queue")
async def worker_queue(worker_id: str, request: Request) -> dict[str, Any]:
    state: HubAppState = request.app.state.hub
    runtime = require_runtime_bridge(state).runtime
    return {"queue": [item.model_dump(mode="json") for item in runtime.list_worker_queue(worker_id)]}


@router.post("/{worker_id}/queue/{queue_id}/cancel")
async def cancel_worker_queue_item(worker_id: str, queue_id: str, request: Request) -> dict[str, Any]:
    state: HubAppState = request.app.state.hub
    logger.info("worker queue cancelled worker_id=%s queue_id=%s", worker_id, queue_id)
    require_runtime_bridge(state)
    return await cancel_worker_queue_item_service(state, worker_id=worker_id, queue_id=queue_id)
