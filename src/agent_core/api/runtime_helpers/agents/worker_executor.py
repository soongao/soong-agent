from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from agent_core.agents.workers import WorkerRuntimeState
from agent_core.events import EventStream
from agent_core.types import AgentDefinition, Node


@dataclass(frozen=True)
class WorkerExecutorConfig:
    type: str
    config: dict[str, Any]


@dataclass(frozen=True)
class WorkerExecutorContext:
    session_id: str
    parent_run_id: str
    parent_agent_id: str
    task_id: str
    instruction: str
    worker: WorkerRuntimeState
    worker_agent_id: str
    worker_run_id: str
    worker_start_node: Node
    worker_stream: EventStream
    dispatch_context: str | None = None
    constraints: dict[str, Any] | None = None
    allowed_step_ids: list[str] | None = None
    allowed_tools: list[str] | None = None
    expected_output_schema: dict[str, Any] | None = None
    timeout_ms: int | None = None
    executor_config: dict[str, Any] | None = None
    agent_definition: AgentDefinition | None = None
    parent_handle: Any | None = None


@dataclass(frozen=True)
class WorkerExecutorResult:
    text: str = ""
    data: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None

    def as_worker_result(self) -> dict[str, Any]:
        result = dict(self.data or {})
        result.setdefault("text", self.text)
        if self.metadata:
            result["metadata"] = dict(self.metadata)
        return result


class WorkerExecutor(Protocol):
    async def run(self, runtime: Any, context: WorkerExecutorContext) -> WorkerExecutorResult:
        ...


def worker_executor_config(metadata: dict[str, Any] | None) -> WorkerExecutorConfig | None:
    raw = (metadata or {}).get("worker_executor")
    if not isinstance(raw, dict):
        return None
    executor_type = raw.get("type")
    if not isinstance(executor_type, str) or not executor_type.strip():
        return None
    raw_config = raw.get("config")
    config = dict(raw_config) if isinstance(raw_config, dict) else {}
    return WorkerExecutorConfig(type=executor_type.strip(), config=config)
