from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1

from agent_core.config.models import AgentsConfig
from agent_core.errors import AgentCoreError
from agent_core.errors.codes import ErrorCode


@dataclass
class WorkerRuntimeState:
    worker_id: str
    pool_id: str
    agent_definition_id: str
    allowed_tools: list[str] | None = None
    status: str = "idle"
    current_task_id: str | None = None
    current_run_id: str | None = None
    current_step_id: str | None = None


class WorkerPoolRuntime:
    def __init__(self, config: AgentsConfig) -> None:
        self._workers: dict[str, WorkerRuntimeState] = {}
        self._by_pool: dict[str, list[str]] = {}
        self.configure(config)

    def configure(self, config: AgentsConfig) -> None:
        workers: dict[str, WorkerRuntimeState] = {}
        by_pool: dict[str, list[str]] = {}
        for pool in config.worker_pools:
            ids: list[str] = []
            for index, worker in enumerate(pool.workers):
                worker_id = worker.worker_id or f"{pool.pool_id}_{worker.agent_definition_id}_{index}"
                if worker_id in workers:
                    raise AgentCoreError(ErrorCode.CONFIG_ERROR, f"duplicate worker_id: {worker_id}")
                existing = self._workers.get(worker_id)
                if existing is not None:
                    existing.pool_id = pool.pool_id
                    existing.agent_definition_id = worker.agent_definition_id
                    existing.allowed_tools = worker.allowed_tools
                    workers[worker_id] = existing
                else:
                    workers[worker_id] = WorkerRuntimeState(
                        worker_id=worker_id,
                        pool_id=pool.pool_id,
                        agent_definition_id=worker.agent_definition_id,
                        allowed_tools=worker.allowed_tools,
                    )
                ids.append(worker_id)
            by_pool[pool.pool_id] = ids
        self._workers = workers
        self._by_pool = by_pool

    def list_workers(self, worker_pool_id: str | None = None) -> list[WorkerRuntimeState]:
        if worker_pool_id:
            return [self._workers[worker_id] for worker_id in self._by_pool.get(worker_pool_id, [])]
        return [self._workers[key] for key in sorted(self._workers)]

    def select_worker(
        self,
        *,
        worker_pool_id: str | None = None,
        worker_agent_id: str | None = None,
        session_id: str | None = None,
    ) -> WorkerRuntimeState:
        if worker_agent_id:
            worker = self._workers.get(worker_agent_id) or self._worker_by_agent_id(worker_agent_id, session_id=session_id)
            if worker is None:
                raise AgentCoreError(ErrorCode.WORKER_NOT_AVAILABLE, f"worker not available: {worker_agent_id}")
            if worker_pool_id is not None and worker.pool_id != worker_pool_id:
                raise AgentCoreError(ErrorCode.WORKER_NOT_AVAILABLE, f"worker {worker_agent_id} is not in pool {worker_pool_id}")
            if worker.status != "idle":
                raise AgentCoreError(ErrorCode.WORKER_BUSY, f"worker busy: {worker_agent_id}")
            return worker
        pools = [worker_pool_id] if worker_pool_id else list(self._by_pool)
        for pool in pools:
            for worker_id in self._by_pool.get(pool or "", []):
                worker = self._workers[worker_id]
                if worker.status == "idle":
                    return worker
        raise AgentCoreError(ErrorCode.WORKER_POOL_BUSY, "no idle worker available")

    def mark_busy(self, worker: WorkerRuntimeState, *, task_id: str, run_id: str | None = None, step_id: str | None = None) -> None:
        worker.status = "running"
        worker.current_task_id = task_id
        worker.current_run_id = run_id
        worker.current_step_id = step_id

    def mark_idle(self, worker: WorkerRuntimeState) -> None:
        worker.status = "idle"
        worker.current_task_id = None
        worker.current_run_id = None
        worker.current_step_id = None

    def _worker_by_agent_id(self, worker_agent_id: str, *, session_id: str | None) -> WorkerRuntimeState | None:
        if session_id is None:
            return None
        for worker in self._workers.values():
            if worker_agent_id_for_session(session_id=session_id, worker_id=worker.worker_id) == worker_agent_id:
                return worker
        return None


def worker_agent_id_for_session(*, session_id: str, worker_id: str) -> str:
    digest = sha1(f"{session_id}:{worker_id}".encode("utf-8")).hexdigest()[:16]
    return f"agent_worker_{digest}"
