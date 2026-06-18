from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from agent_core.agents.dynamic import (
    RunDirectives,
    WorkerConfigCreate,
    WorkerConfigUpdate,
    WorkerConfigView,
    WorkerMentionResolution,
    WorkerQueueItem,
    agent_definition_from_worker_config,
    agents_config_from_worker_views,
    config_worker_views,
    load_json_worker_configs,
    merge_worker_views,
    worker_view_from_create,
)
from agent_core.agents.registry import AgentDefinitionRegistry
from agent_core.agents.child import ChildAgentManager
from agent_core.agents.workers import WorkerPoolRuntime, worker_agent_id_for_session
from agent_core.api.handles import RunHandle
from agent_core.api.runtime_helpers import compaction_runtime
from agent_core.api.runtime_helpers import cleanup as runtime_cleanup
from agent_core.api.runtime_helpers import events as runtime_events
from agent_core.api.runtime_helpers import hooks_runtime
from agent_core.api.runtime_helpers import main_loop as runtime_main_loop
from agent_core.api.runtime_helpers import mcp_runtime
from agent_core.api.runtime_helpers import memory_runtime as runtime_memory
from agent_core.api.runtime_helpers import replay as runtime_replay
from agent_core.api.runtime_helpers import run_control
from agent_core.api.runtime_helpers import sessions as runtime_sessions
from agent_core.api.runtime_helpers import skills as runtime_skills
from agent_core.api.runtime_helpers.agents import child as runtime_child
from agent_core.api.runtime_helpers.agents import compact as runtime_compact
from agent_core.api.runtime_helpers.agents import tools as runtime_agent_tools
from agent_core.api.runtime_helpers.agents import worker as runtime_worker
from agent_core.api.runtime_helpers import artifacts as runtime_artifacts
from agent_core.api.runtime_helpers.tools import execute_tool_calls
from agent_core.api.runtime_helpers.views import (
    _synthetic_context_nodes_from_tool_results,
)
from agent_core.artifacts import ArtifactManager
from agent_core.config.loader import load_runtime_config, write_required_project_dirs
from agent_core.config.models import AgentCoreConfig, ContextConfig
from agent_core.config.paths import ResolvedPaths
from agent_core.context.state import RuntimeContextState
from agent_core.errors import AgentCoreError, ConfigError
from agent_core.errors.codes import ErrorCode
from agent_core.events import EventStream, make_event
from agent_core.hooks.loader import load_hooks, normalize_hooks
from agent_core.mcp.config import load_mcp_config
from agent_core.mcp.discovery import McpToolManager
from agent_core.memory import resolve_memory_dir
from agent_core.providers import ModelMessage, ProviderAdapter, ProviderRegistry, SystemBlock, default_provider_registry
from agent_core.storage import SQLiteStore, new_id
from agent_core.tools.builtin_code import register_builtin_code_tools
from agent_core.tools.agent_tools import register_agent_tools
from agent_core.tools.declarative import load_declarative_tools
from agent_core.tools.execution import ToolExecutionContext
from agent_core.tools.internal import register_internal_tools
from agent_core.tools.registry import ToolRegistry
from agent_core.tasks.service import TaskService
from agent_core.tasks.tools import register_task_tools
from agent_core.types import (
    AgentDefinition,
    BranchableNodeView,
    CancelResult,
    CleanupResult,
    DeleteSessionResult,
    ErrorPayload,
    ForkSessionResult,
    InspectResult,
    LoadSkillResult,
    Node,
    PermissionDecision,
    PermissionRequest,
    ReplayResult,
    RunMode,
    RunStatus,
    RuntimeEvent,
    SessionInfo,
    SessionNodeInfo,
    SkillInfo,
    TextBlock,
    ToolCall,
    ToolDefinition,
    SwitchNodeResult,
    UserMessage,
)

PermissionCallback = Callable[[PermissionRequest], Awaitable[PermissionDecision]]


def _validate_worker_create(request: WorkerConfigCreate | dict[str, Any]) -> WorkerConfigCreate:
    if isinstance(request, WorkerConfigCreate):
        return request
    try:
        return WorkerConfigCreate.model_validate(request)
    except ValidationError as exc:
        raise AgentCoreError(ErrorCode.VALIDATION_ERROR, "invalid worker config", details={"errors": exc.errors()}) from exc


def _validate_worker_update(request: WorkerConfigUpdate | dict[str, Any]) -> WorkerConfigUpdate:
    if isinstance(request, WorkerConfigUpdate):
        return request
    try:
        return WorkerConfigUpdate.model_validate(request)
    except ValidationError as exc:
        raise AgentCoreError(ErrorCode.VALIDATION_ERROR, "invalid worker config update", details={"errors": exc.errors()}) from exc


class AgentRuntime:
    def __init__(
        self,
        project_dir: str | Path | None = None,
        config_path: str | Path | None = None,
        home_dir: str | Path | None = None,
        session_db_path: str | Path | None = None,
        permission_callback: PermissionCallback | None = None,
        provider_registry: ProviderRegistry | None = None,
        tool_registry: ToolRegistry | None = None,
        debug: bool = False,
    ) -> None:
        self._project_dir_arg = project_dir
        self._config_path_arg = config_path
        self._home_dir_arg = home_dir
        self._session_db_path_arg = session_db_path
        self.permission_callback = permission_callback
        self.debug = debug
        self.provider_registry = provider_registry or default_provider_registry()
        self.tool_registry = tool_registry or ToolRegistry()
        register_builtin_code_tools(self.tool_registry)
        self.agent_definitions = AgentDefinitionRegistry()
        self.task_service = TaskService()
        self.worker_runtime: WorkerPoolRuntime | None = None
        self._code_agent_definitions: dict[str, AgentDefinition] = {}
        self._effective_worker_configs: list[WorkerConfigView] = []
        self.context_state = RuntimeContextState()
        self._session_context_states: dict[str, RuntimeContextState] = defaultdict(RuntimeContextState)
        register_internal_tools(self.tool_registry)
        register_task_tools(self.tool_registry, self.task_service)
        self.config: AgentCoreConfig | None = None
        self.paths: ResolvedPaths | None = None
        self.store: SQLiteStore | None = None
        self.artifacts: ArtifactManager | None = None
        self._provider: ProviderAdapter | None = None
        self._provider_cache: dict[str, ProviderAdapter] = {}
        self._mcp_manager: McpToolManager | None = None
        self._mcp_discovered = False
        self._hooks: list[dict[str, Any]] = []
        self._closed = False
        self._started = False
        self._session_active: dict[str, RunHandle] = {}
        self._session_queues: dict[str, deque[RunHandle]] = defaultdict(deque)
        self._worker_queues: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
        self._worker_queue_limit = 20
        self._worker_run_tasks: dict[str, asyncio.Task[Any]] = {}
        self._worker_run_meta: dict[str, dict[str, Any]] = {}
        self._child_managers: dict[str, ChildAgentManager] = {}
        self._child_run_streams: dict[str, EventStream] = {}
        self._session_child_counts: dict[str, int] = defaultdict(int)
        self._memory_idle_tasks: dict[str, asyncio.Task[None]] = {}
        from agent_core.permissions import PermissionSessionCache

        self._permission_caches: dict[str, PermissionSessionCache] = defaultdict(PermissionSessionCache)

    def _context_state_for_session(self, session_id: str) -> RuntimeContextState:
        return self._session_context_states[session_id]

    async def __aenter__(self) -> "AgentRuntime":
        await self._ensure_started()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for task in list(self._memory_idle_tasks.values()):
            task.cancel()
        if self._memory_idle_tasks:
            await asyncio.gather(*self._memory_idle_tasks.values(), return_exceptions=True)
            self._memory_idle_tasks.clear()
        if self._provider is not None:
            await self._provider.close()
        for provider in list(self._provider_cache.values()):
            if provider is self._provider:
                continue
            await provider.close()
        if self._mcp_manager is not None:
            await self._mcp_manager.close()
        if self.store is not None:
            await self.store.close()

    def register_provider(self, key: str, factory: Any) -> None:
        self.provider_registry.register(key, factory)

    def register_tool(self, definition: ToolDefinition, handler: Any) -> None:
        self.tool_registry.register_tool(definition, handler)

    def register_agent_definition(self, definition: AgentDefinition, source: Literal["code"] = "code") -> None:
        missing = [name for name in definition.suggested_tools if self.tool_registry.get(name) is None]
        if missing:
            raise AgentCoreError(
                ErrorCode.INVALID_AGENT_DEFINITION,
                f"agent definition {definition.agent_definition_id} references unknown suggested_tools: {missing}",
            )
        stored = definition.model_copy(update={"source": source})
        self.agent_definitions.register(stored, source=source)
        if source == "code":
            self._code_agent_definitions[stored.agent_definition_id] = stored

    async def run_child_agent(
        self,
        *,
        session_id: str,
        parent_run_id: str,
        parent_agent_id: str,
        agent_definition_id: str,
        task: str,
        mode: Literal["sub", "fork"] = "sub",
        constraints: dict[str, Any] | None = None,
        allowed_tools: list[str] | None = None,
        expected_output_schema: dict[str, Any] | None = None,
        timeout_ms: int | None = None,
        parent_handle: RunHandle | None = None,
    ) -> dict[str, Any]:
        return await runtime_child.run_child_agent(
            self,
            session_id=session_id,
            parent_run_id=parent_run_id,
            parent_agent_id=parent_agent_id,
            agent_definition_id=agent_definition_id,
            task=task,
            mode=mode,
            constraints=constraints,
            allowed_tools=allowed_tools,
            expected_output_schema=expected_output_schema,
            timeout_ms=timeout_ms,
            parent_handle=parent_handle,
        )

    async def run_worker_agent(
        self,
        *,
        session_id: str,
        parent_run_id: str,
        parent_agent_id: str,
        task_id: str,
        instruction: str,
        worker_pool_id: str | None = None,
        worker_agent_id: str | None = None,
        allowed_step_ids: list[str] | None = None,
        dispatch_context: str | None = None,
        constraints: dict[str, Any] | None = None,
        allowed_tools: list[str] | None = None,
        expected_output_schema: dict[str, Any] | None = None,
        timeout_ms: int | None = None,
        parent_handle: RunHandle | None = None,
    ) -> dict[str, Any]:
        return await runtime_worker.run_worker_agent(
            self,
            session_id=session_id,
            parent_run_id=parent_run_id,
            parent_agent_id=parent_agent_id,
            task_id=task_id,
            instruction=instruction,
            worker_pool_id=worker_pool_id,
            worker_agent_id=worker_agent_id,
            allowed_step_ids=allowed_step_ids,
            dispatch_context=dispatch_context,
            constraints=constraints,
            allowed_tools=allowed_tools,
            expected_output_schema=expected_output_schema,
            timeout_ms=timeout_ms,
            parent_handle=parent_handle,
        )

    async def run_compact_agent(
        self,
        *,
        session_id: str,
        source_node_ids: list[str] | None = None,
        reason: str = "manual",
        first_kept_node_id: str | None = None,
    ) -> dict[str, Any]:
        return await runtime_compact.run_compact_agent(
            self,
            session_id=session_id,
            source_node_ids=source_node_ids,
            reason=reason,
            first_kept_node_id=first_kept_node_id,
        )

    async def select_memory(
        self,
        *,
        session_id: str,
        query: str,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        return await runtime_memory.select_memory(self, session_id=session_id, query=query, top_k=top_k)

    async def cancel_worker_runs(
        self,
        *,
        session_id: str,
        task_id: str,
        worker_run_ids: list[str] | None = None,
        reason: str = "task_terminated",
    ) -> dict[str, Any]:
        return await run_control.cancel_worker_runs(
            self,
            session_id=session_id,
            task_id=task_id,
            worker_run_ids=worker_run_ids,
            reason=reason,
        )

    def list_worker_queue(self, worker_id: str | None = None) -> list[WorkerQueueItem]:
        queues = (
            [(worker_id, self._worker_queues.get(worker_id, deque()))]
            if worker_id is not None
            else sorted(self._worker_queues.items(), key=lambda item: item[0])
        )
        items: list[WorkerQueueItem] = []
        for current_worker_id, queue in queues:
            for index, item in enumerate(queue, start=1):
                items.append(
                    WorkerQueueItem(
                        queue_id=item["queue_id"],
                        worker_id=current_worker_id,
                        worker_agent_id=item["worker_agent_id"],
                        session_id=item["session_id"],
                        parent_run_id=item["parent_run_id"],
                        parent_agent_id=item["parent_agent_id"],
                        task_id=item["task_id"],
                        status=item["status"],
                        position=index,
                        created_at=item["created_at"],
                        updated_at=item["updated_at"],
                        cancelled=item.get("cancelled", False),
                    )
                )
        return items

    def cancel_worker_queue_item(self, queue_id: str) -> bool:
        for queue in self._worker_queues.values():
            for item in list(queue):
                if item["queue_id"] != queue_id:
                    continue
                item["status"] = "cancelled"
                item["cancelled"] = True
                from agent_core.types.common import utc_iso

                item["updated_at"] = utc_iso()
                future = item.get("future")
                if future is not None and not future.done():
                    future.set_exception(AgentCoreError(ErrorCode.CANCELLED, "worker queue item cancelled"))
                try:
                    queue.remove(item)
                except ValueError:
                    pass
                return True
        return False

    def _effective_tools(self, *, agent_role: str) -> list[ToolDefinition]:
        tools = [self._apply_tool_config(definition) for definition in self.tool_registry.list_definitions()]
        if self.config is not None and self.config.tools.disabled:
            disabled = set(self.config.tools.disabled)
            tools = [tool for tool in tools if tool.name not in disabled]
        if agent_role == "main":
            return [
                tool
                for tool in tools
                if not tool.name.startswith("agent.task")
                and tool.name not in {"agent.list_workers", "agent.dispatch_worker"}
            ]
        if agent_role == "orchestrator":
            return [tool for tool in tools if tool.name != "agent.fork_agent"]
        if agent_role in {"sub", "fork"}:
            return [
                tool
                for tool in tools
                if not tool.name.startswith("agent.")
                and not tool.name.startswith("internal.")
            ]
        if agent_role == "worker":
            allowed = {
                "agent.task_get",
                "agent.task_query_steps",
                "agent.task_claim_step",
                "agent.task_update_step",
                "code.read_file",
                "code.list_dir",
                "code.search",
            }
            return [tool for tool in tools if tool.name in allowed]
        if agent_role == "compact":
            return []
        return tools

    @staticmethod
    def _root_agent_id(*, session_id: str, mode: RunMode) -> str:
        prefix = "agent_orchestrator" if mode == RunMode.ORCHESTRATOR else "agent_main"
        return f"{prefix}_{session_id}"

    @staticmethod
    def _worker_agent_id(*, session_id: str, worker_id: str) -> str:
        return worker_agent_id_for_session(session_id=session_id, worker_id=worker_id)

    def _apply_tool_config(self, definition: ToolDefinition) -> ToolDefinition:
        if self.config is None:
            return definition
        override_config = self.config.tools.overrides.get(definition.name)
        override = override_config.model_dump(exclude_none=True) if override_config is not None else {}
        if not override:
            return definition
        tags = set(definition.tags)
        if override.get("tags") is not None:
            tags.update(str(tag) for tag in override.get("tags") or [])
        updates: dict[str, Any] = {"tags": tags}
        if override.get("description") is not None:
            updates["description"] = str(override["description"])
        if override.get("permission") in {"readonly", "write"}:
            updates["permission"] = override["permission"]
            if override["permission"] == "readonly":
                tags.add("readonly")
                tags.discard("write")
            else:
                tags.add("write")
                tags.discard("readonly")
        metadata = dict(definition.metadata)
        metadata["config_override"] = override
        updates["metadata"] = metadata
        return definition.model_copy(update=updates)

    async def start(
        self,
        message: str | UserMessage,
        session_id: str | None = None,
        mode: Literal["normal", "orchestrator"] = "normal",
        directives: RunDirectives | dict[str, Any] | None = None,
    ) -> RunHandle:
        await self._ensure_started()
        assert self.paths and self.config and self.store
        if mode == "orchestrator" and not (self.worker_runtime and self.worker_runtime.list_workers()):
            raise ConfigError("orchestrator mode requires at least one configured worker pool")
        run_directives = self._normalize_run_directives(directives, session_id=session_id or "")
        run_mode = RunMode(mode)
        session_id = session_id or new_id("sess")
        if run_directives.mentioned_worker is not None and not run_directives.mentioned_worker.worker_agent_id:
            resolved = self.resolve_worker_mention(run_directives.mentioned_worker.mention, session_id=session_id)
            if not resolved.resolved:
                raise AgentCoreError(
                    ErrorCode.WORKER_NOT_AVAILABLE,
                    resolved.error_message or f"worker mention not resolved: {run_directives.mentioned_worker.mention}",
                    details={"resolution": resolved.model_dump(mode="json")},
                )
            run_directives = run_directives.model_copy(update={"mentioned_worker": resolved.to_directive()})
        directives_metadata = run_directives.model_dump(mode="json", exclude_none=True)
        session = await self.store.get_session(session_id)
        if session is not None:
            agent_id = str(session["root_agent_id"])
        else:
            agent_id = self._root_agent_id(session_id=session_id, mode=run_mode)
        run_id = new_id("run")
        stream = EventStream()
        handle = RunHandle(
            run_id=run_id,
            session_id=session_id,
            agent_id=agent_id,
            status=RunStatus.PENDING,
            mode=run_mode,
            _runtime=self,
            _stream=stream,
            _message=message,
            directives=directives_metadata,
        )
        session_created = await self.store.ensure_session(session_id=session_id, cwd=str(self.paths.project_dir), root_agent_id=agent_id)
        await self.store.ensure_agent(
            agent_id=agent_id,
            session_id=session_id,
            agent_type="orchestrator" if run_mode == RunMode.ORCHESTRATOR else "main",
        )
        if session_created:
            await self._run_observe_hook(
                event_type="session_started",
                session_id=session_id,
                agent_id=agent_id,
                run_id=run_id,
                payload={
                    "event_type": "SessionStart",
                    "session_id": session_id,
                    "agent_id": agent_id,
                    "run_id": run_id,
                    "mode": run_mode.value,
                    "cwd": str(self.paths.project_dir),
                },
            )
        if session_id in self._session_active:
            handle.status = RunStatus.QUEUED
            handle._queued = True
            await self.store.create_run(
                run_id=run_id,
                session_id=session_id,
                agent_id=agent_id,
                status=RunStatus.QUEUED.value,
                metadata={"directives": directives_metadata} if directives_metadata else None,
            )
            self._session_queues[session_id].append(handle)
            await self._emit(handle, "run_queued")
            return handle
        await self.store.create_run(
            run_id=run_id,
            session_id=session_id,
            agent_id=agent_id,
            status=RunStatus.PENDING.value,
            metadata={"directives": directives_metadata} if directives_metadata else None,
        )
        self._session_active[session_id] = handle
        handle._task = asyncio.create_task(self._run(handle, message))
        return handle

    async def replay_session(
        self,
        session_id: str,
        from_seq: int | None = None,
        to_seq: int | None = None,
        include_sensitive: bool = False,
    ) -> ReplayResult:
        return await runtime_replay.replay_session(
            self,
            session_id,
            from_seq=from_seq,
            to_seq=to_seq,
            include_sensitive=include_sensitive,
        )

    async def replay_run(self, run_id: str, include_sensitive: bool = False) -> ReplayResult:
        return await runtime_replay.replay_run(self, run_id, include_sensitive=include_sensitive)

    async def get_node_path(self, node_id: str) -> list[Node]:
        return await runtime_replay.get_node_path(self, node_id)

    async def list_sessions(self, limit: int = 20, offset: int = 0) -> list[SessionInfo]:
        return await runtime_sessions.list_sessions(self, limit=limit, offset=offset)

    async def list_session_nodes(self, session_id: str, limit: int = 20, offset: int = 0) -> list[SessionNodeInfo]:
        return await runtime_sessions.list_session_nodes(self, session_id, limit=limit, offset=offset)

    async def list_branchable_nodes(self, session_id: str, limit: int = 100, offset: int = 0) -> list[BranchableNodeView]:
        return await runtime_sessions.list_branchable_nodes(self, session_id, limit=limit, offset=offset)

    async def fork_session(
        self,
        source_session_id: str,
        node_id: str | None = None,
        new_session_id: str | None = None,
        mode: Literal["normal", "orchestrator"] = "normal",
    ) -> ForkSessionResult:
        return await runtime_sessions.fork_session(
            self,
            source_session_id,
            node_id=node_id,
            new_session_id=new_session_id,
            mode=mode,
        )

    async def list_skills(self) -> list[SkillInfo]:
        return await runtime_skills.list_skills(self)

    async def load_skill(
        self,
        session_id: str,
        name: str,
        mode: Literal["normal", "orchestrator"] = "normal",
    ) -> LoadSkillResult:
        return await runtime_skills.load_skill(self, session_id, name, mode=mode)

    def resolve_worker_mention(self, mention: str, *, session_id: str) -> WorkerMentionResolution:
        normalized = mention.strip()
        if normalized.startswith("@"):
            normalized = normalized[1:]
        if not normalized:
            return WorkerMentionResolution(
                mention=mention,
                status="missing",
                error_code="worker_not_found",
                error_message="worker mention is empty",
            )
        workers = list(self._effective_worker_configs)
        by_id = [worker for worker in workers if worker.worker_id == normalized]
        if by_id:
            return self._worker_resolution_from_match(mention, by_id[0], session_id=session_id)
        by_name = [worker for worker in workers if worker.name == normalized]
        if len(by_name) == 1:
            return self._worker_resolution_from_match(mention, by_name[0], session_id=session_id)
        if len(by_name) > 1:
            return WorkerMentionResolution(
                mention=mention,
                status="ambiguous",
                error_code="worker_ambiguous",
                error_message=f"worker mention is ambiguous: {mention}",
            )
        return WorkerMentionResolution(
            mention=mention,
            status="missing",
            error_code="worker_not_found",
            error_message=f"worker not found: {mention}",
        )

    async def list_worker_configs(self, include_disabled: bool = True, include_deleted: bool = False) -> list[WorkerConfigView]:
        await self._ensure_started()
        workers = self._effective_worker_configs
        if not include_disabled:
            workers = [worker for worker in workers if worker.enabled]
        if not include_deleted:
            workers = [worker for worker in workers if worker.deleted_at is None]
        return list(workers)

    async def get_worker_config(self, worker_id: str) -> WorkerConfigView | None:
        await self._ensure_started()
        for worker in self._effective_worker_configs:
            if worker.worker_id == worker_id:
                return worker
        return None

    async def create_worker_config(self, request: WorkerConfigCreate | dict[str, Any]) -> WorkerConfigView:
        await self._ensure_started()
        assert self.store
        config = _validate_worker_create(request)
        existing = await self.store.get_dynamic_worker_config(config.worker_id, include_disabled=True, include_deleted=True)
        if existing is not None and existing.deleted_at is None:
            raise AgentCoreError(ErrorCode.CONFIG_ERROR, f"dynamic worker already exists: {config.worker_id}")
        view = worker_view_from_create(config, source="dynamic")
        definition = agent_definition_from_worker_config(view, source="dynamic", metadata=view.metadata)
        if definition is not None:
            await self.store.upsert_dynamic_agent_definition(definition, source="dynamic")
        stored = await self.store.upsert_dynamic_worker_config(view, source="dynamic")
        await self.reload_worker_configs()
        return stored

    async def update_worker_config(self, worker_id: str, request: WorkerConfigUpdate | dict[str, Any]) -> WorkerConfigView:
        await self._ensure_started()
        assert self.store
        current = await self.store.get_dynamic_worker_config(worker_id, include_disabled=True, include_deleted=True)
        if current is None or current.deleted_at is not None:
            raise AgentCoreError(ErrorCode.WORKER_NOT_AVAILABLE, f"dynamic worker not found: {worker_id}")
        update = _validate_worker_update(request)
        data = current.model_dump(mode="python")
        data.update(update.model_dump(mode="python", exclude_unset=True, exclude_none=False))
        if data.get("agent_definition_id") is None and data.get("system_prompt"):
            from agent_core.agents.dynamic import synthesized_agent_definition_id

            data["agent_definition_id"] = synthesized_agent_definition_id(worker_id)
        view = WorkerConfigView.model_validate(data)
        definition = agent_definition_from_worker_config(view, source="dynamic", metadata=view.metadata)
        if definition is not None:
            await self.store.upsert_dynamic_agent_definition(definition, source="dynamic")
        stored = await self.store.upsert_dynamic_worker_config(view, source="dynamic")
        await self.reload_worker_configs()
        return stored

    async def disable_worker_config(self, worker_id: str) -> WorkerConfigView:
        await self._ensure_started()
        assert self.store
        stored = await self.store.mark_dynamic_worker_enabled(worker_id, False)
        if stored is None:
            raise AgentCoreError(ErrorCode.WORKER_NOT_AVAILABLE, f"dynamic worker not found: {worker_id}")
        await self.reload_worker_configs()
        return stored

    async def enable_worker_config(self, worker_id: str) -> WorkerConfigView:
        await self._ensure_started()
        assert self.store
        stored = await self.store.mark_dynamic_worker_enabled(worker_id, True)
        if stored is None:
            raise AgentCoreError(ErrorCode.WORKER_NOT_AVAILABLE, f"dynamic worker not found: {worker_id}")
        await self.reload_worker_configs()
        return stored

    async def soft_delete_worker_config(self, worker_id: str) -> WorkerConfigView:
        await self._ensure_started()
        assert self.store
        stored = await self.store.soft_delete_dynamic_worker_config(worker_id)
        if stored is None:
            raise AgentCoreError(ErrorCode.WORKER_NOT_AVAILABLE, f"dynamic worker not found: {worker_id}")
        await self.reload_worker_configs()
        return stored

    async def reload_worker_configs(self) -> None:
        await self._ensure_started()
        assert self.config and self.paths and self.store
        await self._reload_dynamic_agents_and_workers()

    def _normalize_run_directives(self, directives: RunDirectives | dict[str, Any] | None, *, session_id: str) -> RunDirectives:
        if directives is None:
            return RunDirectives()
        try:
            parsed = directives if isinstance(directives, RunDirectives) else RunDirectives.model_validate(directives)
        except ValidationError as exc:
            raise AgentCoreError(ErrorCode.VALIDATION_ERROR, "invalid run directives", details={"errors": exc.errors()}) from exc
        mentioned = parsed.mentioned_worker
        if mentioned is None:
            return parsed
        if mentioned.worker_agent_id:
            return parsed
        if session_id:
            resolved = self.resolve_worker_mention(mentioned.mention, session_id=session_id)
            if not resolved.resolved:
                raise AgentCoreError(
                    ErrorCode.WORKER_NOT_AVAILABLE,
                    resolved.error_message or f"worker mention not resolved: {mentioned.mention}",
                    details={"resolution": resolved.model_dump(mode="json")},
                )
            return parsed.model_copy(update={"mentioned_worker": resolved.to_directive()})
        return parsed

    def _worker_resolution_from_match(self, mention: str, worker: WorkerConfigView, *, session_id: str) -> WorkerMentionResolution:
        if worker.deleted_at is not None:
            return WorkerMentionResolution(
                mention=mention,
                worker_id=worker.worker_id,
                name=worker.name,
                status="deleted",
                error_code="worker_deleted",
                error_message=f"worker is deleted: {mention}",
            )
        if not worker.enabled:
            return WorkerMentionResolution(
                mention=mention,
                worker_id=worker.worker_id,
                name=worker.name,
                status="disabled",
                error_code="worker_disabled",
                error_message=f"worker is disabled: {mention}",
            )
        return WorkerMentionResolution(
            mention=mention,
            worker_id=worker.worker_id,
            worker_agent_id=worker_agent_id_for_session(session_id=session_id, worker_id=worker.worker_id),
            worker_pool_id=worker.worker_pool_id,
            name=worker.name,
            status="resolved",
        )

    async def delete_session(self, session_id: str) -> DeleteSessionResult:
        return await runtime_sessions.delete_session(self, session_id)

    async def switch_node(self, session_id: str, node_id: str) -> SwitchNodeResult:
        return await runtime_sessions.switch_node(self, session_id, node_id)

    async def cleanup_project_tasks(
        self,
        project: str | Path,
        dry_run: bool = True,
        include_failed: bool = False,
        include_cancelled: bool = False,
        older_than: datetime | None = None,
    ) -> CleanupResult:
        return await runtime_cleanup.cleanup_project_tasks(
            self,
            project,
            dry_run=dry_run,
            include_failed=include_failed,
            include_cancelled=include_cancelled,
            older_than=older_than,
        )

    async def delete_artifact(self, artifact_id: str) -> CleanupResult:
        return await runtime_cleanup.delete_artifact(self, artifact_id)

    async def cleanup_artifacts(
        self,
        session_id: str | None = None,
        dry_run: bool = True,
        include_all: bool = False,
        older_than: datetime | None = None,
        max_bytes: int | None = None,
    ) -> CleanupResult:
        return await runtime_cleanup.cleanup_artifacts(
            self,
            session_id=session_id,
            dry_run=dry_run,
            include_all=include_all,
            older_than=older_than,
            max_bytes=max_bytes,
        )

    async def _ensure_started(self) -> None:
        if self._started:
            return
        self.config, self.paths = load_runtime_config(
            home_dir=self._home_dir_arg,
            config_path=self._config_path_arg,
            project_dir=self._project_dir_arg,
            session_db_path=self._session_db_path_arg,
        )
        write_required_project_dirs(self.paths)
        resolve_memory_dir(
            self.config.memory.memory_dir,
            home_dir=self.paths.home_dir,
            project_dir=self.paths.project_dir,
        )
        self.store = SQLiteStore(self.paths.session_db_path)
        self.artifacts = ArtifactManager(home_dir=self.paths.home_dir)
        self._hooks = normalize_hooks(load_hooks(self.paths.home_dir)) if self.config.hooks.enabled else []
        await self._reload_dynamic_agents_and_workers()
        if self.config.tools.declarative_enabled:
            load_declarative_tools(self.tool_registry, self.paths.home_dir)
        self.task_service.replay_project(self.paths.project_dir)
        effective_agents_config = agents_config_from_worker_views(self.config.agents, self._effective_worker_configs)
        self.worker_runtime = WorkerPoolRuntime(effective_agents_config)
        register_agent_tools(self.tool_registry, self.agent_definitions, self.worker_runtime)
        self.agent_definitions.validate_suggested_tools(self.tool_registry.names())
        self._mcp_manager = McpToolManager(load_mcp_config(self.paths.home_dir), self.config.tools)
        try:
            self._provider = self.provider_registry.create(self.config.model.provider, self.config.model)
            self._provider_cache[self._provider_cache_key(self.config.model)] = self._provider
        except KeyError as exc:
            raise ConfigError(f"provider not registered: {self.config.model.provider}") from exc
        self._started = True

    async def _reload_dynamic_agents_and_workers(self) -> None:
        assert self.config and self.paths and self.store
        code_definitions = dict(self._code_agent_definitions)
        self.agent_definitions.reset_to_builtin()
        for definition in code_definitions.values():
            self.agent_definitions.register(definition, source="code")
        self.agent_definitions.load_user_dir(self.paths.home_dir / "agents")
        self.agent_definitions.load_json_dir(self.paths.home_dir / "agents")

        json_workers = load_json_worker_configs(self.paths.home_dir / "workers")
        for worker in json_workers:
            definition = agent_definition_from_worker_config(worker, source="json", metadata=worker.metadata)
            if definition is not None:
                self.agent_definitions.register_overlay(definition, source="json")

        dynamic_definitions = await self.store.list_dynamic_agent_definitions()
        for definition in dynamic_definitions:
            self.agent_definitions.register_overlay(definition, source="dynamic")

        dynamic_workers = await self.store.list_dynamic_worker_configs(include_disabled=True, include_deleted=True)
        for worker in dynamic_workers:
            if worker.deleted_at is not None:
                continue
            definition = agent_definition_from_worker_config(worker, source="dynamic", metadata=worker.metadata)
            if definition is not None:
                self.agent_definitions.register_overlay(definition, source="dynamic")
        config_workers = config_worker_views(
            self.config.agents,
            {definition.agent_definition_id: definition for definition in self.agent_definitions.list()},
        )
        self._effective_worker_configs = merge_worker_views(config_workers, json_workers, dynamic_workers)
        effective_agents_config = agents_config_from_worker_views(self.config.agents, self._effective_worker_configs)
        if self.worker_runtime is not None:
            self.worker_runtime.configure(effective_agents_config)

    def _provider_for_model(self, model_config) -> ProviderAdapter:
        key = self._provider_cache_key(model_config)
        cached = self._provider_cache.get(key)
        if cached is not None:
            return cached
        try:
            provider = self.provider_registry.create(model_config.provider, model_config)
        except KeyError as exc:
            raise ConfigError(f"provider not registered: {model_config.provider}") from exc
        self._provider_cache[key] = provider
        return provider

    @staticmethod
    def _provider_cache_key(model_config) -> str:
        import json

        return json.dumps(model_config.model_dump(mode="json"), sort_keys=True)

    async def _run(self, handle: RunHandle, message: str | UserMessage) -> None:
        await runtime_main_loop.run_main_loop(self, handle, message)

    async def _persist_partial_assistant_node(
        self,
        handle: RunHandle,
        *,
        parent_id: str | None,
        text: str,
        metadata: dict[str, Any],
    ) -> Node:
        assert self.store
        return await self.store.add_node(
            session_id=handle.session_id,
            parent_id=parent_id,
            agent_id=handle.agent_id,
            run_id=handle.run_id,
            role="assistant",
            node_type="message",
            content=[TextBlock(text=text)],
            metadata=metadata,
            make_active=False,
        )

    async def _execute_tool_calls(self, handle: RunHandle, calls: list[ToolCall]):
        return await execute_tool_calls(self, handle, calls)

    def _worker_tool_context(
        self,
        *,
        session_id: str,
        run_id: str,
        agent_id: str,
        parent_agent_id: str,
        parent_run_id: str,
        worker_scope: dict[str, Any],
        allowed_tool_names: set[str] | None = None,
    ) -> ToolExecutionContext:
        return runtime_agent_tools.worker_tool_context(
            self,
            session_id=session_id,
            run_id=run_id,
            agent_id=agent_id,
            parent_agent_id=parent_agent_id,
            parent_run_id=parent_run_id,
            worker_scope=worker_scope,
            allowed_tool_names=allowed_tool_names,
        )

    async def _persist_result_artifacts(self, handle: RunHandle, call: ToolCall, result) -> None:
        await runtime_artifacts.persist_result_artifacts(self, handle, call, result)

    async def _persist_provider_debug_artifact(self, handle: RunHandle, model_event: Any, *, node_id: str | None) -> None:
        await runtime_artifacts.persist_provider_debug_artifact(self, handle, model_event, node_id=node_id)

    async def _persist_run_debug_artifact(
        self,
        *,
        session_id: str,
        agent_id: str | None,
        run_id: str | None,
        model_event: Any,
        node_id: str | None,
    ) -> runtime_artifacts.ProviderDebugArtifact | None:
        return await runtime_artifacts.persist_run_debug_artifact(
            self,
            session_id=session_id,
            agent_id=agent_id,
            run_id=run_id,
            model_event=model_event,
            node_id=node_id,
        )

    async def _run_stop_hooks(self, handle: RunHandle, *, end_node_id: str | None):
        return await hooks_runtime.run_stop_hooks(self, handle, end_node_id=end_node_id)

    async def _maybe_start_background_compact(self, handle: RunHandle) -> None:
        await compaction_runtime.maybe_start_background_compact(self, handle)

    async def _try_recovery_compact(
        self,
        handle: RunHandle,
        *,
        user_node: Node,
        messages: list[ModelMessage],
        system_blocks: list[SystemBlock],
        context_config: ContextConfig,
        tools: list[ToolDefinition],
    ) -> dict[str, Any] | None:
        return await compaction_runtime.try_recovery_compact(
            self,
            handle,
            user_node=user_node,
            messages=messages,
            system_blocks=system_blocks,
            context_config=context_config,
            tools=tools,
        )

    async def _maybe_run_memory_extraction(self, handle: RunHandle, *, prompt_text: str) -> None:
        await runtime_memory.maybe_run_memory_extraction(self, handle, prompt_text=prompt_text)

    async def _memory_extraction_trigger_reason(
        self,
        *,
        session_id: str,
        sources: list[tuple[int, Node]],
        latest_user_text: str,
        max_pending_messages: int,
        token_threshold: int,
    ) -> str | None:
        return await runtime_memory.memory_extraction_trigger_reason(
            self,
            session_id=session_id,
            sources=sources,
            latest_user_text=latest_user_text,
            max_pending_messages=max_pending_messages,
            token_threshold=token_threshold,
        )

    async def _has_explicit_memory_intent(self, *, session_id: str, latest_user_text: str) -> bool:
        return await runtime_memory.has_explicit_memory_intent(self, session_id=session_id, latest_user_text=latest_user_text)

    async def _run_structured_json_model(
        self,
        *,
        model_config: Any,
        schema: dict[str, Any],
        purpose: str,
        session_id: str,
        system_text: str,
        user_text: str,
        max_output_tokens: int,
    ) -> dict[str, Any]:
        return await runtime_memory.run_structured_json_model(
            self,
            model_config=model_config,
            schema=schema,
            purpose=purpose,
            session_id=session_id,
            system_text=system_text,
            user_text=user_text,
            max_output_tokens=max_output_tokens,
        )

    async def _run_memory_extraction_for_sources(
        self,
        *,
        session_id: str,
        cursor_seq: int,
        sources: list[tuple[int, Node]],
        reason: str,
    ) -> None:
        await runtime_memory.run_memory_extraction_for_sources(
            self,
            session_id=session_id,
            cursor_seq=cursor_seq,
            sources=sources,
            reason=reason,
        )

    def _schedule_memory_idle_extraction(self, *, session_id: str) -> None:
        runtime_memory.schedule_memory_idle_extraction(self, session_id=session_id)

    def _cancel_memory_idle_task(self, session_id: str) -> None:
        runtime_memory.cancel_memory_idle_task(self, session_id)

    async def _memory_idle_extraction_after_delay(self, session_id: str, idle_seconds: float) -> None:
        await runtime_memory.memory_idle_extraction_after_delay(self, session_id, idle_seconds)

    async def _run_memory_extraction_model(self, *, session_id: str, model_config: Any, source_nodes: list[Node]) -> str:
        return await runtime_memory.run_memory_extraction_model(
            self,
            session_id=session_id,
            model_config=model_config,
            source_nodes=source_nodes,
        )

    async def _run_observe_hook(
        self,
        *,
        event_type: str,
        session_id: str,
        agent_id: str | None,
        run_id: str | None,
        payload: dict[str, Any],
    ) -> None:
        await hooks_runtime.run_observe_hook(
            self,
            event_type=event_type,
            session_id=session_id,
            agent_id=agent_id,
            run_id=run_id,
            payload=payload,
        )

    async def _emit(
        self,
        handle: RunHandle,
        event_type: str,
        *,
        level: str = "info",
        node_id: str | None = None,
        tool_call_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> RuntimeEvent:
        return await runtime_events.emit(
            self,
            handle,
            event_type,
            level=level,
            node_id=node_id,
            tool_call_id=tool_call_id,
            payload=payload,
        )

    async def _emit_realtime(
        self,
        handle: RunHandle,
        event_type: str,
        *,
        level: str = "info",
        node_id: str | None = None,
        tool_call_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> RuntimeEvent | None:
        return runtime_events.emit_realtime(
            handle,
            event_type,
            level=level,
            node_id=node_id,
            tool_call_id=tool_call_id,
            payload=payload,
        )

    def _open_child_run_stream(self, run_id: str) -> EventStream:
        return runtime_events.open_child_run_stream(self, run_id)

    async def _close_child_run_stream(self, run_id: str) -> None:
        await runtime_events.close_child_run_stream(self, run_id)

    async def _emit_child_run_event(
        self,
        *,
        stream: EventStream | None,
        mirror_handle: RunHandle | None = None,
        session_id: str,
        agent_id: str,
        run_id: str,
        event_type: str,
        level: str = "info",
        node_id: str | None = None,
        tool_call_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> RuntimeEvent:
        return await runtime_events.emit_child_run_event(
            self,
            stream=stream,
            mirror_handle=mirror_handle,
            session_id=session_id,
            agent_id=agent_id,
            run_id=run_id,
            event_type=event_type,
            level=level,
            node_id=node_id,
            tool_call_id=tool_call_id,
            payload=payload,
        )

    async def _emit_child_model_event(
        self,
        *,
        stream: EventStream,
        session_id: str,
        agent_id: str,
        run_id: str,
        event,
    ) -> None:
        await runtime_events.emit_child_model_event(
            self,
            stream=stream,
            session_id=session_id,
            agent_id=agent_id,
            run_id=run_id,
            event=event,
        )

    async def _cancel_run(self, handle: RunHandle) -> CancelResult:
        return await run_control.cancel_run(self, handle)

    async def _start_next_queued(self, session_id: str) -> None:
        await run_control.start_next_queued(self, session_id)

    async def _inspect_run(self, run_id: str, include_sensitive: bool = False) -> InspectResult:
        replay = await self.replay_run(run_id, include_sensitive=include_sensitive)
        return InspectResult(
            run_id=run_id,
            nodes=replay.nodes,
            events=replay.events,
            artifacts=replay.artifacts,
            model_requests=replay.model_requests,
            task_wal_errors=replay.task_wal_errors,
        )

    async def _child_events(self, child_run_id: str, debug: bool = False):
        async for event in runtime_events.child_events(self, child_run_id, debug=debug):
            yield event

    async def _ensure_mcp_tools(
        self,
        *,
        handle: RunHandle | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
    ) -> None:
        await mcp_runtime.ensure_mcp_tools(
            self,
            handle=handle,
            session_id=session_id,
            agent_id=agent_id,
            run_id=run_id,
        )
