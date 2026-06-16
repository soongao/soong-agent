from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any, Literal

from agent_core.agents.registry import AgentDefinitionRegistry
from agent_core.agents.child import ChildAgentManager
from agent_core.agents.workers import WorkerPoolRuntime, worker_agent_id_for_session
from agent_core.api.handles import RunHandle
from agent_core.artifacts.redaction import redact_value
from agent_core.artifacts import ArtifactManager
from agent_core.config.loader import load_runtime_config, resolve_model_config, write_required_project_dirs
from agent_core.compact import DEFAULT_COMPACT_AGENT_ID
from agent_core.config.models import AgentCoreConfig, ContextConfig, ModelConfig
from agent_core.config.paths import ResolvedPaths
from agent_core.context import build_context_messages, build_system_blocks
from agent_core.context.state import RuntimeContextState
from agent_core.errors import AgentCoreError, ConfigError
from agent_core.errors.codes import ErrorCode
from agent_core.events import EventStream, make_event
from agent_core.hooks.loader import load_hooks, normalize_hooks
from agent_core.mcp.config import load_mcp_config
from agent_core.mcp.discovery import McpToolManager
from agent_core.mcp.tools import register_mcp_tools
from agent_core.memory import MemoryExtractionJob, MemoryScanCursor, parse_memory_candidates, resolve_memory_dir
from agent_core.providers import ModelMessage, ModelRequest, ProviderAdapter, ProviderRegistry, SystemBlock, default_provider_registry
from agent_core.providers.base import ModelRole, StopReason
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
    ArtifactRefBlock,
    CancelResult,
    CleanupResult,
    DeleteSessionResult,
    ErrorPayload,
    InspectResult,
    JsonBlock,
    Node,
    PermissionDecision,
    PermissionRequest,
    ReplayResult,
    RunMode,
    RunStatus,
    RuntimeEvent,
    TextBlock,
    ToolCall,
    ToolCallBlock,
    ToolDefinition,
    ToolResultBlock,
    SwitchNodeResult,
    UserMessage,
)
from agent_core.types.tools import error_tool_result

PermissionCallback = Callable[[PermissionRequest], Awaitable[PermissionDecision]]
RAW_DEBUG_METADATA_KEY = "raw_debug"
_MEMORY_EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "memories": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "decision": {"type": "string", "enum": ["new"]},
                    "category": {"type": "string", "enum": ["user", "feedback", "reference"]},
                    "filename": {"type": "string", "pattern": r"^[a-z0-9][a-z0-9_-]*\.md$"},
                    "summary": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "source_node_ids": {"type": "array", "items": {"type": "string"}},
                    "content": {"type": "string"},
                },
                "required": ["decision", "category", "filename", "summary", "tags", "source_node_ids", "content"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["memories"],
    "additionalProperties": False,
}
_MEMORY_RECALL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "selected_paths": {
            "type": "array",
            "items": {"type": "string"},
        }
    },
    "required": ["selected_paths"],
    "additionalProperties": False,
}
_MEMORY_INTENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "explicit": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["explicit", "reason"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class _ProviderDebugArtifact:
    artifact_id: str
    provider: str
    summary: str | None


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
        self.agent_definitions.register(definition.model_copy(update={"source": source}), source=source)

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
        await self._ensure_started()
        assert self.store and self.paths and self.config and self._provider
        await self._ensure_mcp_tools(session_id=session_id, agent_id=parent_agent_id, run_id=parent_run_id)
        definition = self.agent_definitions.get(agent_definition_id)
        if definition is None:
            raise AgentCoreError(ErrorCode.INVALID_AGENT_DEFINITION, f"agent definition not found: {agent_definition_id}")
        if allowed_tools is not None:
            unknown = [name for name in allowed_tools if self.tool_registry.get(name) is None]
            if unknown:
                raise AgentCoreError(ErrorCode.VALIDATION_ERROR, f"allowed_tools contains unavailable tools: {unknown}")
            effective_names = {tool.name for tool in self._effective_tools(agent_role=mode)}
            excluded = [name for name in allowed_tools if name not in effective_names]
            if excluded:
                raise AgentCoreError(ErrorCode.VALIDATION_ERROR, f"allowed_tools contains tools outside effective set: {excluded}")
        child_manager = self._child_managers.setdefault(
            parent_run_id,
            ChildAgentManager(max_children_per_run=self.config.agents.max_children_per_run),
        )
        if not child_manager.can_start():
            raise AgentCoreError(
                ErrorCode.CHILD_AGENT_LIMIT_EXCEEDED,
                f"parent run exceeds max_children_per_run={self.config.agents.max_children_per_run}",
            )
        if self._session_child_counts[session_id] >= self.config.agents.max_concurrent_children_per_session:
            raise AgentCoreError(
                ErrorCode.CHILD_AGENT_LIMIT_EXCEEDED,
                f"session exceeds max_concurrent_children_per_session={self.config.agents.max_concurrent_children_per_session}",
            )
        child_manager.started()
        self._session_child_counts[session_id] += 1
        agent_id = new_id(f"agent_{mode}")
        run_id = new_id("run")
        child_stream = self._open_child_run_stream(run_id)
        parent_id = await self.store.active_node_id(session_id)
        fork_from_node_id = parent_id if mode == "fork" else None
        await self.store.ensure_agent(
            agent_id=agent_id,
            session_id=session_id,
            agent_type=mode,
            status="running",
            parent_agent_id=parent_agent_id,
            created_by_run_id=parent_run_id,
            fork_from_node_id=fork_from_node_id,
            metadata={
                "agent_definition_id": agent_definition_id,
                "purpose": mode,
            },
        )
        await self.store.create_run(run_id=run_id, session_id=session_id, agent_id=agent_id, status=RunStatus.RUNNING.value)
        await self._emit_child_run_event(
            stream=child_stream,
            mirror_handle=parent_handle,
            session_id=session_id,
            agent_id=agent_id,
            run_id=run_id,
            event_type="child_agent_created" if mode == "sub" else "fork_agent_created",
            payload={
                "parent_agent_id": parent_agent_id,
                "parent_run_id": parent_run_id,
                "agent_definition_id": agent_definition_id,
                "child_agent_id": agent_id,
                "child_run_id": run_id,
                "fork_from_node_id": fork_from_node_id,
            },
        )
        task_node = await self.store.add_node(
            session_id=session_id,
            parent_id=parent_id,
            agent_id=agent_id,
            run_id=run_id,
            role="user",
            node_type="message",
            content=[TextBlock(text=task)],
            metadata={
                "parent_run_id": parent_run_id,
                "parent_agent_id": parent_agent_id,
                "agent_definition_id": agent_definition_id,
                "constraints": constraints,
                "expected_output_schema": expected_output_schema,
            },
            make_active=False,
        )
        tools = self._effective_tools(agent_role=mode)
        base_tool_names = {tool.name for tool in tools}
        if allowed_tools is not None:
            allowed_set = set(allowed_tools)
            tools = [tool for tool in tools if tool.name in allowed_set]
            base_tool_names &= allowed_set
        child_prompt = _child_prompt(
            task=task,
            constraints=constraints,
            expected_output_schema=expected_output_schema,
        )
        model_config = resolve_model_config(self.config, definition.model_profile)
        provider = self._provider_for_model(model_config)
        messages = [
            ModelMessage(
                role=ModelRole.USER,
                content=[TextBlock(text=child_prompt)],
                node_type="message",
                metadata={"node_id": task_node.node_id},
            )
        ]
        end_node_id = task_node.node_id
        result_text = ""
        timeout_seconds = _child_timeout_seconds(self.config, timeout_ms)
        try:
            async with asyncio.timeout(timeout_seconds):
                for _turn in range(self.config.runtime.max_turns):
                    system_blocks = build_system_blocks(
                        home_dir=self.paths.home_dir,
                        project_dir=self.paths.project_dir,
                        context_state=self._context_state_for_session(session_id),
                        memory_enabled=self.config.memory.enabled,
                        memory_dir_template=self.config.memory.memory_dir,
                    ) + [
                        SystemBlock(
                            block_id=f"agent_definition.{agent_definition_id}",
                            source="agent_definition",
                            content=_agent_definition_body_with_default(
                                self.agent_definitions,
                                definition,
                                default_id=self.config.agents.default_sub_agent_definition
                                if mode == "sub"
                                else self.config.agents.default_fork_agent_definition,
                            ),
                            priority=900,
                            dynamic=True,
                            metadata={
                                "agent_definition_id": agent_definition_id,
                                "fallback_default_id": None
                                if definition.body
                                else (
                                    self.config.agents.default_sub_agent_definition
                                    if mode == "sub"
                                    else self.config.agents.default_fork_agent_definition
                                ),
                            },
                        )
                    ]
                    await self._emit_child_run_event(
                        stream=child_stream,
                        mirror_handle=parent_handle,
                        session_id=session_id,
                        agent_id=agent_id,
                        run_id=run_id,
                        event_type="context_built",
                        payload=_context_build_report(messages, system_blocks, tools) | {"model": model_config.name},
                    )
                    request = ModelRequest(
                        model=model_config.name,
                        system=system_blocks,
                        messages=messages,
                        tools=tools,
                        temperature=model_config.temperature,
                        max_output_tokens=model_config.max_output_tokens,
                        metadata={"session_id": session_id, "run_id": run_id, "parent_run_id": parent_run_id},
                    )
                    _ensure_provider_supports_request(provider, request)
                    completed, text_parts = await _collect_model_completion(
                        provider,
                        request,
                        provider_failure_message=f"{mode} provider failed",
                        on_model_event=lambda event: self._emit_child_model_event(
                            stream=child_stream,
                            session_id=session_id,
                            agent_id=agent_id,
                            run_id=run_id,
                            event=event,
                        ),
                        on_completed=lambda event: self._persist_run_debug_artifact(
                            session_id=session_id,
                            agent_id=agent_id,
                            run_id=run_id,
                            model_event=event,
                            node_id=None,
                        ),
                    )
                    assistant_content = list(completed.content)
                    turn_text = ""
                    for block in completed.content:
                        if getattr(block, "type", None) == "text":
                            turn_text += getattr(block, "text", "")
                    if not turn_text:
                        turn_text = "".join(text_parts)
                    if turn_text:
                        result_text = turn_text
                    for call in completed.tool_calls:
                        assistant_content.append(
                            ToolCallBlock(tool_call_id=call.tool_call_id, name=call.name, arguments=call.arguments, metadata=call.metadata)
                        )
                    assistant_node = await self.store.add_node(
                        session_id=session_id,
                        parent_id=end_node_id,
                        agent_id=agent_id,
                        run_id=run_id,
                        role="assistant",
                        node_type="child_message",
                        content=assistant_content,
                        metadata={"stop_reason": completed.stop_reason.value if completed.stop_reason else None, "mode": mode},
                        make_active=False,
                    )
                    end_node_id = assistant_node.node_id
                    messages.append(
                        ModelMessage(
                            role=ModelRole.ASSISTANT,
                            content=assistant_content,
                            node_type="child_message",
                            metadata={"node_id": assistant_node.node_id},
                        )
                    )
                    if not completed.tool_calls:
                        _validate_expected_output_schema(result_text, expected_output_schema)
                        break
                    tool_results = await self._execute_child_tool_calls(
                        session_id=session_id,
                        run_id=run_id,
                        agent_id=agent_id,
                        agent_role=mode,
                        parent_agent_id=parent_agent_id,
                        parent_run_id=parent_run_id,
                        calls=completed.tool_calls,
                        allowed_tool_names=base_tool_names,
                        stream=child_stream,
                    )
                    tool_content = [
                        ToolResultBlock(
                            tool_call_id=result.tool_call_id,
                            is_error=result.is_error,
                            content=result.content,
                            error=result.error,
                            metadata={**result.metadata, "tool_name": result.tool_name},
                        )
                        for result in tool_results
                    ]
                    tool_node = await self.store.add_node(
                        session_id=session_id,
                        parent_id=end_node_id,
                        agent_id=agent_id,
                        run_id=run_id,
                        role="tool",
                        node_type="child_tool_result",
                        content=tool_content,
                        metadata={"mode": mode},
                        make_active=False,
                    )
                    end_node_id = tool_node.node_id
                    messages.append(
                        ModelMessage(
                            role=ModelRole.TOOL,
                            content=tool_content,
                            node_type="child_tool_result",
                            metadata={"node_id": tool_node.node_id},
                        )
                    )
                else:
                    raise AgentCoreError(ErrorCode.INTERNAL_ERROR, f"{mode} agent max turns exceeded")
        except TimeoutError as exc:
            child_manager.finished()
            self._session_child_counts[session_id] = max(0, self._session_child_counts[session_id] - 1)
            await self.store.update_run(
                run_id=run_id,
                status=RunStatus.FAILED.value,
                start_node_id=task_node.node_id,
                end_node_id=end_node_id,
                end_reason="failed",
                error={"code": ErrorCode.TIMEOUT.value, "message": f"{mode} agent timed out", "reason": "timeout"},
            )
            await self._emit_child_run_event(
                stream=child_stream,
                mirror_handle=parent_handle,
                session_id=session_id,
                agent_id=agent_id,
                run_id=run_id,
                event_type="child_agent_failed" if mode == "sub" else "fork_agent_failed",
                level="error",
                node_id=end_node_id,
                payload={"code": ErrorCode.TIMEOUT.value, "message": f"{mode} agent timed out", "child_run_id": run_id},
            )
            await self._close_child_run_stream(run_id)
            raise AgentCoreError(ErrorCode.TIMEOUT, f"{mode} agent timed out") from exc
        except AgentCoreError as exc:
            child_manager.finished()
            self._session_child_counts[session_id] = max(0, self._session_child_counts[session_id] - 1)
            await self.store.update_run(
                run_id=run_id,
                status=RunStatus.FAILED.value,
                start_node_id=task_node.node_id,
                end_node_id=end_node_id,
                end_reason="failed",
                error={"code": exc.code.value, "message": exc.message},
            )
            await self._emit_child_run_event(
                stream=child_stream,
                mirror_handle=parent_handle,
                session_id=session_id,
                agent_id=agent_id,
                run_id=run_id,
                event_type="child_agent_failed" if mode == "sub" else "fork_agent_failed",
                level="error",
                node_id=end_node_id,
                payload={"code": exc.code.value, "message": exc.message, "child_run_id": run_id},
            )
            await self._close_child_run_stream(run_id)
            raise
        except Exception as exc:
            child_manager.finished()
            self._session_child_counts[session_id] = max(0, self._session_child_counts[session_id] - 1)
            await self.store.update_run(
                run_id=run_id,
                status=RunStatus.FAILED.value,
                start_node_id=task_node.node_id,
                end_node_id=end_node_id,
                end_reason="failed",
                error={"code": ErrorCode.INTERNAL_ERROR.value, "message": str(exc)},
            )
            await self._emit_child_run_event(
                stream=child_stream,
                mirror_handle=parent_handle,
                session_id=session_id,
                agent_id=agent_id,
                run_id=run_id,
                event_type="child_agent_failed" if mode == "sub" else "fork_agent_failed",
                level="error",
                node_id=end_node_id,
                payload={"code": ErrorCode.INTERNAL_ERROR.value, "message": str(exc), "child_run_id": run_id},
            )
            await self._close_child_run_stream(run_id)
            raise
        result_node = await self.store.add_node(
            session_id=session_id,
            parent_id=end_node_id,
            agent_id=agent_id,
            run_id=run_id,
            role="assistant",
            node_type="child_result",
            content=[TextBlock(text=result_text)],
            metadata={
                "agent_definition_id": agent_definition_id,
                "mode": mode,
                "constraints": constraints,
                "expected_output_schema": expected_output_schema,
            },
            make_active=False,
        )
        await self.store.update_run(
            run_id=run_id,
            status=RunStatus.COMPLETED.value,
            start_node_id=task_node.node_id,
            end_node_id=result_node.node_id,
            end_reason="completed",
        )
        await self.store.update_agent(
            agent_id=agent_id,
            status=RunStatus.COMPLETED.value,
            result={
                "result_summary": result_text,
                "child_run_id": run_id,
                "agent_definition_id": agent_definition_id,
            },
        )
        await self._emit_child_run_event(
            stream=child_stream,
            mirror_handle=parent_handle,
            session_id=session_id,
            agent_id=agent_id,
            run_id=run_id,
            event_type="child_agent_completed" if mode == "sub" else "fork_agent_completed",
            node_id=result_node.node_id,
            payload={"result_summary": result_text, "child_agent_id": agent_id, "child_run_id": run_id},
        )
        await self._close_child_run_stream(run_id)
        child_manager.finished()
        self._session_child_counts[session_id] = max(0, self._session_child_counts[session_id] - 1)
        return {
            "child_run_id": run_id,
            "child_agent_id": agent_id,
            "agent_definition_id": agent_definition_id,
            "result_summary": result_text,
            "status": "completed",
        }

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
        await self._ensure_started()
        assert self.store and self.paths and self.config and self._provider and self.worker_runtime and self.artifacts
        if allowed_step_ids == []:
            raise AgentCoreError(ErrorCode.VALIDATION_ERROR, "allowed_step_ids cannot be empty")
        step_scope = list(dict.fromkeys(str(item) for item in allowed_step_ids)) if allowed_step_ids is not None else None
        worker = self.worker_runtime.select_worker(
            worker_pool_id=worker_pool_id,
            worker_agent_id=worker_agent_id,
            session_id=session_id,
        )
        definition = self.agent_definitions.get(worker.agent_definition_id)
        if definition is None:
            raise AgentCoreError(ErrorCode.INVALID_AGENT_DEFINITION, f"worker agent definition not found: {worker.agent_definition_id}")
        preflight_context = ToolExecutionContext(
            session_id=session_id,
            run_id=parent_run_id,
            agent_id=parent_agent_id,
            agent_role="orchestrator",
            project_dir=self.paths.project_dir,
            home_dir=self.paths.home_dir,
            config=self.config,
            artifact_manager=self.artifacts,
            permission_callback=self.permission_callback,
            permission_cache=self._permission_caches[session_id],
        )
        dispatchable_steps = self.task_service.dispatchable_steps(
            preflight_context,
            task_id=task_id,
            worker_pool_id=worker.pool_id,
            allowed_step_ids=step_scope,
        )
        agent_id = self._worker_agent_id(session_id=session_id, worker_id=worker.worker_id)
        if not dispatchable_steps:
            return {
                "worker_agent_id": agent_id,
                "worker_id": worker.worker_id,
                "child_run_id": None,
                "stream_id": None,
                "selection_reason": "first_idle" if worker_agent_id is None else "specified_worker",
                "worker_result": None,
                "claimed_step_id": None,
                "step_status": None,
                "step_result_summary": None,
                "no_step_claimed": True,
            }
        base_tools = self._effective_tools(agent_role="worker")
        base_names = {tool.name for tool in base_tools}
        worker_allowed = worker.allowed_tools
        if worker_allowed is not None:
            unknown = [name for name in worker_allowed if self.tool_registry.get(name) is None]
            if unknown:
                raise AgentCoreError(ErrorCode.VALIDATION_ERROR, f"worker allowed_tools contains unavailable tools: {unknown}")
            excluded = [name for name in worker_allowed if name not in base_names]
            if excluded:
                raise AgentCoreError(ErrorCode.VALIDATION_ERROR, f"worker allowed_tools contains tools outside effective set: {excluded}")
            base_names &= set(worker_allowed)
        if allowed_tools is not None:
            unknown = [name for name in allowed_tools if self.tool_registry.get(name) is None]
            if unknown:
                raise AgentCoreError(ErrorCode.VALIDATION_ERROR, f"allowed_tools contains unavailable tools: {unknown}")
            excluded = [name for name in allowed_tools if name not in base_names]
            if excluded:
                raise AgentCoreError(ErrorCode.VALIDATION_ERROR, f"allowed_tools contains tools outside worker effective set: {excluded}")
            base_names &= set(allowed_tools)
        tools = [tool for tool in base_tools if tool.name in base_names]

        run_id = new_id("run_worker")
        worker_stream = self._open_child_run_stream(run_id)
        self.worker_runtime.mark_busy(worker, task_id=task_id, run_id=run_id, step_id=dispatchable_steps[0].step_id)
        current_task = asyncio.current_task()
        if current_task is not None:
            self._worker_run_tasks[run_id] = current_task
            self._worker_run_meta[run_id] = {
                "session_id": session_id,
                "task_id": task_id,
                "worker_id": worker.worker_id,
                "worker_agent_id": agent_id,
            }
        await self.store.ensure_agent(
            agent_id=agent_id,
            session_id=session_id,
            agent_type="sub",
            status="running",
            parent_agent_id=parent_agent_id,
            created_by_run_id=parent_run_id,
            metadata={
                "purpose": "worker",
                "worker_id": worker.worker_id,
                "worker_pool_id": worker.pool_id,
                "agent_definition_id": worker.agent_definition_id,
            },
        )
        await self.store.create_run(run_id=run_id, session_id=session_id, agent_id=agent_id, status=RunStatus.RUNNING.value)
        parent_id = await self.store.active_node_id(session_id)
        worker_prompt = _worker_prompt(
            instruction=instruction,
            task_id=task_id,
            worker_pool_id=worker.pool_id,
            allowed_step_ids=step_scope,
            dispatch_context=dispatch_context,
            constraints=constraints,
            expected_output_schema=expected_output_schema,
        )
        start_node = await self.store.add_node(
            session_id=session_id,
            parent_id=parent_id,
            agent_id=agent_id,
            run_id=run_id,
            role="user",
            node_type="worker_dispatch",
            content=[TextBlock(text=worker_prompt)],
            metadata={
                "parent_agent_id": parent_agent_id,
                "parent_run_id": parent_run_id,
                "task_id": task_id,
                "worker_pool_id": worker.pool_id,
                "allowed_step_ids": step_scope,
            },
            make_active=False,
        )
        await self.store.update_run(run_id=run_id, status=RunStatus.RUNNING.value, start_node_id=start_node.node_id)
        await self._emit_child_run_event(
            stream=worker_stream,
            mirror_handle=parent_handle,
            session_id=session_id,
            agent_id=agent_id,
            run_id=run_id,
            event_type="worker_run_started",
            node_id=start_node.node_id,
            payload={
                "parent_agent_id": parent_agent_id,
                "parent_run_id": parent_run_id,
                "task_id": task_id,
                "worker_id": worker.worker_id,
                "worker_agent_id": agent_id,
                "child_run_id": run_id,
                "stream_id": run_id,
            },
        )
        worker_scope = {"task_id": task_id, "allowed_step_ids": step_scope, "worker_pool_id": worker.pool_id}
        messages = [
            ModelMessage(
                role=ModelRole.USER,
                content=[TextBlock(text=worker_prompt)],
                node_type="worker_dispatch",
                metadata={"node_id": start_node.node_id},
            )
        ]
        end_node_id: str | None = start_node.node_id
        final_text = ""
        worker_result: dict[str, Any] | None = None
        error_payload: ErrorPayload | None = None
        try:
            model_config = resolve_model_config(self.config, definition.model_profile)
            async with asyncio.timeout(_child_timeout_seconds(self.config, timeout_ms)):
                for _turn in range(self.config.runtime.max_turns):
                    system_blocks = build_system_blocks(
                        home_dir=self.paths.home_dir,
                        project_dir=self.paths.project_dir,
                        context_state=self._context_state_for_session(session_id),
                        memory_enabled=self.config.memory.enabled,
                        memory_dir_template=self.config.memory.memory_dir,
                    ) + [
                        SystemBlock(
                            block_id=f"agent_definition.{worker.agent_definition_id}",
                            source="agent_definition",
                            content=_agent_definition_body_with_default(
                                self.agent_definitions,
                                definition,
                                default_id="default_worker_agent",
                            ),
                            priority=900,
                            dynamic=True,
                            metadata={
                                "agent_definition_id": worker.agent_definition_id,
                                "fallback_default_id": None if definition.body else "default_worker_agent",
                            },
                        )
                    ]
                    await self._emit_child_run_event(
                        stream=worker_stream,
                        mirror_handle=parent_handle,
                        session_id=session_id,
                        agent_id=agent_id,
                        run_id=run_id,
                        event_type="context_built",
                        payload=_context_build_report(messages, system_blocks, tools) | {"model": model_config.name},
                    )
                    request = ModelRequest(
                        model=model_config.name,
                        system=system_blocks,
                        messages=messages,
                        tools=tools,
                        temperature=model_config.temperature,
                        max_output_tokens=model_config.max_output_tokens,
                        metadata={"session_id": session_id, "run_id": run_id, "parent_run_id": parent_run_id, "worker_id": worker.worker_id},
                    )
                    provider = self._provider_for_model(model_config)
                    _ensure_provider_supports_request(provider, request)
                    completed, text_parts = await _collect_model_completion(
                        provider,
                        request,
                        provider_failure_message="worker provider failed",
                        on_model_event=lambda event: self._emit_child_model_event(
                            stream=worker_stream,
                            session_id=session_id,
                            agent_id=agent_id,
                            run_id=run_id,
                            event=event,
                        ),
                        on_completed=lambda event: self._persist_run_debug_artifact(
                            session_id=session_id,
                            agent_id=agent_id,
                            run_id=run_id,
                            model_event=event,
                            node_id=None,
                        ),
                    )
                    final_text += "".join(text_parts)
                    assistant_content = list(completed.content)
                    for block in completed.content:
                        if getattr(block, "type", None) == "text":
                            final_text = getattr(block, "text", final_text)
                    for call in completed.tool_calls:
                        assistant_content.append(ToolCallBlock(tool_call_id=call.tool_call_id, name=call.name, arguments=call.arguments, metadata=call.metadata))
                    assistant_node = await self.store.add_node(
                        session_id=session_id,
                        parent_id=end_node_id,
                        agent_id=agent_id,
                        run_id=run_id,
                        role="assistant",
                        node_type="worker_message",
                        content=assistant_content,
                        metadata={"stop_reason": completed.stop_reason.value if completed.stop_reason else None},
                        make_active=False,
                    )
                    end_node_id = assistant_node.node_id
                    messages.append(
                        ModelMessage(
                            role=ModelRole.ASSISTANT,
                            content=assistant_content,
                            node_type="worker_message",
                            metadata={"node_id": assistant_node.node_id},
                        )
                    )
                    if not completed.tool_calls:
                        _validate_expected_output_schema(final_text, expected_output_schema)
                        worker_result = {"text": final_text}
                        break
                    tool_results = await self._execute_worker_tool_calls(
                        session_id=session_id,
                        run_id=run_id,
                        agent_id=agent_id,
                        parent_agent_id=parent_agent_id,
                        parent_run_id=parent_run_id,
                        calls=completed.tool_calls,
                        worker_scope=worker_scope,
                        allowed_tool_names=base_names,
                        stream=worker_stream,
                    )
                    tool_content = [
                        ToolResultBlock(
                            tool_call_id=result.tool_call_id,
                            is_error=result.is_error,
                            content=result.content,
                            error=result.error,
                            metadata={**result.metadata, "tool_name": result.tool_name},
                        )
                        for result in tool_results
                    ]
                    tool_node = await self.store.add_node(
                        session_id=session_id,
                        parent_id=end_node_id,
                        agent_id=agent_id,
                        run_id=run_id,
                        role="tool",
                        node_type="worker_tool_result",
                        content=tool_content,
                        metadata={},
                        make_active=False,
                    )
                    end_node_id = tool_node.node_id
                    messages.append(
                        ModelMessage(
                            role=ModelRole.TOOL,
                            content=tool_content,
                            node_type="worker_tool_result",
                            metadata={"node_id": tool_node.node_id},
                        )
                    )
                else:
                    raise AgentCoreError(ErrorCode.INTERNAL_ERROR, "worker max turns exceeded")
            if error_payload:
                raise AgentCoreError(error_payload.code, error_payload.message, details=error_payload.details)
            fallback_context = self._worker_tool_context(
                session_id=session_id,
                run_id=run_id,
                agent_id=agent_id,
                parent_agent_id=parent_agent_id,
                parent_run_id=parent_run_id,
                worker_scope=worker_scope,
            )
            fallback = self.task_service.fail_unclosed_worker_step(
                fallback_context,
                task_id=task_id,
                worker_run_id=run_id,
                reason="worker_finished_without_terminal_step_status",
            )
            await self.store.update_run(
                run_id=run_id,
                status=RunStatus.COMPLETED.value,
                end_node_id=end_node_id,
                end_reason="completed",
            )
            summary = self._worker_step_summary(session_id=session_id, task_id=task_id, worker_run_id=run_id)
            if fallback is not None:
                summary = _summary_from_step(fallback["step"])
            await self._emit_child_run_event(
                stream=worker_stream,
                mirror_handle=parent_handle,
                session_id=session_id,
                agent_id=agent_id,
                run_id=run_id,
                event_type="worker_run_completed",
                node_id=end_node_id,
                payload={
                    "task_id": task_id,
                    "summary": summary,
                    "worker_id": worker.worker_id,
                    "worker_agent_id": agent_id,
                    "child_run_id": run_id,
                    "stream_id": run_id,
                },
            )
            await self.store.update_agent(
                agent_id=agent_id,
                status="idle",
                result={
                    "last_run_id": run_id,
                    "task_id": task_id,
                    "summary": summary,
                    "worker_result": worker_result or {"text": final_text},
                },
            )
            return {
                "worker_agent_id": agent_id,
                "worker_id": worker.worker_id,
                "child_run_id": run_id,
                "stream_id": run_id,
                "selection_reason": "first_idle" if worker_agent_id is None else "specified_worker",
                "worker_result": worker_result or {"text": final_text},
                **summary,
            }
        except TimeoutError as exc:
            fallback_context = self._worker_tool_context(
                session_id=session_id,
                run_id=run_id,
                agent_id=agent_id,
                parent_agent_id=parent_agent_id,
                parent_run_id=parent_run_id,
                worker_scope=worker_scope,
            )
            self.task_service.fail_unclosed_worker_step(
                fallback_context,
                task_id=task_id,
                worker_run_id=run_id,
                reason="worker_timeout",
            )
            await self.store.update_run(
                run_id=run_id,
                status=RunStatus.FAILED.value,
                end_node_id=end_node_id,
                end_reason="failed",
                error={"code": ErrorCode.TIMEOUT.value, "message": "worker agent timed out", "reason": "timeout"},
            )
            await self._emit_child_run_event(
                stream=worker_stream,
                mirror_handle=parent_handle,
                session_id=session_id,
                agent_id=agent_id,
                run_id=run_id,
                event_type="worker_run_failed",
                level="error",
                node_id=end_node_id,
                payload={
                    "task_id": task_id,
                    "code": ErrorCode.TIMEOUT.value,
                    "message": "worker agent timed out",
                    "worker_id": worker.worker_id,
                    "worker_agent_id": agent_id,
                    "child_run_id": run_id,
                    "stream_id": run_id,
                },
            )
            await self.store.update_agent(
                agent_id=agent_id,
                status=RunStatus.FAILED.value,
                result={"last_run_id": run_id, "task_id": task_id, "error": "worker agent timed out"},
            )
            raise AgentCoreError(ErrorCode.TIMEOUT, "worker agent timed out") from exc
        except asyncio.CancelledError:
            fallback_context = self._worker_tool_context(
                session_id=session_id,
                run_id=run_id,
                agent_id=agent_id,
                parent_agent_id=parent_agent_id,
                parent_run_id=parent_run_id,
                worker_scope=worker_scope,
            )
            self.task_service.fail_unclosed_worker_step(
                fallback_context,
                task_id=task_id,
                worker_run_id=run_id,
                reason="worker_cancelled",
            )
            await self.store.update_run(
                run_id=run_id,
                status=RunStatus.CANCELLED.value,
                end_node_id=end_node_id,
                end_reason="aborted_tools",
                error={"code": ErrorCode.CANCELLED.value, "message": "worker agent cancelled", "reason": "cancelled"},
            )
            await self._emit_child_run_event(
                stream=worker_stream,
                mirror_handle=parent_handle,
                session_id=session_id,
                agent_id=agent_id,
                run_id=run_id,
                event_type="worker_run_cancelled",
                node_id=end_node_id,
                payload={
                    "task_id": task_id,
                    "worker_id": worker.worker_id,
                    "worker_agent_id": agent_id,
                    "child_run_id": run_id,
                    "stream_id": run_id,
                },
            )
            await self.store.update_agent(
                agent_id=agent_id,
                status=RunStatus.CANCELLED.value,
                result={"last_run_id": run_id, "task_id": task_id, "cancelled": True},
            )
            raise
        except Exception as exc:
            fallback_context = self._worker_tool_context(
                session_id=session_id,
                run_id=run_id,
                agent_id=agent_id,
                parent_agent_id=parent_agent_id,
                parent_run_id=parent_run_id,
                worker_scope=worker_scope,
            )
            self.task_service.fail_unclosed_worker_step(
                fallback_context,
                task_id=task_id,
                worker_run_id=run_id,
                reason="worker_failed",
            )
            code = getattr(exc, "code", ErrorCode.INTERNAL_ERROR)
            message_text = getattr(exc, "message", str(exc))
            await self.store.update_run(
                run_id=run_id,
                status=RunStatus.FAILED.value,
                end_node_id=end_node_id,
                end_reason="failed",
                error={"code": str(code), "message": message_text},
            )
            await self._emit_child_run_event(
                stream=worker_stream,
                mirror_handle=parent_handle,
                session_id=session_id,
                agent_id=agent_id,
                run_id=run_id,
                event_type="worker_run_failed",
                level="error",
                node_id=end_node_id,
                payload={
                    "task_id": task_id,
                    "code": str(code),
                    "message": message_text,
                    "worker_id": worker.worker_id,
                    "worker_agent_id": agent_id,
                    "child_run_id": run_id,
                    "stream_id": run_id,
                },
            )
            await self.store.update_agent(
                agent_id=agent_id,
                status=RunStatus.FAILED.value,
                result={"last_run_id": run_id, "task_id": task_id, "error": message_text},
            )
            raise
        finally:
            self._worker_run_tasks.pop(run_id, None)
            self._worker_run_meta.pop(run_id, None)
            self.worker_runtime.mark_idle(worker)
            await self._close_child_run_stream(run_id)

    async def run_compact_agent(
        self,
        *,
        session_id: str,
        source_node_ids: list[str] | None = None,
        reason: str = "manual",
        first_kept_node_id: str | None = None,
    ) -> dict[str, Any]:
        await self._ensure_started()
        assert self.store and self.paths and self.config
        definition = self.agent_definitions.get(DEFAULT_COMPACT_AGENT_ID)
        if definition is None:
            raise AgentCoreError(ErrorCode.INVALID_AGENT_DEFINITION, "default compact agent missing")
        active_node_id = await self.store.active_node_id(session_id)
        active_node = await self.store.get_node(session_id, active_node_id) if active_node_id else None
        session = await self.store.get_session(session_id)
        parent_agent_id = active_node.agent_id if active_node is not None else (str(session["root_agent_id"]) if session else None)
        parent_run_id = active_node.run_id if active_node is not None else None
        agent_id = new_id("agent_compact")
        run_id = new_id("run_compact")
        await self.store.ensure_agent(
            agent_id=agent_id,
            session_id=session_id,
            agent_type="fork",
            status="running",
            parent_agent_id=parent_agent_id,
            created_by_run_id=parent_run_id,
            fork_from_node_id=active_node_id,
            metadata={"purpose": "compact", "reason": reason, "agent_definition_id": DEFAULT_COMPACT_AGENT_ID},
        )
        await self.store.create_run(run_id=run_id, session_id=session_id, agent_id=agent_id, status=RunStatus.RUNNING.value)
        replay = await self.replay_session(session_id)
        selected = [node for node in replay.nodes if not source_node_ids or node.node_id in set(source_node_ids)]
        compact_input = _compact_input(selected)
        await self.store.add_event(
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
        input_node = await self.store.add_node(
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
        model_config = resolve_model_config(self.config, self.config.compact.model_profile)
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
            max_output_tokens=min(model_config.max_output_tokens, self.config.compact.max_summary_tokens),
            metadata={"session_id": session_id, "run_id": run_id, "purpose": "compact"},
        )
        provider = self._provider_for_model(model_config)
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
            await self.store.update_run(
                run_id=run_id,
                status=RunStatus.FAILED.value,
                start_node_id=input_node.node_id,
                end_reason="failed",
                error=error_payload.model_dump(mode="json"),
            )
            await self.store.add_event(
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
        active_now = await self.store.active_node_id(session_id)
        stale = active_now != active_node_id
        compaction_node = None
        if not stale:
            kept_node_id = first_kept_node_id if first_kept_node_id is not None else active_node_id
            compaction_node = await self.store.add_node(
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
        await self.store.update_run(
            run_id=run_id,
            status=RunStatus.COMPLETED.value,
            start_node_id=input_node.node_id,
            end_node_id=compaction_node.node_id if compaction_node is not None else input_node.node_id,
            end_reason="completed",
        )
        await self.store.update_agent(
            agent_id=agent_id,
            status=RunStatus.COMPLETED.value,
            result={
                "compact_run_id": run_id,
                "compaction_node_id": compaction_node.node_id if compaction_node is not None else None,
                "stale": stale,
                "summary_length": len(summary),
            },
        )
        await self.store.add_event(
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

    async def select_memory(
        self,
        *,
        session_id: str,
        query: str,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        await self._ensure_started()
        assert self.config and self.paths
        memory_root = resolve_memory_dir(
            self.config.memory.memory_dir,
            home_dir=self.paths.home_dir,
            project_dir=self.paths.project_dir,
        )
        candidates = _memory_frontmatter_candidates(memory_root)
        if not candidates:
            return {"selected_paths": [], "selected_by_model": False, "candidates": []}
        model_config = resolve_model_config(self.config, self.config.memory.recall_model_profile)
        result = await self._run_structured_json_model(
            model_config=model_config,
            schema=_MEMORY_RECALL_SCHEMA,
            purpose="memory_recall_selector",
            session_id=session_id,
            system_text=(
                "You are the Soong Agent memory recall selector. "
                "Select user-level memory files that may help answer the query. "
                "Match semantically using summaries, categories, tags, and filenames; do not require exact word overlap. "
                "For identity, profile, role, language, preference, or background questions, include memories describing the user. "
                "Return selected_paths with at most the requested top_k. Return [] only when no candidate is plausibly relevant. "
                "Do not invent paths."
            ),
            user_text=(
                f"Query: {query}\n"
                f"top_k: {top_k or self.config.memory.recall_top_k}\n"
                "Memory candidates:\n"
                + "\n".join(_memory_candidate_selector_line(item) for item in candidates)
            ),
            max_output_tokens=min(model_config.max_output_tokens, 1024),
        )
        selected = [str(path) for path in result.get("selected_paths") or []]
        allowed = {item["relative_path"]: item for item in candidates}
        max_items = top_k or self.config.memory.recall_top_k
        selected = [path for path in selected if path in allowed][:max_items]
        return {
            "selected_paths": selected,
            "selected_by_model": True,
            "candidates": candidates,
        }

    async def cancel_worker_runs(
        self,
        *,
        session_id: str,
        task_id: str,
        worker_run_ids: list[str] | None = None,
        reason: str = "task_terminated",
    ) -> dict[str, Any]:
        await self._ensure_started()
        assert self.store
        requested = set(worker_run_ids or [])
        for run_id, meta in list(self._worker_run_meta.items()):
            if meta.get("session_id") == session_id and meta.get("task_id") == task_id:
                requested.add(run_id)
        current_task = asyncio.current_task()
        cancelled: list[str] = []
        missing: list[str] = []
        for run_id in sorted(requested):
            task = self._worker_run_tasks.get(run_id)
            meta = self._worker_run_meta.get(run_id)
            if task is None or task.done():
                missing.append(run_id)
                continue
            await self.store.add_event(
                make_event(
                    session_id=session_id,
                    agent_id=meta.get("worker_agent_id") if meta else None,
                    run_id=run_id,
                    event_type="worker_run_cancel_requested",
                    payload={
                        "task_id": task_id,
                        "reason": reason,
                        "worker_id": meta.get("worker_id") if meta else None,
                        "worker_agent_id": meta.get("worker_agent_id") if meta else None,
                    },
                )
            )
            if task is current_task:
                continue
            task.cancel()
            cancelled.append(run_id)
        timeout = (self.config.runtime.cancel_timeout_ms if self.config else 10000) / 1000
        for run_id in cancelled:
            task = self._worker_run_tasks.get(run_id)
            if task is None:
                continue
            try:
                await asyncio.wait_for(task, timeout=timeout)
            except asyncio.CancelledError:
                pass
            except asyncio.TimeoutError:
                await self.store.add_event(
                    make_event(
                        session_id=session_id,
                        run_id=run_id,
                        event_type="child_agent_cancel_timeout",
                        level="warning",
                        payload={"task_id": task_id, "reason": reason},
                    )
                )
            except Exception:
                pass
        return {"cancelled_worker_run_ids": cancelled, "missing_worker_run_ids": missing}

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
    ) -> RunHandle:
        await self._ensure_started()
        assert self.paths and self.config and self.store
        if mode == "orchestrator" and not any(pool.workers for pool in self.config.agents.worker_pools):
            raise ConfigError("orchestrator mode requires at least one configured worker pool")
        run_mode = RunMode(mode)
        session_id = session_id or new_id("sess")
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
            await self.store.create_run(run_id=run_id, session_id=session_id, agent_id=agent_id, status=RunStatus.QUEUED.value)
            self._session_queues[session_id].append(handle)
            await self._emit(handle, "run_queued")
            return handle
        await self.store.create_run(run_id=run_id, session_id=session_id, agent_id=agent_id, status=RunStatus.PENDING.value)
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
        await self._ensure_started()
        assert self.store
        nodes, events = await self.store.replay_session(session_id, from_seq=from_seq, to_seq=to_seq)
        artifacts = await self.store.list_artifacts(session_id=session_id)
        if not include_sensitive:
            nodes, events, artifacts = _redact_replay_payload(nodes, events, artifacts)
        model_requests = _model_request_views_from_events(events, run_id=None)
        return ReplayResult(
            session_id=session_id,
            nodes=nodes,
            events=events,
            artifacts=artifacts,
            model_requests=model_requests,
            task_wal_errors=self.task_service.unavailable_task_summaries(session_id),
        )

    async def replay_run(self, run_id: str, include_sensitive: bool = False) -> ReplayResult:
        await self._ensure_started()
        assert self.store
        session_id = await self.store.find_run_session(run_id)
        if session_id is None:
            return ReplayResult(session_id="", run_id=run_id)
        nodes, events = await self.store.replay_run(session_id, run_id)
        artifacts = [artifact for artifact in await self.store.list_artifacts(session_id=session_id) if artifact.get("run_id") == run_id]
        if not include_sensitive:
            nodes, events, artifacts = _redact_replay_payload(nodes, events, artifacts)
        model_requests = _model_request_views_from_events(events, run_id=run_id)
        return ReplayResult(
            session_id=session_id,
            run_id=run_id,
            nodes=nodes,
            events=events,
            artifacts=artifacts,
            model_requests=model_requests,
            task_wal_errors=self.task_service.unavailable_task_summaries(session_id),
        )

    async def get_node_path(self, node_id: str) -> list[Node]:
        await self._ensure_started()
        assert self.store
        return await self.store.get_node_path(node_id)

    async def delete_session(self, session_id: str) -> DeleteSessionResult:
        await self._ensure_started()
        assert self.store
        if session_id in self._session_active or self._session_queues.get(session_id):
            return DeleteSessionResult(
                session_id=session_id,
                deleted=False,
                error=ErrorPayload(code=ErrorCode.SESSION_ACTIVE, message="session has active or queued runs"),
            )
        if await self.store.has_active_runs(session_id):
            return DeleteSessionResult(
                session_id=session_id,
                deleted=False,
                error=ErrorPayload(code=ErrorCode.SESSION_ACTIVE, message="session has active or queued runs"),
            )
        artifacts = await self.store.list_artifacts(session_id=session_id)
        deletion_errors: list[dict[str, Any]] = []
        for artifact in artifacts:
            artifact_id = str(artifact.get("artifact_id") or "")
            path = Path(str(artifact.get("path") or ""))
            try:
                _delete_artifact_path(path)
            except OSError as exc:
                deletion_errors.append({"artifact_id": artifact_id, "path": str(path), "error": str(exc)})
        if deletion_errors:
            return DeleteSessionResult(
                session_id=session_id,
                deleted=False,
                error=ErrorPayload(
                    code=ErrorCode.STORAGE_ERROR,
                    message=f"failed to delete session artifacts: {session_id}",
                    details={"artifacts": deletion_errors},
                ),
            )
        await self.store.delete_session(session_id)
        return DeleteSessionResult(session_id=session_id, deleted=True)

    async def switch_node(self, session_id: str, node_id: str) -> SwitchNodeResult:
        await self._ensure_started()
        assert self.store
        if session_id in self._session_active or self._session_queues.get(session_id):
            return SwitchNodeResult(
                session_id=session_id,
                node_id=node_id,
                switched=False,
                error=ErrorPayload(code=ErrorCode.SESSION_ACTIVE, message="cannot switch active node while session has active or queued runs"),
            )
        if not await self.store.node_exists(session_id, node_id):
            return SwitchNodeResult(
                session_id=session_id,
                node_id=node_id,
                switched=False,
                error=ErrorPayload(code=ErrorCode.VALIDATION_ERROR, message="node not found in session"),
            )
        await self.store.set_active_node(session_id, node_id)
        await self.store.add_event(
            make_event(
                session_id=session_id,
                event_type="active_node_switched",
                node_id=node_id,
                payload={"node_id": node_id},
            )
        )
        return SwitchNodeResult(session_id=session_id, node_id=node_id, switched=True)

    async def cleanup_project_tasks(
        self,
        project: str | Path,
        dry_run: bool = True,
        include_failed: bool = False,
        include_cancelled: bool = False,
        older_than: datetime | None = None,
    ) -> CleanupResult:
        await self._ensure_started()
        assert self.paths
        root = Path(project).expanduser().resolve()
        task_root = root / ".soong-agent" / "tasks"
        candidates: list[dict[str, Any]] = []
        if task_root.exists():
            for wal in sorted(task_root.rglob("*.wal.jsonl")):
                if older_than is not None and not _path_older_than(wal, older_than):
                    continue
                text = wal.read_text(encoding="utf-8", errors="replace")
                if "task_completed" in text or (include_failed and "task_failed" in text) or (include_cancelled and "task_cancelled" in text):
                    candidates.append({"path": str(wal), "reason": "terminal_task_wal", "modified_at": _path_mtime_iso(wal)})
        deleted: list[str] = []
        errors: list[ErrorPayload] = []
        if not dry_run:
            for candidate in candidates:
                try:
                    Path(candidate["path"]).unlink(missing_ok=True)
                except OSError as exc:
                    errors.append(
                        ErrorPayload(
                            code=ErrorCode.STORAGE_ERROR,
                            message=f"failed to delete task WAL: {candidate['path']}",
                            details={"path": candidate["path"], "error": str(exc)},
                        )
                    )
                    continue
                deleted.append(candidate["path"])
        return CleanupResult(dry_run=dry_run, candidates=candidates, deleted=deleted, errors=errors)

    async def delete_artifact(self, artifact_id: str) -> CleanupResult:
        await self._ensure_started()
        assert self.store
        artifact = await self.store.get_artifact(artifact_id)
        if artifact is None:
            return CleanupResult(
                dry_run=False,
                errors=[ErrorPayload(code=ErrorCode.VALIDATION_ERROR, message=f"artifact not found: {artifact_id}")],
            )
        try:
            _delete_artifact_path(Path(artifact["path"]))
        except OSError as exc:
            return CleanupResult(
                dry_run=False,
                candidates=[{"artifact_id": artifact_id, "path": artifact["path"], "reason": "delete_artifact"}],
                errors=[
                    ErrorPayload(
                        code=ErrorCode.STORAGE_ERROR,
                        message=f"failed to delete artifact: {artifact_id}",
                        details={"artifact_id": artifact_id, "path": artifact["path"], "error": str(exc)},
                    )
                ],
            )
        await self.store.delete_artifact(artifact_id)
        return CleanupResult(
            dry_run=False,
            candidates=[{"artifact_id": artifact_id, "path": artifact["path"], "reason": "delete_artifact"}],
            deleted=[artifact_id],
        )

    async def cleanup_artifacts(
        self,
        session_id: str | None = None,
        dry_run: bool = True,
        include_all: bool = False,
        older_than: datetime | None = None,
        max_bytes: int | None = None,
    ) -> CleanupResult:
        await self._ensure_started()
        assert self.store
        import json

        artifacts = await self.store.list_artifacts(session_id=session_id)
        candidates = []
        for artifact in artifacts:
            try:
                metadata = json.loads(artifact.get("metadata_json") or "{}")
            except json.JSONDecodeError:
                metadata = {}
            if not _artifact_selected_for_cleanup(
                artifact=artifact,
                metadata=metadata,
                include_all=include_all,
                older_than=older_than,
                max_bytes=max_bytes,
            ):
                continue
            candidates.append(
                {
                    "artifact_id": artifact["artifact_id"],
                    "path": artifact["path"],
                    "reason": _artifact_cleanup_reason(metadata, include_all=include_all, max_bytes=max_bytes),
                    "size_bytes": artifact.get("size_bytes"),
                    "created_at": artifact.get("created_at"),
                }
            )
        deleted: list[str] = []
        errors: list[ErrorPayload] = []
        if not dry_run:
            for candidate in candidates:
                try:
                    _delete_artifact_path(Path(candidate["path"]))
                except OSError as exc:
                    errors.append(
                        ErrorPayload(
                            code=ErrorCode.STORAGE_ERROR,
                            message=f"failed to delete artifact: {candidate['artifact_id']}",
                            details={
                                "artifact_id": candidate["artifact_id"],
                                "path": candidate["path"],
                                "error": str(exc),
                            },
                        )
                    )
                    continue
                await self.store.delete_artifact(candidate["artifact_id"])
                deleted.append(candidate["artifact_id"])
        return CleanupResult(dry_run=dry_run, candidates=candidates, deleted=deleted, errors=errors)

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
        self.agent_definitions.load_user_dir(self.paths.home_dir / "agents")
        if self.config.tools.declarative_enabled:
            load_declarative_tools(self.tool_registry, self.paths.home_dir)
        self.task_service.replay_project(self.paths.project_dir)
        self.worker_runtime = WorkerPoolRuntime(self.config.agents)
        register_agent_tools(self.tool_registry, self.agent_definitions, self.worker_runtime)
        self.agent_definitions.validate_suggested_tools(self.tool_registry.names())
        self._mcp_manager = McpToolManager(load_mcp_config(self.paths.home_dir), self.config.tools)
        try:
            self._provider = self.provider_registry.create(self.config.model.provider, self.config.model)
            self._provider_cache[self._provider_cache_key(self.config.model)] = self._provider
        except KeyError as exc:
            raise ConfigError(f"provider not registered: {self.config.model.provider}") from exc
        self._started = True

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
        assert self.store and self.paths and self.config and self._provider and self.artifacts
        self._cancel_memory_idle_task(handle.session_id)
        handle.status = RunStatus.RUNNING
        await self.store.update_run(run_id=handle.run_id, status=RunStatus.RUNNING.value)
        await self._emit(handle, "loop_started")
        await self._ensure_mcp_tools(handle=handle)
        user = message if isinstance(message, UserMessage) else UserMessage.from_text(message)
        parent_id = await self.store.active_node_id(handle.session_id)
        user_content = user.content if isinstance(user.content, list) else [TextBlock(text=user.content)]
        prompt_text = "\n".join(getattr(block, "text", "") for block in user_content if getattr(block, "type", None) == "text")
        await self._run_observe_hook(
            event_type="user_prompt_submitted",
            session_id=handle.session_id,
            agent_id=handle.agent_id,
            run_id=handle.run_id,
            payload={
                "event_type": "UserPromptSubmit",
                "session_id": handle.session_id,
                "agent_id": handle.agent_id,
                "run_id": handle.run_id,
                "prompt": prompt_text[:12000],
                "metadata": user.metadata,
            },
        )
        user_node = await self.store.add_node(
            session_id=handle.session_id,
            parent_id=parent_id,
            agent_id=handle.agent_id,
            run_id=handle.run_id,
            role="user",
            node_type="message",
            content=user_content,
            metadata=user.metadata,
            make_active=True,
        )
        await self.store.update_run(run_id=handle.run_id, status=RunStatus.RUNNING.value, start_node_id=user_node.node_id)
        await self._emit(handle, "message_created", node_id=user_node.node_id, payload={"role": "user"})
        messages = build_context_messages(await self.store.get_node_path(user_node.node_id))
        task_board_message = _task_board_context_message(self.task_service, handle.session_id)
        if task_board_message is not None:
            messages.append(task_board_message)
        end_node_id: str | None = None
        partial_text_parts: list[str] = []
        model_parent_node_id: str | None = None
        empty_tool_result_retries = 0
        try:
            for _turn in range(self.config.runtime.max_turns):
                partial_text_parts = []
                system_blocks = build_system_blocks(
                    home_dir=self.paths.home_dir,
                    project_dir=self.paths.project_dir,
                    context_state=self._context_state_for_session(handle.session_id),
                    memory_enabled=self.config.memory.enabled,
                    memory_dir_template=self.config.memory.memory_dir,
                )
                system_budget = _apply_system_block_budget(system_blocks, self.config.context.dynamic_system_budget)
                system_blocks = system_budget["system_blocks"]
                tools = self._effective_tools(agent_role="orchestrator" if handle.mode == RunMode.ORCHESTRATOR else "main")
                context_bundle = _apply_context_budget(
                    messages=messages,
                    system_blocks=system_blocks,
                    context_config=self.config.context,
                    model_config=self.config.model,
                )
                await self._emit(
                    handle,
                    "context_built",
                    payload=_context_build_report(
                        context_bundle["messages"],
                        system_blocks,
                        tools,
                        trimmed_node_ids=context_bundle["trimmed_node_ids"],
                        trimmed_system_blocks=system_budget["trimmed_system_blocks"],
                        budget=context_bundle["budget"],
                        tokens_before_trim=context_bundle["tokens_before_trim"],
                        tokens_after_trim=context_bundle["tokens_after_trim"],
                        non_system_tokens_before_trim=context_bundle["non_system_tokens_before_trim"],
                        non_system_tokens_after_trim=context_bundle["non_system_tokens_after_trim"],
                        too_long=context_bundle["too_long"],
                    )
                    | {"model": self.config.model.name},
                )
                if context_bundle["too_long"]:
                    recovered = await self._try_recovery_compact(
                        handle,
                        user_node=user_node,
                        messages=messages,
                        system_blocks=system_blocks,
                        context_config=self.config.context,
                        tools=tools,
                    )
                    if recovered is not None:
                        messages = recovered["messages"]
                        context_bundle = recovered["context_bundle"]
                        end_node_id = recovered["end_node_id"]
                        await self._emit(
                            handle,
                            "context_built",
                            payload=_context_build_report(
                                context_bundle["messages"],
                                system_blocks,
                                tools,
                                trimmed_node_ids=context_bundle["trimmed_node_ids"],
                                trimmed_system_blocks=system_budget["trimmed_system_blocks"],
                                budget=context_bundle["budget"],
                                tokens_before_trim=context_bundle["tokens_before_trim"],
                                tokens_after_trim=context_bundle["tokens_after_trim"],
                                non_system_tokens_before_trim=context_bundle["non_system_tokens_before_trim"],
                                non_system_tokens_after_trim=context_bundle["non_system_tokens_after_trim"],
                                too_long=context_bundle["too_long"],
                            )
                            | {"recovery_compact": True, "model": self.config.model.name},
                        )
                    if context_bundle["too_long"]:
                        raise AgentCoreError(
                            ErrorCode.VALIDATION_ERROR,
                            "prompt_too_long",
                            details={
                                "end_reason": "prompt_too_long",
                                "estimated_input_tokens": context_bundle["tokens_after_trim"],
                                "non_system_budget": context_bundle["budget"],
                            },
                        )
                request = ModelRequest(
                    model=self.config.model.name,
                    system=system_blocks,
                    messages=context_bundle["messages"],
                    tools=tools,
                    temperature=self.config.model.temperature,
                    max_output_tokens=self.config.model.max_output_tokens,
                    metadata={"session_id": handle.session_id, "run_id": handle.run_id},
                )
                _ensure_provider_supports_request(self._provider, request)
                completed = None
                model_parent_node_id = user_node.node_id if end_node_id is None else end_node_id
                async for model_event in self._provider.stream(request):
                    if model_event.event_type == "model_started":
                        await self._emit(handle, "model_started", payload=model_event.metadata)
                    elif model_event.event_type == "model_text_delta":
                        partial_text_parts.append(model_event.text_delta or "")
                        await self._emit_realtime(handle, "model_text_delta", payload={"text": model_event.text_delta or ""})
                    elif model_event.event_type == "model_failed":
                        error = model_event.error or ErrorPayload(code=ErrorCode.PROVIDER_ERROR, message="provider failed")
                        if partial_text_parts:
                            partial_node = await self.store.add_node(
                                session_id=handle.session_id,
                                parent_id=user_node.node_id if end_node_id is None else end_node_id,
                                agent_id=handle.agent_id,
                                run_id=handle.run_id,
                                role="assistant",
                                node_type="message",
                                content=[TextBlock(text="".join(partial_text_parts))],
                                metadata={"partial": True, "failed": True, "error": error.model_dump(mode="json")},
                                make_active=False,
                            )
                            end_node_id = partial_node.node_id
                        raise AgentCoreError(error.code, error.message, details=error.details)
                    elif model_event.event_type == "model_completed":
                        completed = model_event
                        break
                if completed is None:
                    raise AgentCoreError(ErrorCode.PROVIDER_ERROR, "provider stream ended without model_completed")
                assistant_content = list(completed.content)
                for call in completed.tool_calls:
                    assistant_content.append(
                        ToolCallBlock(tool_call_id=call.tool_call_id, name=call.name, arguments=call.arguments, metadata=call.metadata)
                    )
                recover_empty_tool_result = (
                    not completed.tool_calls
                    and not _content_has_text(assistant_content)
                    and _last_message_is_tool_result(messages)
                    and empty_tool_result_retries < 1
                )
                assistant_node = await self.store.add_node(
                    session_id=handle.session_id,
                    parent_id=user_node.node_id if end_node_id is None else end_node_id,
                    agent_id=handle.agent_id,
                    run_id=handle.run_id,
                    role="assistant",
                    node_type="message",
                    content=assistant_content,
                    metadata={"stop_reason": completed.stop_reason.value if completed.stop_reason else None},
                    make_active=not completed.tool_calls and not recover_empty_tool_result,
                )
                end_node_id = assistant_node.node_id
                await self._persist_provider_debug_artifact(handle, completed, node_id=assistant_node.node_id)
                await self._emit(handle, "model_completed", node_id=assistant_node.node_id)
                messages.append(
                    ModelMessage(
                        role=ModelRole.ASSISTANT,
                        content=assistant_content,
                        node_type="message",
                        metadata={"node_id": assistant_node.node_id},
                    )
                )
                if not completed.tool_calls:
                    if recover_empty_tool_result:
                        empty_tool_result_retries += 1
                        retry_text = (
                            "The previous tool results are available in context. "
                            "Provide the final answer now, using those tool results."
                        )
                        retry_node = await self.store.add_node(
                            session_id=handle.session_id,
                            parent_id=end_node_id,
                            agent_id=handle.agent_id,
                            run_id=handle.run_id,
                            role="user",
                            node_type="empty_tool_result_recovery",
                            content=[TextBlock(text=retry_text)],
                            metadata={"synthetic": True, "reason": "empty_model_response_after_tool_result"},
                            make_active=False,
                        )
                        messages.append(
                            ModelMessage(
                                role=ModelRole.USER,
                                content=[TextBlock(text=retry_text)],
                                node_type="empty_tool_result_recovery",
                                metadata={"node_id": retry_node.node_id, "synthetic": True},
                            )
                        )
                        end_node_id = retry_node.node_id
                        await self._emit(
                            handle,
                            "model_empty_response_recovered",
                            node_id=retry_node.node_id,
                            payload={"reason": "empty_model_response_after_tool_result"},
                        )
                        continue
                    handle.status = RunStatus.COMPLETED
                    stop_decision = await self._run_stop_hooks(handle, end_node_id=end_node_id)
                    if stop_decision.denied:
                        await self._emit(
                            handle,
                            "stop_hook_prevented",
                            node_id=end_node_id,
                            payload={"reason": stop_decision.reason, "metadata": stop_decision.metadata},
                        )
                        note_text = f"Stop hook prevented completion. Reason: {stop_decision.reason or 'unspecified'}"
                        note_node = await self.store.add_node(
                            session_id=handle.session_id,
                            parent_id=end_node_id,
                            agent_id=handle.agent_id,
                            run_id=handle.run_id,
                            role="user",
                            node_type="hook_context",
                            content=[TextBlock(text=note_text)],
                            metadata={"hook_event": "Stop", "reason": stop_decision.reason, "metadata": stop_decision.metadata},
                            make_active=False,
                        )
                        messages.append(
                            ModelMessage(
                                role=ModelRole.USER,
                                content=[TextBlock(text=note_text)],
                                node_type="hook_context",
                                metadata={"node_id": note_node.node_id},
                            )
                        )
                        end_node_id = note_node.node_id
                        handle.status = RunStatus.RUNNING
                        continue
                    await self.store.update_run(
                        run_id=handle.run_id,
                        status=RunStatus.COMPLETED.value,
                        end_node_id=end_node_id,
                        end_reason="completed",
                    )
                    await self._emit(handle, "run_completed", node_id=end_node_id)
                    await self._emit(handle, "loop_completed", node_id=end_node_id)
                    await self._maybe_run_memory_extraction(handle, prompt_text=prompt_text)
                    await self._maybe_start_background_compact(handle)
                    return
                tool_results = await self._execute_tool_calls(handle, completed.tool_calls)
                tool_content = [
                    ToolResultBlock(
                        tool_call_id=result.tool_call_id,
                        is_error=result.is_error,
                        content=result.content,
                        error=result.error,
                        metadata={**result.metadata, "tool_name": result.tool_name},
                    )
                    for result in tool_results
                ]
                tool_node = await self.store.add_node(
                    session_id=handle.session_id,
                    parent_id=end_node_id,
                    agent_id=handle.agent_id,
                    run_id=handle.run_id,
                    role="tool",
                    node_type="message",
                    content=tool_content,
                    metadata={},
                    make_active=False,
                )
                end_node_id = tool_node.node_id
                messages.append(
                    ModelMessage(
                        role=ModelRole.TOOL,
                        content=tool_content,
                        node_type="message",
                        metadata={"node_id": tool_node.node_id},
                    )
                )
                for synthetic in _synthetic_context_nodes_from_tool_results(tool_results):
                    synthetic_node = await self.store.add_node(
                        session_id=handle.session_id,
                        parent_id=end_node_id,
                        agent_id=handle.agent_id,
                        run_id=handle.run_id,
                        role="user",
                        node_type=synthetic["node_type"],
                        content=[TextBlock(text=synthetic["text"])],
                        metadata=synthetic["metadata"],
                        make_active=False,
                    )
                    end_node_id = synthetic_node.node_id
                    messages.append(
                        ModelMessage(
                            role=ModelRole.USER,
                            content=[TextBlock(text=synthetic["text"])],
                            node_type=synthetic["node_type"],
                            metadata={"node_id": synthetic_node.node_id, **synthetic["metadata"]},
                        )
                    )
                refreshed_task_board_message = _task_board_context_message(self.task_service, handle.session_id)
                if refreshed_task_board_message is not None:
                    messages = [message for message in messages if message.node_type != "task_board"]
                    messages.append(refreshed_task_board_message)
            raise AgentCoreError(ErrorCode.INTERNAL_ERROR, "max turns exceeded")
        except asyncio.CancelledError:
            if partial_text_parts:
                partial_node = await self._persist_partial_assistant_node(
                    handle,
                    parent_id=model_parent_node_id,
                    text="".join(partial_text_parts),
                    metadata={"partial": True, "aborted": True, "abort_reason": "cancelled"},
                )
                end_node_id = partial_node.node_id
                await self._emit(
                    handle,
                    "aborted_streaming",
                    node_id=partial_node.node_id,
                    payload={"reason": "cancelled"},
                )
            handle.status = RunStatus.CANCELLED
            await self.store.update_run(
                run_id=handle.run_id,
                status=RunStatus.CANCELLED.value,
                end_node_id=end_node_id,
                end_reason="aborted_streaming" if partial_text_parts else "aborted_tools",
                error={"code": ErrorCode.CANCELLED.value, "message": "run cancelled", "reason": "cancelled"},
            )
            await self._emit(handle, "run_cancelled")
        except Exception as exc:
            handle.status = RunStatus.FAILED
            code = getattr(exc, "code", ErrorCode.INTERNAL_ERROR)
            message_text = getattr(exc, "message", str(exc))
            end_reason = getattr(exc, "details", {}).get("end_reason") if isinstance(getattr(exc, "details", None), dict) else None
            await self.store.update_run(
                run_id=handle.run_id,
                status=RunStatus.FAILED.value,
                end_node_id=end_node_id,
                end_reason=end_reason or "failed",
                error={"code": str(code), "message": message_text},
            )
            payload = {"code": str(code), "message": message_text}
            if end_reason:
                payload["end_reason"] = end_reason
            await self._emit(handle, "loop_failed", level="error", payload=payload)
        finally:
            self._session_active.pop(handle.session_id, None)
            await handle._stream.close()
            await self._start_next_queued(handle.session_id)

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
        assert self.paths and self.config and self.artifacts
        effective_tools = self._effective_tools(agent_role="orchestrator" if handle.mode == RunMode.ORCHESTRATOR else "main")
        context = ToolExecutionContext(
            session_id=handle.session_id,
            run_id=handle.run_id,
            agent_id=handle.agent_id,
            agent_role="orchestrator" if handle.mode == RunMode.ORCHESTRATOR else "main",
            project_dir=self.paths.project_dir,
            home_dir=self.paths.home_dir,
            config=self.config,
            artifact_manager=self.artifacts,
            permission_callback=self.permission_callback,
            permission_cache=self._permission_caches[handle.session_id],
            allowed_tool_names={tool.name for tool in effective_tools},
            effective_tool_definitions={tool.name: tool for tool in effective_tools},
            debug=self.debug,
            run_handle=handle,
            services={
                "task_service": self.task_service,
                "agent_definitions": self.agent_definitions,
                "context_state": self._context_state_for_session(handle.session_id),
                "runtime": self,
            },
            hooks=self._hooks,
        )
        async def run_one(call: ToolCall):
            await self._emit(handle, "tool_started", tool_call_id=call.tool_call_id, payload={"name": call.name})
            try:
                result = await self.tool_registry.execute(call, context)
            except asyncio.CancelledError:
                if handle._task is not None and handle._task.cancelling():
                    raise
                result = error_tool_result(
                    tool_call_id=call.tool_call_id,
                    tool_name=call.name,
                    error=ErrorPayload(code=ErrorCode.CANCELLED, message="tool execution cancelled"),
                )
            await self._persist_result_artifacts(handle, call, result)
            event_type = "tool_completed"
            if result.is_error:
                if getattr(result, "metadata", {}).get("permission_failed"):
                    event_type = "permission_failed"
                elif result.error and result.error.code == ErrorCode.PERMISSION_DENIED:
                    event_type = "tool_denied"
                else:
                    event_type = "tool_failed"
            await self._emit(
                handle,
                event_type,
                tool_call_id=call.tool_call_id,
                payload=_tool_event_payload(call.name, result),
            )
            return result
        return await _run_scheduled_tool_calls(calls, run_one, effective_tools)

    async def _execute_child_tool_calls(
        self,
        *,
        session_id: str,
        run_id: str,
        agent_id: str,
        agent_role: Literal["sub", "fork"],
        parent_agent_id: str,
        parent_run_id: str,
        calls: list[ToolCall],
        allowed_tool_names: set[str],
        stream: EventStream | None = None,
    ) -> list[Any]:
        context = self._child_tool_context(
            session_id=session_id,
            run_id=run_id,
            agent_id=agent_id,
            agent_role=agent_role,
            parent_agent_id=parent_agent_id,
            parent_run_id=parent_run_id,
            allowed_tool_names=allowed_tool_names,
        )

        async def run_one(call: ToolCall):
            await self._emit_child_run_event(
                stream=stream,
                session_id=session_id,
                agent_id=agent_id,
                run_id=run_id,
                event_type="tool_started",
                tool_call_id=call.tool_call_id,
                payload={"name": call.name},
            )
            result = await self.tool_registry.execute(call, context)
            child_handle = RunHandle(
                run_id=run_id,
                session_id=session_id,
                agent_id=agent_id,
                status=RunStatus.RUNNING,
                mode=RunMode.NORMAL,
                _runtime=self,
                _stream=EventStream(),
            )
            await self._persist_result_artifacts(child_handle, call, result)
            event_type = "tool_completed"
            if result.is_error:
                if getattr(result, "metadata", {}).get("permission_failed"):
                    event_type = "permission_failed"
                elif result.error and result.error.code == ErrorCode.PERMISSION_DENIED:
                    event_type = "tool_denied"
                else:
                    event_type = "tool_failed"
            await self._emit_child_run_event(
                stream=stream,
                session_id=session_id,
                agent_id=agent_id,
                run_id=run_id,
                event_type=event_type,
                tool_call_id=call.tool_call_id,
                payload=_tool_event_payload(call.name, result),
            )
            return result

        return await _run_scheduled_tool_calls(
            calls,
            run_one,
            [tool for tool in self._effective_tools(agent_role=agent_role) if tool.name in allowed_tool_names],
        )

    async def _execute_worker_tool_calls(
        self,
        *,
        session_id: str,
        run_id: str,
        agent_id: str,
        parent_agent_id: str,
        parent_run_id: str,
        calls: list[ToolCall],
        worker_scope: dict[str, Any],
        allowed_tool_names: set[str],
        stream: EventStream | None = None,
    ) -> list[Any]:
        context = self._worker_tool_context(
            session_id=session_id,
            run_id=run_id,
            agent_id=agent_id,
            parent_agent_id=parent_agent_id,
            parent_run_id=parent_run_id,
            worker_scope=worker_scope,
            allowed_tool_names=allowed_tool_names,
        )

        async def run_one(call: ToolCall):
            await self._emit_child_run_event(
                stream=stream,
                session_id=session_id,
                agent_id=agent_id,
                run_id=run_id,
                event_type="tool_started",
                tool_call_id=call.tool_call_id,
                payload={"name": call.name},
            )
            result = await self.tool_registry.execute(call, context)
            worker_handle = RunHandle(
                run_id=run_id,
                session_id=session_id,
                agent_id=agent_id,
                status=RunStatus.RUNNING,
                mode=RunMode.ORCHESTRATOR,
                _runtime=self,
                _stream=EventStream(),
            )
            await self._persist_result_artifacts(worker_handle, call, result)
            event_type = "tool_completed"
            if result.is_error:
                if getattr(result, "metadata", {}).get("permission_failed"):
                    event_type = "permission_failed"
                elif result.error and result.error.code == ErrorCode.PERMISSION_DENIED:
                    event_type = "tool_denied"
                else:
                    event_type = "tool_failed"
            await self._emit_child_run_event(
                stream=stream,
                session_id=session_id,
                agent_id=agent_id,
                run_id=run_id,
                event_type=event_type,
                tool_call_id=call.tool_call_id,
                payload=_tool_event_payload(call.name, result),
            )
            return result

        return await _run_scheduled_tool_calls(
            calls,
            run_one,
            [tool for tool in self._effective_tools(agent_role="worker") if tool.name in allowed_tool_names],
        )

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
        assert self.paths and self.config and self.artifacts
        return ToolExecutionContext(
            session_id=session_id,
            run_id=run_id,
            agent_id=agent_id,
            agent_role="worker",
            project_dir=self.paths.project_dir,
            home_dir=self.paths.home_dir,
            config=self.config,
            artifact_manager=self.artifacts,
            permission_callback=self.permission_callback,
            permission_cache=self._permission_caches[session_id],
            parent_agent_id=parent_agent_id,
            parent_run_id=parent_run_id,
            allowed_tool_names=allowed_tool_names or {tool.name for tool in self._effective_tools(agent_role="worker")},
            effective_tool_definitions={tool.name: tool for tool in self._effective_tools(agent_role="worker")},
            debug=self.debug,
            services={
                "task_service": self.task_service,
                "agent_definitions": self.agent_definitions,
                "context_state": self._context_state_for_session(session_id),
                "runtime": self,
                "worker_scope": worker_scope,
            },
            hooks=self._hooks,
        )

    def _child_tool_context(
        self,
        *,
        session_id: str,
        run_id: str,
        agent_id: str,
        agent_role: Literal["sub", "fork"],
        parent_agent_id: str,
        parent_run_id: str,
        allowed_tool_names: set[str] | None = None,
    ) -> ToolExecutionContext:
        assert self.paths and self.config and self.artifacts
        effective_tools = self._effective_tools(agent_role=agent_role)
        allowed_names = allowed_tool_names or {tool.name for tool in effective_tools}
        return ToolExecutionContext(
            session_id=session_id,
            run_id=run_id,
            agent_id=agent_id,
            agent_role=agent_role,
            project_dir=self.paths.project_dir,
            home_dir=self.paths.home_dir,
            config=self.config,
            artifact_manager=self.artifacts,
            permission_callback=self.permission_callback,
            permission_cache=self._permission_caches[session_id],
            parent_agent_id=parent_agent_id,
            parent_run_id=parent_run_id,
            allowed_tool_names=allowed_names,
            effective_tool_definitions={tool.name: tool for tool in effective_tools if tool.name in allowed_names},
            debug=self.debug,
            services={
                "task_service": self.task_service,
                "agent_definitions": self.agent_definitions,
                "context_state": self._context_state_for_session(session_id),
                "runtime": self,
            },
            hooks=self._hooks,
        )

    def _worker_step_summary(self, *, session_id: str, task_id: str, worker_run_id: str) -> dict[str, Any]:
        try:
            step = self.task_service.claimed_step_for_run(session_id, task_id, worker_run_id)
        except AgentCoreError:
            step = None
        if step is None:
            return {
                "claimed_step_id": None,
                "step_status": None,
                "step_result_summary": None,
                "no_step_claimed": True,
            }
        return _summary_from_step(step.model_dump(mode="json"))

    async def _persist_result_artifacts(self, handle: RunHandle, call: ToolCall, result) -> None:
        assert self.store and self.paths
        artifact_refs: list[tuple[str, str]] = []
        metadata = getattr(result, "metadata", None) or {}
        for key in ("stdout_artifact_id", "stderr_artifact_id", "output_artifact_id"):
            artifact_id = metadata.get(key)
            if not artifact_id:
                continue
            artifact_refs.append((str(artifact_id), key))
        for artifact_id in metadata.get("artifact_ids") or []:
            artifact_refs.append((str(artifact_id), "tool_output"))
        for block in getattr(result, "content", None) or []:
            if isinstance(block, ArtifactRefBlock):
                artifact_refs.append((block.artifact_id, block.summary or "tool_output"))
            elif isinstance(block, JsonBlock) and block.artifact_id:
                artifact_refs.append((block.artifact_id, block.summary or "tool_output"))
        seen: set[str] = set()
        for artifact_id, summary in artifact_refs:
            if artifact_id in seen:
                continue
            seen.add(artifact_id)
            artifact_dir = self.paths.home_dir / "sessions" / handle.session_id / "artifacts" / artifact_id
            files = list(artifact_dir.iterdir()) if artifact_dir.exists() else []
            path = files[0] if files else artifact_dir
            await self.store.add_artifact(
                artifact_id=artifact_id,
                session_id=handle.session_id,
                agent_id=handle.agent_id,
                run_id=handle.run_id,
                tool_call_id=call.tool_call_id,
                path=str(path),
                filename=path.name,
                mime_type=_guess_artifact_mime_type(path),
                size_bytes=path.stat().st_size if path.exists() and path.is_file() else None,
                summary=summary,
                metadata={"debug": False, "raw": False},
            )

    async def _persist_provider_debug_artifact(self, handle: RunHandle, model_event: Any, *, node_id: str | None) -> None:
        artifact = await self._persist_run_debug_artifact(
            session_id=handle.session_id,
            agent_id=handle.agent_id,
            run_id=handle.run_id,
            model_event=model_event,
            node_id=node_id,
        )
        if artifact is None:
            return
        await self._emit(
            handle,
            "provider_debug_artifact_created",
            level="debug",
            node_id=node_id,
            payload={
                "artifact_id": artifact.artifact_id,
                "provider": artifact.provider,
                "summary": artifact.summary,
            },
        )

    async def _persist_run_debug_artifact(
        self,
        *,
        session_id: str,
        agent_id: str | None,
        run_id: str | None,
        model_event: Any,
        node_id: str | None,
    ) -> Any | None:
        if not self.debug or self.artifacts is None or self.store is None:
            return None
        metadata = getattr(model_event, "metadata", {}) or {}
        raw_debug = metadata.get(RAW_DEBUG_METADATA_KEY)
        if raw_debug is None:
            return None
        provider = str(metadata.get("provider") or "provider")
        artifact = self.artifacts.write_text(
            session_id=session_id,
            text=json.dumps(raw_debug, ensure_ascii=False, indent=2, default=str),
            filename=f"{provider}_raw_model.json",
            mime_type="application/json",
            summary="Raw provider request/response",
        )
        await self.store.add_artifact(
            artifact_id=artifact.artifact_id,
            session_id=session_id,
            agent_id=agent_id,
            run_id=run_id,
            node_id=node_id,
            path=str(artifact.path),
            filename=artifact.filename,
            mime_type=artifact.mime_type,
            size_bytes=artifact.size_bytes,
            summary=artifact.summary,
            metadata={"debug": True, "raw": True, "provider": provider},
        )
        return _ProviderDebugArtifact(artifact_id=artifact.artifact_id, provider=provider, summary=artifact.summary)

    async def _run_stop_hooks(self, handle: RunHandle, *, end_node_id: str | None):
        assert self.paths and self.config
        from agent_core.hooks.runner import HookRunner

        return await HookRunner(self._hooks or []).run(
            event_type="stop",
            payload={
                "event_type": "Stop",
                "session_id": handle.session_id,
                "run_id": handle.run_id,
                "agent_id": handle.agent_id,
                "end_node_id": end_node_id,
            },
            cwd=self.paths.project_dir,
            timeout_ms=self.config.hooks.default_timeout_ms,
            env_allowlist=self.config.tools.env_allowlist,
        )

    async def _maybe_start_background_compact(self, handle: RunHandle) -> None:
        assert self.config and self.store
        if not self.config.compact.enabled or not self.config.compact.auto_background:
            return
        replay = await self.replay_session(handle.session_id)
        text = "\n".join(
            getattr(block, "text", "")
            for node in replay.nodes
            for block in node.content
            if getattr(block, "type", None) == "text"
        )
        estimated_tokens = max(len(text) // 4, 0)
        threshold = max(self.config.model.context_window - self.config.compact.reserve_tokens, self.config.compact.keep_recent_tokens)
        if estimated_tokens < threshold:
            return
        await self.store.add_event(
            make_event(
                session_id=handle.session_id,
                agent_id=handle.agent_id,
                run_id=handle.run_id,
                event_type="compact_pending",
                payload={"estimated_tokens": estimated_tokens, "threshold": threshold},
            )
        )
        asyncio.create_task(self.run_compact_agent(session_id=handle.session_id, reason="auto_background"))

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
        assert self.config and self.store
        if not self.config.compact.enabled or not self.config.compact.recovery_sync:
            return None
        budget = context_config.non_system_budget
        if budget is None:
            budget = max(self.config.model.context_window - self.config.model.max_output_tokens - _estimate_system_tokens(system_blocks), 0)
        latest_tokens = _estimate_message_tokens(
            ModelMessage(role=ModelRole.USER, content=user_node.content, node_type=user_node.node_type, metadata={"node_id": user_node.node_id})
        )
        if latest_tokens > max(budget, 0):
            return None
        replay = await self.replay_session(handle.session_id)
        source_node_ids = [
            node.node_id
            for node in replay.nodes
            if node.node_id != user_node.node_id
            and node.node_type not in {"compaction", "compact_input"}
            and node.role in {"user", "assistant", "tool"}
        ]
        if not source_node_ids:
            return None
        try:
            await self._emit(
                handle,
                "compact_pending",
                payload={"reason": "recovery_sync", "source_node_ids": source_node_ids},
            )
            result = await self.run_compact_agent(
                session_id=handle.session_id,
                source_node_ids=source_node_ids,
                reason="recovery_sync",
                first_kept_node_id=user_node.parent_id,
            )
        except AgentCoreError as exc:
            await self._emit(
                handle,
                "compact_failed",
                level="warning",
                payload={"reason": "recovery_sync", "code": exc.code.value, "message": exc.message},
            )
            return None
        if result.get("stale") or not result.get("compaction_node_id"):
            return None
        active_path = await self.store.get_node_path(str(result["compaction_node_id"]))
        recovered_messages = build_context_messages(active_path)
        task_board_message = _task_board_context_message(self.task_service, handle.session_id)
        if task_board_message is not None:
            recovered_messages.append(task_board_message)
        context_bundle = _apply_context_budget(
            messages=recovered_messages,
            system_blocks=system_blocks,
            context_config=context_config,
            model_config=self.config.model,
        )
        return {"messages": recovered_messages, "context_bundle": context_bundle, "end_node_id": result["compaction_node_id"]}

    async def _maybe_run_memory_extraction(self, handle: RunHandle, *, prompt_text: str) -> None:
        assert self.config and self.paths and self.store
        if not self.config.memory.enabled:
            return
        metadata = await self.store.session_metadata(handle.session_id)
        cursor_seq = int(metadata.get("memory_scan_node_seq") or 0)
        sources = await self.store.memory_source_nodes_since(handle.session_id, cursor_seq)
        if not sources:
            return
        reason = await self._memory_extraction_trigger_reason(
            session_id=handle.session_id,
            sources=sources,
            latest_user_text=prompt_text,
            max_pending_messages=max(1, self.config.memory.extract_every_messages),
            token_threshold=max(1, self.config.memory.extract_every_tokens),
        )
        if reason is None:
            self._schedule_memory_idle_extraction(session_id=handle.session_id)
            return
        await self._run_memory_extraction_for_sources(session_id=handle.session_id, cursor_seq=cursor_seq, sources=sources, reason=reason)

    async def _memory_extraction_trigger_reason(
        self,
        *,
        session_id: str,
        sources: list[tuple[int, Node]],
        latest_user_text: str,
        max_pending_messages: int,
        token_threshold: int,
    ) -> str | None:
        if await self._has_explicit_memory_intent(session_id=session_id, latest_user_text=latest_user_text):
            return "explicit"
        if len(sources) >= max_pending_messages:
            return "message_backlog"
        if _estimate_memory_source_tokens([node for _seq, node in sources]) >= token_threshold:
            return "token_backlog"
        return None

    async def _has_explicit_memory_intent(self, *, session_id: str, latest_user_text: str) -> bool:
        text = latest_user_text.strip()
        if not text:
            return False
        assert self.config
        model_config = resolve_model_config(self.config, self.config.memory.extract_model_profile)
        try:
            decision = await self._run_structured_json_model(
                model_config=model_config,
                schema=_MEMORY_INTENT_SCHEMA,
                purpose="memory_intent_classifier",
                session_id=session_id,
                system_text=(
                    "Decide whether the user's latest message explicitly asks the assistant to remember, "
                    "store, keep for future conversations, or always apply a user preference/profile fact. "
                    "Return JSON with explicit=true only for direct memory/storage intent. "
                    "Return explicit=false for ordinary questions, facts, status updates, or tasks that do not ask to remember."
                ),
                user_text=f"Latest user message:\n{text}",
                max_output_tokens=min(model_config.max_output_tokens, 256),
            )
        except AgentCoreError:
            return False
        return bool(decision.get("explicit"))

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
        request = _structured_json_request(
            model_config=model_config,
            schema=schema,
            purpose=purpose,
            session_id=session_id,
            system_text=system_text,
            user_text=user_text,
            max_output_tokens=max_output_tokens,
        )
        provider = self._provider_for_model(model_config)
        _ensure_provider_supports_request(provider, request)
        delta_parts: list[str] = []
        final_parts: list[str] = []
        tool_payload: dict[str, Any] | None = None
        async for model_event in provider.stream(request):
            if model_event.event_type == "model_text_delta" and model_event.text_delta:
                delta_parts.append(model_event.text_delta)
            elif model_event.event_type == "model_failed":
                error = model_event.error or ErrorPayload(code=ErrorCode.PROVIDER_ERROR, message=f"{purpose} provider failed")
                raise AgentCoreError(error.code, error.message, details=error.details)
            elif model_event.event_type == "model_completed":
                for call in model_event.tool_calls:
                    if call.name == "internal.structured_json":
                        tool_payload = dict(call.arguments)
                        break
                for block in model_event.content:
                    if getattr(block, "type", None) == "text":
                        final_parts.append(getattr(block, "text", ""))
                break
        if tool_payload is not None:
            return tool_payload
        text = ("".join(final_parts) if final_parts else "".join(delta_parts)).strip()
        parsed = _parse_json_object(text)
        if parsed is None:
            raise AgentCoreError(ErrorCode.PROVIDER_ERROR, f"{purpose} returned invalid JSON")
        return parsed

    async def _run_memory_extraction_for_sources(
        self,
        *,
        session_id: str,
        cursor_seq: int,
        sources: list[tuple[int, Node]],
        reason: str,
    ) -> None:
        assert self.config and self.paths and self.store
        source_node_ids = [node.node_id for _seq, node in sources]
        max_seq = max(seq for seq, _node in sources)
        await self.store.add_event(
            make_event(
                session_id=session_id,
                event_type="memory_extraction_started",
                payload={
                    "reason": reason,
                    "from_node_seq": cursor_seq + 1,
                    "to_node_seq": max_seq,
                    "source_node_ids": source_node_ids,
                },
            )
        )
        try:
            model_config = resolve_model_config(self.config, self.config.memory.extract_model_profile)
            text = await self._run_memory_extraction_model(
                session_id=session_id,
                model_config=model_config,
                source_nodes=[node for _seq, node in sources],
            )
            candidates = parse_memory_candidates(text)
            allowed_sources = set(source_node_ids)
            invalid_sources = sorted(
                {
                    source_node_id
                    for candidate in candidates
                    for source_node_id in candidate.source_node_ids
                    if source_node_id not in allowed_sources
                }
            )
            if invalid_sources:
                raise AgentCoreError(
                    ErrorCode.MEMORY_WRITE_FAILED,
                    f"memory candidate references source nodes outside extraction range: {invalid_sources}",
                )
            job = MemoryExtractionJob(
                home_dir=self.paths.home_dir,
                memory_dir=resolve_memory_dir(
                    self.config.memory.memory_dir,
                    home_dir=self.paths.home_dir,
                    project_dir=self.paths.project_dir,
                ),
                cursor=MemoryScanCursor(node_seq=cursor_seq),
                source_session_id=session_id,
            )
            result = job.apply(candidates, source_node_seq=max_seq)
            await self.store.update_session_metadata(
                session_id,
                {"memory_scan_node_seq": result.scan_cursor.node_seq},
            )
            await self.store.add_event(
                make_event(
                    session_id=session_id,
                    event_type="memory_extraction_completed",
                    payload={
                        "reason": reason,
                        "created_memory_ids": result.created,
                        "updated_memory_ids": result.updated,
                        "ignored_candidates": result.ignored,
                        "duplicate_decisions": result.duplicate,
                        "conflicts": [],
                        "source_node_ids": source_node_ids,
                        "files_changed": result.files_changed,
                        "scan_cursor": {"node_seq": result.scan_cursor.node_seq},
                    },
                )
            )
        except Exception as exc:
            code = getattr(exc, "code", ErrorCode.MEMORY_WRITE_FAILED)
            message = getattr(exc, "message", str(exc))
            await self.store.add_event(
                make_event(
                    session_id=session_id,
                    event_type="memory_extraction_failed",
                    level="error",
                    payload={
                        "reason": reason,
                        "code": str(code),
                        "message": redact_value(message[:500]),
                        "from_node_seq": cursor_seq + 1,
                        "to_node_seq": max_seq,
                        "source_node_ids": source_node_ids,
                    },
                )
            )

    def _schedule_memory_idle_extraction(self, *, session_id: str) -> None:
        assert self.config
        self._cancel_memory_idle_task(session_id)
        idle_seconds = max(float(self.config.memory.idle_seconds), 0.0)
        self._memory_idle_tasks[session_id] = asyncio.create_task(self._memory_idle_extraction_after_delay(session_id, idle_seconds))

    def _cancel_memory_idle_task(self, session_id: str) -> None:
        task = self._memory_idle_tasks.pop(session_id, None)
        if task is not None and not task.done():
            task.cancel()

    async def _memory_idle_extraction_after_delay(self, session_id: str, idle_seconds: float) -> None:
        try:
            await asyncio.sleep(idle_seconds)
            assert self.config and self.store
            metadata = await self.store.session_metadata(session_id)
            cursor_seq = int(metadata.get("memory_scan_node_seq") or 0)
            sources = await self.store.memory_source_nodes_since(session_id, cursor_seq)
            if sources:
                await self._run_memory_extraction_for_sources(
                    session_id=session_id,
                    cursor_seq=cursor_seq,
                    sources=sources,
                    reason="idle",
                )
        except asyncio.CancelledError:
            raise
        finally:
            current = self._memory_idle_tasks.get(session_id)
            if current is asyncio.current_task():
                self._memory_idle_tasks.pop(session_id, None)

    async def _run_memory_extraction_model(self, *, session_id: str, model_config: Any, source_nodes: list[Node]) -> str:
        assert self.paths
        source_text = _memory_extraction_source_text(source_nodes)
        assert self.config
        catalog = resolve_memory_dir(
            self.config.memory.memory_dir,
            home_dir=self.paths.home_dir,
            project_dir=self.paths.project_dir,
        ) / "MEMORY.md"
        catalog_text = catalog.read_text(encoding="utf-8", errors="replace") if catalog.exists() else "# Memory Catalog\n"
        payload = await self._run_structured_json_model(
            model_config=model_config,
            schema=_MEMORY_EXTRACTION_SCHEMA,
            purpose="memory_extraction",
            session_id=session_id,
            system_text=(
                "You are the Soong Agent memory extraction job. "
                "Decide whether new user-visible context should be written as long-term memory. "
                "For each memory, decision must be \"new\". "
                "Category must be exactly one of these strings: \"user\", \"feedback\", or \"reference\". "
                "Use category \"user\" for user profile, preferences, skills, and facts explicitly requested to remember. "
                "Do not output combined category strings such as \"user|reference\". "
                "Filename must be a local lowercase markdown filename ending in .md, for example backend_developer.md. "
                "source_node_ids must copy node IDs exactly from the source nodes. "
                "Use an empty memories array when nothing should be stored. "
                "Never store secrets, credentials, transient task state, plans, full transcripts, or command output."
            ),
            user_text=(
                f"Session: {session_id}\n\n"
                f"Existing MEMORY.md catalog:\n{catalog_text[:12000]}\n\n"
                f"New source nodes:\n{source_text}"
            ),
            max_output_tokens=model_config.max_output_tokens,
        )
        return json.dumps(payload, ensure_ascii=False)

    async def _run_observe_hook(
        self,
        *,
        event_type: str,
        session_id: str,
        agent_id: str | None,
        run_id: str | None,
        payload: dict[str, Any],
    ) -> None:
        assert self.paths and self.config and self.store
        if not self._hooks:
            return
        from agent_core.hooks.runner import HookRunner

        decision = await HookRunner(self._hooks).run(
            event_type=event_type,
            payload=payload,
            cwd=self.paths.project_dir,
            timeout_ms=self.config.hooks.default_timeout_ms,
            env_allowlist=self.config.tools.env_allowlist,
        )
        if decision.hook or decision.error or decision.logs or decision.denied:
            await self.store.add_event(
                make_event(
                    session_id=session_id,
                    agent_id=agent_id,
                    run_id=run_id,
                    event_type=f"{event_type}_hook_observed",
                    level="warning" if decision.error else "info",
                    payload={
                        "decision": decision.decision,
                        "reason": decision.reason,
                        "metadata": decision.metadata,
                        "logs": decision.logs,
                        "error": decision.error.model_dump(mode="json") if decision.error else None,
                    },
                )
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
        assert self.store
        event = make_event(
            session_id=handle.session_id,
            agent_id=handle.agent_id,
            run_id=handle.run_id,
            event_type=event_type,
            level=level,
            node_id=node_id,
            tool_call_id=tool_call_id,
            payload=payload,
        )
        stored = await self.store.add_event(event)
        await handle._stream.put(stored)
        return stored

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
        event = make_event(
            session_id=handle.session_id,
            agent_id=handle.agent_id,
            run_id=handle.run_id,
            event_type=event_type,
            level=level,
            node_id=node_id,
            tool_call_id=tool_call_id,
            payload=payload,
        )
        return event if handle._stream.put_nowait(event) else None

    def _open_child_run_stream(self, run_id: str) -> EventStream:
        stream = EventStream()
        self._child_run_streams[run_id] = stream
        return stream

    async def _close_child_run_stream(self, run_id: str) -> None:
        stream = self._child_run_streams.pop(run_id, None)
        if stream is not None:
            await stream.close()

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
        assert self.store
        stored = await self.store.add_event(
            make_event(
                session_id=session_id,
                agent_id=agent_id,
                run_id=run_id,
                event_type=event_type,
                level=level,
                node_id=node_id,
                tool_call_id=tool_call_id,
                payload=payload,
            )
        )
        if stream is not None and stream.has_consumer:
            await stream.put(stored)
        if mirror_handle is not None:
            await mirror_handle._stream.put(stored)
        return stored

    async def _emit_child_model_event(
        self,
        *,
        stream: EventStream,
        session_id: str,
        agent_id: str,
        run_id: str,
        event,
    ) -> None:
        if event.event_type == "model_started":
            await self._emit_child_run_event(
                stream=stream,
                session_id=session_id,
                agent_id=agent_id,
                run_id=run_id,
                event_type="model_started",
                payload=event.metadata,
            )
        elif event.event_type == "model_text_delta":
            if not stream.has_consumer:
                return
            transient = make_event(
                session_id=session_id,
                agent_id=agent_id,
                run_id=run_id,
                event_type="model_text_delta",
                payload={"text": event.text_delta or ""},
            )
            await stream.put(transient)
        elif event.event_type == "model_completed":
            await self._emit_child_run_event(
                stream=stream,
                session_id=session_id,
                agent_id=agent_id,
                run_id=run_id,
                event_type="model_completed",
                payload={
                    "stop_reason": event.stop_reason.value if event.stop_reason else None,
                    "tool_calls": [call.model_dump(mode="json") for call in event.tool_calls],
                    "usage": event.usage.model_dump(mode="json") if event.usage else None,
                },
            )
        elif event.event_type == "model_failed":
            error = event.error or ErrorPayload(code=ErrorCode.PROVIDER_ERROR, message="provider failed")
            await self._emit_child_run_event(
                stream=stream,
                session_id=session_id,
                agent_id=agent_id,
                run_id=run_id,
                event_type="model_failed",
                level="error",
                payload=error.model_dump(mode="json"),
            )

    async def _cancel_run(self, handle: RunHandle) -> CancelResult:
        if handle._queued:
            try:
                self._session_queues[handle.session_id].remove(handle)
            except ValueError:
                pass
            handle.status = RunStatus.CANCELLED
            handle._queued = False
            if self.store is not None:
                await self.store.update_run(
                    run_id=handle.run_id,
                    status=RunStatus.CANCELLED.value,
                    end_reason="aborted_tools",
                    error={"code": ErrorCode.CANCELLED.value, "message": "queued run cancelled", "reason": "cancelled"},
                )
            await self._emit(handle, "run_dequeued", payload={"cancelled": True})
            await self._emit(handle, "run_cancelled", payload={"queued": True})
            await handle._stream.close()
            return CancelResult(run_id=handle.run_id, status=handle.status, cancelled=True)
        if handle._task is not None and not handle._task.done():
            handle._task.cancel()
            try:
                await asyncio.wait_for(handle._task, timeout=(self.config.runtime.cancel_timeout_ms if self.config else 10000) / 1000)
            except asyncio.TimeoutError:
                await self._emit(handle, "cancel_timeout", level="warning")
        return CancelResult(run_id=handle.run_id, status=handle.status, cancelled=handle.status == RunStatus.CANCELLED)

    async def _start_next_queued(self, session_id: str) -> None:
        queue = self._session_queues.get(session_id)
        if not queue:
            return
        next_handle = queue.popleft()
        next_handle._queued = False
        self._session_active[session_id] = next_handle
        await self._emit(next_handle, "run_dequeued")
        next_handle._task = asyncio.create_task(self._run(next_handle, next_handle._message or UserMessage.from_text("")))

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
        stream = self._child_run_streams.get(child_run_id)
        if stream is not None and not stream.has_consumer:
            async for event in stream.iter():
                if debug or event.level != "debug":
                    yield event
            return
        replay = await self.replay_run(child_run_id)
        for event in replay.events:
            if debug or event.level != "debug":
                yield event

    async def _ensure_mcp_tools(
        self,
        *,
        handle: RunHandle | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
    ) -> None:
        assert self.store
        if self._mcp_discovered or self._mcp_manager is None:
            return
        self._mcp_discovered = True
        result = await self._mcp_manager.discover()
        register_mcp_tools(self.tool_registry, result.tools)
        for failure in result.failures:
            payload = {"server_id": failure.get("server_id"), "message": failure.get("message")}
            if handle is not None:
                await self._emit(handle, "mcp_server_failed", level="warning", payload=payload)
            elif session_id is not None:
                await self.store.add_event(
                    make_event(
                        session_id=session_id,
                        agent_id=agent_id,
                        run_id=run_id,
                        event_type="mcp_server_failed",
                        level="warning",
                        payload=payload,
                    )
                )


def _worker_prompt(
    *,
    instruction: str,
    task_id: str,
    worker_pool_id: str,
    allowed_step_ids: list[str] | None,
    dispatch_context: str | None,
    constraints: dict[str, Any] | None,
    expected_output_schema: dict[str, Any] | None,
) -> str:
    lines = [
        f"Task id: {task_id}",
        f"Worker pool: {worker_pool_id}",
        "Instruction:",
        instruction,
        "",
        "You may query and claim at most one ready step for this dispatch, then update only your claimed step.",
    ]
    if allowed_step_ids is not None:
        lines.extend(["Allowed step ids:", ", ".join(allowed_step_ids) or "(none)"])
    if dispatch_context:
        lines.extend(["", "Context:", dispatch_context])
    if constraints:
        lines.extend(["", "Constraints:", str(constraints)])
    if expected_output_schema:
        lines.extend(["", "Expected output schema:", json.dumps(expected_output_schema, ensure_ascii=False, indent=2)])
    return "\n".join(lines)


def _child_prompt(
    *,
    task: str,
    constraints: dict[str, Any] | None,
    expected_output_schema: dict[str, Any] | None,
) -> str:
    lines = ["Task:", task]
    if constraints:
        lines.extend(["", "Constraints:", str(constraints)])
    if expected_output_schema:
        lines.extend(["", "Expected output schema:", json.dumps(expected_output_schema, ensure_ascii=False, indent=2)])
    return "\n".join(lines)


def _child_timeout_seconds(config: AgentCoreConfig, timeout_ms: int | None) -> float:
    resolved = config.agents.default_child_timeout_ms if timeout_ms is None else int(timeout_ms)
    if resolved <= 0:
        raise AgentCoreError(ErrorCode.VALIDATION_ERROR, "timeout_ms must be greater than 0")
    return resolved / 1000


async def _collect_model_completion(
    provider: ProviderAdapter,
    request: ModelRequest,
    *,
    provider_failure_message: str,
    on_model_event: Callable[[Any], Awaitable[None]] | None = None,
    on_completed: Callable[[Any], Awaitable[None]] | None = None,
) -> tuple[Any, list[str]]:
    completed = None
    text_parts: list[str] = []
    async for model_event in provider.stream(request):
        if on_model_event is not None:
            await on_model_event(model_event)
        if model_event.event_type == "model_text_delta" and model_event.text_delta:
            text_parts.append(model_event.text_delta)
        elif model_event.event_type == "model_failed":
            error = model_event.error or ErrorPayload(code=ErrorCode.PROVIDER_ERROR, message=provider_failure_message)
            raise AgentCoreError(error.code, error.message, details=error.details)
        elif model_event.event_type == "model_completed":
            completed = model_event
            if on_completed is not None:
                await on_completed(model_event)
            break
    if completed is None:
        raise AgentCoreError(ErrorCode.PROVIDER_ERROR, "provider stream ended without model_completed")
    return completed, text_parts


def _validate_expected_output_schema(result_text: str, schema: dict[str, Any] | None) -> None:
    if not schema:
        return
    try:
        data: Any = json.loads(result_text)
    except json.JSONDecodeError:
        if _schema_allows_string(schema):
            data = result_text
        else:
            raise AgentCoreError(ErrorCode.VALIDATION_ERROR, "final output does not match expected_output_schema")
    _validate_json_value_against_schema(data, schema, path="$")


def _schema_allows_string(schema: dict[str, Any]) -> bool:
    schema_type = schema.get("type")
    if schema_type == "string":
        return True
    if isinstance(schema_type, list) and "string" in schema_type:
        return True
    return False


def _validate_json_value_against_schema(value: Any, schema: dict[str, Any], *, path: str) -> None:
    expected_type = schema.get("type")
    if isinstance(expected_type, list):
        errors: list[str] = []
        for item in expected_type:
            try:
                narrowed = dict(schema)
                narrowed["type"] = item
                _validate_json_value_against_schema(value, narrowed, path=path)
                errors.clear()
                break
            except AgentCoreError as exc:
                errors.append(exc.message)
        if errors:
            raise AgentCoreError(ErrorCode.VALIDATION_ERROR, errors[0])
        return
    if expected_type == "object":
        if not isinstance(value, dict):
            raise AgentCoreError(ErrorCode.VALIDATION_ERROR, f"{path} must be an object")
        required = schema.get("required") or []
        for key in required:
            if key not in value:
                raise AgentCoreError(ErrorCode.VALIDATION_ERROR, f"{path}.{key} is required")
        properties = schema.get("properties") or {}
        for key, prop_schema in properties.items():
            if key in value and isinstance(prop_schema, dict):
                _validate_json_value_against_schema(value[key], prop_schema, path=f"{path}.{key}")
        return
    if expected_type == "array":
        if not isinstance(value, list):
            raise AgentCoreError(ErrorCode.VALIDATION_ERROR, f"{path} must be an array")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                _validate_json_value_against_schema(item, item_schema, path=f"{path}[{index}]")
        return
    if expected_type == "string" and not isinstance(value, str):
        raise AgentCoreError(ErrorCode.VALIDATION_ERROR, f"{path} must be a string")
    if expected_type == "integer" and not (isinstance(value, int) and not isinstance(value, bool)):
        raise AgentCoreError(ErrorCode.VALIDATION_ERROR, f"{path} must be an integer")
    if expected_type == "number" and not ((isinstance(value, int) or isinstance(value, float)) and not isinstance(value, bool)):
        raise AgentCoreError(ErrorCode.VALIDATION_ERROR, f"{path} must be a number")
    if expected_type == "boolean" and not isinstance(value, bool):
        raise AgentCoreError(ErrorCode.VALIDATION_ERROR, f"{path} must be a boolean")
    if expected_type == "null" and value is not None:
        raise AgentCoreError(ErrorCode.VALIDATION_ERROR, f"{path} must be null")
    if schema.get("enum") is not None and value not in schema["enum"]:
        raise AgentCoreError(ErrorCode.VALIDATION_ERROR, f"{path} must be one of {schema['enum']}")


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


def _ensure_provider_supports_request(provider: ProviderAdapter, request: ModelRequest) -> None:
    supports_tools = getattr(provider, "supports_tools", True)
    if request.tools and supports_tools is False:
        raise AgentCoreError(
            ErrorCode.UNSUPPORTED_CAPABILITY,
            "provider/model does not support tool calls",
            details={
                "capability": "tools",
                "request_feature": "tools",
                "model": request.model,
            },
        )


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


def _artifact_selected_for_cleanup(
    *,
    artifact: dict[str, Any],
    metadata: dict[str, Any],
    include_all: bool,
    older_than: datetime | None,
    max_bytes: int | None,
) -> bool:
    if not (include_all or metadata.get("debug") is True or metadata.get("raw") is True):
        return False
    if older_than is not None:
        created_at = _parse_datetime(artifact.get("created_at"))
        if created_at is None or created_at >= older_than:
            return False
    if max_bytes is not None:
        size = artifact.get("size_bytes")
        if size is None or int(size) <= max_bytes:
            return False
    return True


def _artifact_cleanup_reason(metadata: dict[str, Any], *, include_all: bool, max_bytes: int | None) -> str:
    if metadata.get("debug") is True:
        return "debug_artifact_cleanup"
    if metadata.get("raw") is True:
        return "raw_artifact_cleanup"
    if max_bytes is not None:
        return "artifact_size_limit"
    if include_all:
        return "include_all"
    return "artifact_cleanup"


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not value:
        return None
    try:
        text = str(value)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _path_older_than(path: Path, older_than: datetime) -> bool:
    try:
        modified_at = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
    except OSError:
        return False
    if older_than.tzinfo is None:
        older_than = older_than.replace(tzinfo=UTC)
    return modified_at < older_than


def _path_mtime_iso(path: Path) -> str | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat()
    except OSError:
        return None


def _delete_artifact_path(path: Path) -> None:
    if path.is_file():
        path.unlink(missing_ok=True)
        parent = path.parent
        if parent.name.startswith("art_") and parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
        return
    if path.exists() and path.is_dir():
        for child in path.iterdir():
            if child.is_file():
                child.unlink(missing_ok=True)
        if not any(path.iterdir()):
            path.rmdir()


def _apply_context_budget(
    *,
    messages: list[ModelMessage],
    system_blocks: list[SystemBlock],
    context_config: ContextConfig,
    model_config: ModelConfig,
) -> dict[str, Any]:
    tokens_before = _estimate_model_messages_tokens(messages, system_blocks)
    message_tokens_before = sum(_estimate_message_tokens(message) for message in messages)
    budget = context_config.non_system_budget
    if budget is None:
        system_tokens = _estimate_system_tokens(system_blocks)
        budget = max(model_config.context_window - model_config.max_output_tokens - system_tokens, 0)
    if budget <= 0:
        tokens_after = _estimate_model_messages_tokens(messages, system_blocks)
        return {
            "messages": messages,
            "trimmed_node_ids": [],
            "budget": budget,
            "tokens_before_trim": tokens_before,
            "tokens_after_trim": tokens_after,
            "non_system_tokens_before_trim": message_tokens_before,
            "non_system_tokens_after_trim": message_tokens_before,
            "too_long": message_tokens_before > max(budget, 0),
        }

    protected_start = _protected_message_suffix_start(messages)
    retained_reversed: list[ModelMessage] = []
    retained_tokens = 0
    trimmed_node_ids: list[str] = []
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        message_tokens = _estimate_message_tokens(message)
        node_id = _message_node_id(message)
        if (
            retained_reversed
            and retained_tokens + message_tokens > budget
            and index < protected_start
            and _message_can_trim(message)
        ):
            if node_id is not None:
                trimmed_node_ids.append(node_id)
            continue
        retained_reversed.append(message)
        retained_tokens += message_tokens
    retained = list(reversed(retained_reversed))
    trimmed_node_ids.reverse()
    tokens_after = _estimate_model_messages_tokens(retained, system_blocks)
    message_tokens_after = sum(_estimate_message_tokens(message) for message in retained)
    return {
        "messages": retained,
        "trimmed_node_ids": trimmed_node_ids,
        "budget": budget,
        "tokens_before_trim": tokens_before,
        "tokens_after_trim": tokens_after,
        "non_system_tokens_before_trim": message_tokens_before,
        "non_system_tokens_after_trim": message_tokens_after,
        "too_long": message_tokens_after > budget,
    }


def _apply_system_block_budget(system_blocks: list[SystemBlock], dynamic_system_budget: int | None) -> dict[str, Any]:
    if dynamic_system_budget is None or dynamic_system_budget <= 0:
        return {"system_blocks": system_blocks, "trimmed_system_blocks": []}
    static_blocks = [block for block in system_blocks if not block.dynamic]
    dynamic_blocks = sorted((block for block in system_blocks if block.dynamic), key=lambda block: block.priority, reverse=True)
    retained_dynamic: list[SystemBlock] = []
    trimmed: list[dict[str, Any]] = []
    used = 0
    for block in dynamic_blocks:
        tokens = block.token_count if block.token_count is not None else max(len(block.content) // 4, 0)
        if used + tokens > dynamic_system_budget:
            trimmed.append(
                {
                    "block_id": block.block_id,
                    "source": block.source,
                    "priority": block.priority,
                    "estimated_tokens": tokens,
                }
            )
            continue
        retained_dynamic.append(block)
        used += tokens
    retained_ids = {id(block) for block in retained_dynamic}
    ordered = [block for block in system_blocks if not block.dynamic or id(block) in retained_ids]
    return {"system_blocks": ordered, "trimmed_system_blocks": trimmed}


def _protected_message_suffix_start(messages: list[ModelMessage]) -> int:
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.role == ModelRole.USER and message.node_type in {"message", "hook_context"}:
            return index
    return max(len(messages) - 1, 0)


def _message_can_trim(message: ModelMessage) -> bool:
    if isinstance(message.metadata, dict) and message.metadata.get("synthetic") is True:
        return False
    return message.node_type not in {"task_board", "compaction"}


def _message_node_id(message: ModelMessage) -> str | None:
    if isinstance(message.metadata, dict) and message.metadata.get("node_id") is not None:
        return str(message.metadata["node_id"])
    return None


def _context_build_report(
    messages: list[ModelMessage],
    system_blocks: list[SystemBlock],
    tools: list[ToolDefinition],
    *,
    trimmed_node_ids: list[str] | None = None,
    trimmed_system_blocks: list[dict[str, Any]] | None = None,
    budget: int | None = None,
    tokens_before_trim: int | None = None,
    tokens_after_trim: int | None = None,
    non_system_tokens_before_trim: int | None = None,
    non_system_tokens_after_trim: int | None = None,
    too_long: bool = False,
) -> dict[str, Any]:
    retained_node_ids = [
        str(message.metadata["node_id"])
        for message in messages
        if isinstance(message.metadata, dict) and message.metadata.get("node_id") is not None
    ]
    synthetic_messages = [
        {
            "node_type": message.node_type,
            "source": message.metadata.get("source"),
        }
        for message in messages
        if isinstance(message.metadata, dict) and message.metadata.get("synthetic") is True
    ]
    return {
        "model": None,
        "messages": len(messages),
        "tools": len(tools),
        "tool_names": [tool.name for tool in tools],
        "system_blocks": [
            {
                "block_id": block.block_id,
                "source": block.source,
                "dynamic": block.dynamic,
                "priority": block.priority,
            }
            for block in system_blocks
        ],
        "retained_node_ids": retained_node_ids,
        "trimmed_node_ids": trimmed_node_ids or [],
        "trimmed_system_blocks": trimmed_system_blocks or [],
        "synthetic_messages": synthetic_messages,
        "estimated_input_tokens": _estimate_model_messages_tokens(messages, system_blocks),
        "tokens_before_trim": tokens_before_trim,
        "tokens_after_trim": tokens_after_trim,
        "non_system_tokens_before_trim": non_system_tokens_before_trim,
        "non_system_tokens_after_trim": non_system_tokens_after_trim,
        "non_system_budget": budget,
        "too_long": too_long,
    }


def _estimate_model_messages_tokens(messages: list[ModelMessage], system_blocks: list[SystemBlock]) -> int:
    return _estimate_system_tokens(system_blocks) + sum(_estimate_message_tokens(message) for message in messages)


def _estimate_system_tokens(system_blocks: list[SystemBlock]) -> int:
    return sum(max(len(block.content) // 4, 0) for block in system_blocks)


def _estimate_message_tokens(message: ModelMessage) -> int:
    char_count = 0
    for block in message.content:
        if getattr(block, "type", None) == "text":
            char_count += len(getattr(block, "text", ""))
        elif getattr(block, "type", None) == "json":
            char_count += len(json.dumps(getattr(block, "data", None), ensure_ascii=False))
        elif getattr(block, "type", None) == "tool_call":
            char_count += len(json.dumps(getattr(block, "arguments", {}), ensure_ascii=False))
        elif getattr(block, "type", None) == "tool_result":
            char_count += len(json.dumps(getattr(block, "metadata", {}), ensure_ascii=False))
    return max(char_count // 4, 1)


def _structured_json_request(
    *,
    model_config: Any,
    schema: dict[str, Any],
    purpose: str,
    session_id: str,
    system_text: str,
    user_text: str,
    max_output_tokens: int,
) -> ModelRequest:
    provider = getattr(model_config, "provider", None)
    if provider == "anthropic":
        tool_name = "internal.structured_json"
        return ModelRequest(
            model=model_config.name,
            system=[
                SystemBlock(
                    block_id=purpose,
                    source=purpose,
                    content=system_text + " Use the provided tool exactly once with the JSON object.",
                    priority=900,
                    dynamic=True,
                )
            ],
            messages=[ModelMessage(role=ModelRole.USER, content=[TextBlock(text=user_text)], node_type=purpose)],
            tools=[
                ToolDefinition(
                    name=tool_name,
                    description="Return the structured JSON object for this classifier/selector.",
                    input_schema=schema,
                    permission="readonly",
                    tags={"internal", "structured_json"},
                )
            ],
            tool_choice={"type": "tool", "name": tool_name},
            temperature=model_config.temperature,
            max_output_tokens=max_output_tokens,
            metadata={"session_id": session_id, "purpose": purpose},
        )
    return ModelRequest(
        model=model_config.name,
        system=[
            SystemBlock(
                block_id=purpose,
                source=purpose,
                content=system_text + "\nReturn only one JSON object matching this schema:\n" + json.dumps(schema, ensure_ascii=False),
                priority=900,
                dynamic=True,
            )
        ],
        messages=[ModelMessage(role=ModelRole.USER, content=[TextBlock(text=user_text)], node_type=purpose)],
        tools=[],
        temperature=model_config.temperature,
        max_output_tokens=max_output_tokens,
        provider_options=_structured_json_provider_options(model_config, schema),
        metadata={"session_id": session_id, "purpose": purpose},
    )


def _structured_json_provider_options(model_config: Any, schema: dict[str, Any]) -> dict[str, Any]:
    provider = getattr(model_config, "provider", None)
    if provider == "ollama":
        return {"ollama": {"format": schema}}
    if provider == "openai-compatible":
        return {"openai-compatible": {"response_format": {"type": "json_schema", "json_schema": {"name": "structured_json", "schema": schema}}}}
    return {}


def _estimate_memory_source_tokens(nodes: list[Node]) -> int:
    char_count = 0
    for node in nodes:
        for block in node.content:
            if getattr(block, "type", None) == "text":
                char_count += len(getattr(block, "text", ""))
            elif getattr(block, "type", None) == "json":
                char_count += len(json.dumps(getattr(block, "data", None), ensure_ascii=False))
    return max(char_count // 4, 0)


async def _run_scheduled_tool_calls(calls: list[ToolCall], run_one: Any, definitions: list[ToolDefinition]) -> list[Any]:
    by_name = {definition.name: definition for definition in definitions}
    results: list[Any] = []
    batch: list[ToolCall] = []

    async def flush_readonly_batch() -> None:
        nonlocal batch
        if not batch:
            return
        results.extend(await asyncio.gather(*(run_one(call) for call in batch)))
        batch = []

    for call in calls:
        definition = by_name.get(call.name)
        is_write = definition is None or definition.permission == "write" or "write" in definition.tags
        if not is_write:
            batch.append(call)
            continue
        await flush_readonly_batch()
        result = await run_one(call)
        results.append(result)
        if getattr(result, "is_error", False):
            break
    await flush_readonly_batch()
    return results


def _agent_definition_body_with_default(
    definitions: AgentDefinitionRegistry,
    definition: AgentDefinition,
    *,
    default_id: str,
) -> str:
    if definition.body:
        return definition.body
    default_definition = definitions.get(default_id)
    return default_definition.body if default_definition is not None else ""


def _guess_artifact_mime_type(path: Path) -> str:
    if path.suffix == ".json":
        return "application/json"
    return "text/plain"


def _compact_input(nodes: list[Node]) -> str:
    lines = ["Summarize the following active context for compaction.", ""]
    for node in nodes:
        text_parts = [getattr(block, "text", "") for block in node.content if getattr(block, "type", None) == "text"]
        if not text_parts:
            continue
        lines.append(f"[{node.node_id}] {node.role}/{node.node_type}:")
        lines.append("\n".join(text_parts))
        lines.append("")
    return "\n".join(lines).strip()


def _memory_extraction_source_text(nodes: list[Node]) -> str:
    lines: list[str] = []
    for node in nodes:
        text_parts = [getattr(block, "text", "") for block in node.content if getattr(block, "type", None) == "text"]
        if not text_parts:
            continue
        lines.append(f"<node id=\"{node.node_id}\" role=\"{node.role}\" type=\"{node.node_type}\">")
        lines.append("\n".join(text_parts)[:8000])
        lines.append("</node>")
        lines.append("")
    return "\n".join(lines).strip()


def _memory_frontmatter_candidates(memory_root: Path) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    if not memory_root.exists():
        return candidates
    for path in sorted(memory_root.glob("*/*.md")):
        if path.parent.name not in {"user", "feedback", "reference"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        metadata = _simple_frontmatter(text)
        candidates.append(
            {
                "path": str(path),
                "relative_path": str(path.relative_to(memory_root)),
                "id": metadata.get("id") or path.stem,
                "category": metadata.get("category") or path.parent.name,
                "summary": metadata.get("summary") or path.stem,
                "tags": _frontmatter_list(metadata.get("tags")),
                "excerpt": _memory_candidate_excerpt(text),
            }
        )
    return candidates


def _memory_candidate_selector_line(item: dict[str, Any]) -> str:
    tags = item.get("tags") or []
    tag_text = f" tags={', '.join(str(tag) for tag in tags)}" if tags else ""
    excerpt = str(item.get("excerpt") or "").replace("\n", " ").strip()
    excerpt_text = f" excerpt={excerpt[:800]}" if excerpt else ""
    return f"- {item['relative_path']} [{item.get('category')}] id={item.get('id')} summary={item.get('summary')}{tag_text}{excerpt_text}"


def _memory_candidate_excerpt(text: str) -> str:
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end != -1:
            rest = text.find("\n", end + 4)
            text = text[rest + 1 :] if rest != -1 else ""
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())[:1200]


def _frontmatter_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            return [item.strip().strip('"').strip("'") for item in text[1:-1].split(",") if item.strip()]
        return [text]
    return []


def _parse_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _simple_frontmatter(text: str) -> dict[str, Any]:
    if not text.startswith("---\n"):
        return {}
    end = text.find("\n---", 4)
    if end == -1:
        return {}
    metadata: dict[str, Any] = {}
    current_key: str | None = None
    for line in text[4:end].splitlines():
        stripped = line.strip()
        if current_key and stripped.startswith("- "):
            value = stripped[2:].strip().strip('"').strip("'")
            current = metadata.setdefault(current_key, [])
            if isinstance(current, list):
                current.append(value)
            continue
        current_key = None
        if ":" not in line or line.startswith(" "):
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value:
            metadata[key] = value.strip('"').strip("'")
        else:
            metadata[key] = []
            current_key = key
    return metadata
