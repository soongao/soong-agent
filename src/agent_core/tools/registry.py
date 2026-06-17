from __future__ import annotations

import uuid
from typing import Any

from agent_core.artifacts.redaction import redact_value
from agent_core.errors.codes import ErrorCode
from agent_core.hooks.matcher import hook_matches
from agent_core.hooks.runner import HookRunner
from agent_core.types.common import ErrorPayload
from agent_core.types.permissions import PermissionDecisionKind, PermissionRequest
from agent_core.types.tools import ToolCall, ToolDefinition, ToolResult, error_tool_result, normalize_tool_result
from agent_core.permissions import needs_permission
from agent_core.tools.execution import ToolExecutionContext, ToolHandler
from agent_core.tools.hooks import hook_summary, run_post_tool_hooks
from agent_core.tools.schema import validate_arguments, validate_schema_subset
from agent_core.tools.scope import (
    effective_definition_for_call,
    network_host,
    network_host_allowed,
    sensitive_read_target_hit,
    target_path as resolve_target_path,
    target_scope,
)


class ToolRegistry:
    def __init__(self) -> None:
        self._definitions: dict[str, ToolDefinition] = {}
        self._handlers: dict[str, ToolHandler] = {}

    def register_tool(self, definition: ToolDefinition, handler: ToolHandler) -> None:
        if definition.name in self._definitions:
            raise ValueError(f"duplicate tool canonical name: {definition.name}")
        validate_schema_subset(definition.input_schema)
        self._definitions[definition.name] = definition
        self._handlers[definition.name] = handler

    def get(self, name: str) -> ToolDefinition | None:
        return self._definitions.get(name)

    def list_definitions(self) -> list[ToolDefinition]:
        return list(self._definitions.values())

    def names(self) -> set[str]:
        return set(self._definitions)

    async def execute(self, call: ToolCall, context: ToolExecutionContext) -> ToolResult:
        if context.allowed_tool_names is not None and call.name not in context.allowed_tool_names:
            return error_tool_result(
                tool_call_id=call.tool_call_id,
                tool_name=call.name,
                error=ErrorPayload(code=ErrorCode.TOOL_NOT_AVAILABLE, message=f"tool not available in this run: {call.name}"),
            )
        definition = context.effective_tool_definitions.get(call.name) if context.effective_tool_definitions is not None else self._definitions.get(call.name)
        handler = self._handlers.get(call.name)
        if definition is None or handler is None:
            return error_tool_result(
                tool_call_id=call.tool_call_id,
                tool_name=call.name,
                error=ErrorPayload(code=ErrorCode.TOOL_NOT_AVAILABLE, message=f"tool not available: {call.name}"),
            )
        try:
            validate_arguments(call.arguments, definition.input_schema)
        except ValueError as exc:
            return error_tool_result(
                tool_call_id=call.tool_call_id,
                tool_name=call.name,
                error=ErrorPayload(code=ErrorCode.VALIDATION_ERROR, message=str(exc)),
            )

        definition = effective_definition_for_call(definition, call.arguments)
        target_path = resolve_target_path(call.name, call.arguments, context)
        network_host_value = network_host(call.arguments) if "network" in definition.tags else None
        target_scope_value = target_scope(
            call.name,
            call.arguments,
            context,
            target_path=target_path,
            network_host=network_host_value,
            is_network="network" in definition.tags,
        )
        hook_decision = await HookRunner(context.hooks or []).run(
            event_type="tool_started",
            tool_name=definition.name,
            tags=set(definition.tags),
            target_path=str(target_path) if target_path is not None else None,
            payload={
                "event_type": "PreToolUse",
                "tool_name": definition.name,
                "arguments": call.arguments,
                "session_id": context.session_id,
                "run_id": context.run_id,
                "agent_id": context.agent_id,
            },
            cwd=context.effective_cwd,
            timeout_ms=context.config.hooks.default_timeout_ms,
            env_allowlist=context.config.tools.env_allowlist,
        )
        hook_summary_value = hook_summary(hook_decision)
        if hook_decision.denied:
            return error_tool_result(
                tool_call_id=call.tool_call_id,
                tool_name=definition.name,
                error=ErrorPayload(code=ErrorCode.PERMISSION_DENIED, message=hook_decision.reason or "hook denied tool"),
                metadata={"hook_summary": hook_summary_value or {"action": "deny", "reason": hook_decision.reason, "metadata": hook_decision.metadata}},
            )
        if context.permission_cache and context.permission_cache.is_allowed(
            tool_name=definition.name, target_scope=target_scope_value
        ):
            permitted = True
        else:
            permitted = not needs_permission(
                permission=definition.permission,
                tags=definition.tags,
                target=target_path,
                config=context.config,
            )
            if permitted and sensitive_read_target_hit(call.name, call.arguments, target_path, context):
                permitted = False
            network_denied_by_policy = False
            if "network" in definition.tags:
                network_policy = context.config.permissions.network_policy
                can_auto_allow_network = definition.permission != "write" and "dangerous" not in definition.tags
                if can_auto_allow_network and network_host_allowed(network_host_value, network_policy.allowed_hosts, network_policy.allowed_domains):
                    permitted = True
                elif can_auto_allow_network and network_policy.default == "allow":
                    permitted = True
                elif network_policy.default == "deny":
                    permitted = False
                    network_denied_by_policy = True
            if not permitted and network_denied_by_policy:
                return error_tool_result(
                    tool_call_id=call.tool_call_id,
                    tool_name=definition.name,
                    error=ErrorPayload(code=ErrorCode.PERMISSION_DENIED, message="network policy denied tool"),
                    metadata={"permission_denied": True, "network_host": network_host_value},
                )
            if (
                not permitted
                and context.permission_callback is None
                and definition.permission == "write"
                and "dangerous" not in definition.tags
                and "network" not in definition.tags
                and context.config.permissions.write_without_callback == "allow"
            ):
                permitted = True
            if not permitted and context.permission_callback is not None:
                request = PermissionRequest(
                    request_id=f"perm_{uuid.uuid4().hex}",
                    session_id=context.session_id,
                    agent_id=context.agent_id,
                    run_id=context.run_id,
                    parent_agent_id=context.parent_agent_id,
                    parent_run_id=context.parent_run_id,
                    agent_role=context.agent_role,
                    tool_name=definition.name,
                    permission=definition.permission,
                    tags=sorted(definition.tags),
                    args_summary=str(redact_value(call.arguments))[:1000],
                    target_scope=target_scope_value,
                    cwd=str(context.effective_cwd),
                    env_summary={},
                    network_host=network_host_value,
                    dangerous="dangerous" in definition.tags,
                    hook_summary=hook_summary_value,
                    suggested_decision=PermissionDecisionKind.DENY,
                )
                try:
                    decision = await context.permission_callback(request)
                except Exception as exc:
                    return error_tool_result(
                        tool_call_id=call.tool_call_id,
                        tool_name=definition.name,
                        error=ErrorPayload(code=ErrorCode.PERMISSION_DENIED, message="permission callback failed"),
                        metadata={"permission_failed": True, "permission_error": str(exc)},
                    )
                permitted = decision.decision in {
                    PermissionDecisionKind.ALLOW_ONCE,
                    PermissionDecisionKind.ALLOW_FOR_SESSION,
                }
                if (
                    decision.decision == PermissionDecisionKind.ALLOW_FOR_SESSION
                    and context.config.permissions.allow_for_session_enabled
                    and context.permission_cache is not None
                ):
                    context.permission_cache.allow(tool_name=definition.name, target_scope=target_scope_value)

        if not permitted:
            return error_tool_result(
                tool_call_id=call.tool_call_id,
                tool_name=definition.name,
                error=ErrorPayload(code=ErrorCode.PERMISSION_DENIED, message="permission denied"),
                metadata={"permission_denied": True},
            )

        for hook in context.hooks or []:
            hook_with_tags = dict(hook)
            hook_with_tags["_tool_tags"] = sorted(definition.tags)
            if hook_matches(hook=hook_with_tags, event_type="tool_started", tool_name=definition.name, target_path=str(target_path) if target_path is not None else None):
                action = hook.get("action") or hook.get("decision")
                if action == "deny":
                    return error_tool_result(
                        tool_call_id=call.tool_call_id,
                        tool_name=definition.name,
                        error=ErrorPayload(code=ErrorCode.PERMISSION_DENIED, message=hook.get("reason") or "hook denied tool"),
                        metadata={"hook_summary": {"action": "deny"}},
                    )

        try:
            value = await handler(context, call.arguments)
            result = normalize_tool_result(value, tool_call_id=call.tool_call_id, tool_name=definition.name)
            if result.tool_call_id != call.tool_call_id or result.tool_name != definition.name:
                result = result.model_copy(update={"tool_call_id": call.tool_call_id, "tool_name": definition.name})
            if hook_summary_value and "hook_summary" not in result.metadata:
                result = result.model_copy(update={"metadata": {**result.metadata, "hook_summary": hook_summary_value}})
            return await run_post_tool_hooks(context=context, call=call, definition=definition, target_path=target_path, result=result)
        except Exception as exc:
            code = getattr(exc, "code", ErrorCode.INTERNAL_ERROR)
            message = getattr(exc, "message", str(exc))
            result = error_tool_result(
                tool_call_id=call.tool_call_id,
                tool_name=definition.name,
                error=ErrorPayload(
                    code=code,
                    message=message,
                    retryable=bool(getattr(exc, "retryable", False)),
                    details=dict(getattr(exc, "details", {}) or {}),
                ),
                metadata={"hook_summary": hook_summary_value} if hook_summary_value else None,
            )
            return await run_post_tool_hooks(context=context, call=call, definition=definition, target_path=target_path, result=result)
