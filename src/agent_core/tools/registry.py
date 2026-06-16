from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from agent_core.artifacts.redaction import redact_value
from agent_core.errors.codes import ErrorCode
from agent_core.hooks.matcher import hook_matches
from agent_core.hooks.runner import HookRunner
from agent_core.types.common import ErrorPayload
from agent_core.types.permissions import PermissionDecisionKind, PermissionRequest
from agent_core.types.tools import ToolCall, ToolDefinition, ToolResult, error_tool_result, normalize_tool_result
from agent_core.permissions import is_sensitive_path, needs_permission
from agent_core.tools.execution import ToolExecutionContext, ToolHandler


class ToolRegistry:
    def __init__(self) -> None:
        self._definitions: dict[str, ToolDefinition] = {}
        self._handlers: dict[str, ToolHandler] = {}

    def register_tool(self, definition: ToolDefinition, handler: ToolHandler) -> None:
        if definition.name in self._definitions:
            raise ValueError(f"duplicate tool canonical name: {definition.name}")
        _validate_schema_subset(definition.input_schema)
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
            _validate_arguments(call.arguments, definition.input_schema)
        except ValueError as exc:
            return error_tool_result(
                tool_call_id=call.tool_call_id,
                tool_name=call.name,
                error=ErrorPayload(code=ErrorCode.VALIDATION_ERROR, message=str(exc)),
            )

        target_path = _target_path(call.name, call.arguments, context)
        network_host = _network_host(call.arguments) if "network" in definition.tags else None
        target_scope = _target_scope(
            call.name,
            call.arguments,
            context,
            target_path=target_path,
            network_host=network_host,
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
        hook_summary = _hook_summary(hook_decision)
        if hook_decision.denied:
            return error_tool_result(
                tool_call_id=call.tool_call_id,
                tool_name=definition.name,
                error=ErrorPayload(code=ErrorCode.PERMISSION_DENIED, message=hook_decision.reason or "hook denied tool"),
                metadata={"hook_summary": hook_summary or {"action": "deny", "reason": hook_decision.reason, "metadata": hook_decision.metadata}},
            )
        if context.permission_cache and context.permission_cache.is_allowed(
            tool_name=definition.name, target_scope=target_scope
        ):
            permitted = True
        else:
            permitted = not needs_permission(
                permission=definition.permission,
                tags=definition.tags,
                target=target_path,
                config=context.config,
            )
            if permitted and _sensitive_read_target_hit(call.name, call.arguments, target_path, context):
                permitted = False
            network_denied_by_policy = False
            if "network" in definition.tags:
                network_policy = context.config.permissions.network_policy
                can_auto_allow_network = definition.permission != "write" and "dangerous" not in definition.tags
                if can_auto_allow_network and _network_host_allowed(network_host, network_policy.allowed_hosts, network_policy.allowed_domains):
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
                    metadata={"permission_denied": True, "network_host": network_host},
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
                    target_scope=target_scope,
                    cwd=str(context.effective_cwd),
                    env_summary={},
                    network_host=network_host,
                    dangerous="dangerous" in definition.tags,
                    hook_summary=hook_summary,
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
                    context.permission_cache.allow(tool_name=definition.name, target_scope=target_scope)

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
            return await _run_post_tool_hooks(context=context, call=call, definition=definition, target_path=target_path, result=result)
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
            )
            return await _run_post_tool_hooks(context=context, call=call, definition=definition, target_path=target_path, result=result)


async def _run_post_tool_hooks(
    *,
    context: ToolExecutionContext,
    call: ToolCall,
    definition: ToolDefinition,
    target_path: Any,
    result: ToolResult,
) -> ToolResult:
    post_decision = await HookRunner(context.hooks or []).run(
        event_type="tool_completed",
        tool_name=definition.name,
        tags=set(definition.tags),
        target_path=str(target_path) if target_path is not None else None,
        payload={
            "event_type": "PostToolUse",
            "tool_name": definition.name,
            "arguments": call.arguments,
            "result": result.model_dump(mode="json"),
            "session_id": context.session_id,
            "run_id": context.run_id,
            "agent_id": context.agent_id,
        },
        cwd=context.effective_cwd,
        timeout_ms=context.config.hooks.default_timeout_ms,
        env_allowlist=context.config.tools.env_allowlist,
    )
    if post_decision.hook or post_decision.error:
        metadata = dict(result.metadata)
        metadata["post_hook_summary"] = {
            "decision": post_decision.decision,
            "reason": post_decision.reason,
            "metadata": post_decision.metadata,
            "error": post_decision.error.model_dump(mode="json") if post_decision.error else None,
        }
        return result.model_copy(update={"metadata": metadata})
    return result


def _target_scope(
    tool_name: str,
    arguments: dict[str, Any],
    context: ToolExecutionContext,
    *,
    target_path: Any,
    network_host: str | None,
    is_network: bool = False,
) -> str | None:
    if network_host is not None:
        return f"network:{network_host}"
    if is_network:
        return "network:unknown"
    if "path" in arguments or tool_name == "code.search":
        return str(target_path) if target_path is not None else None
    if "argv" in arguments:
        argv = arguments.get("argv") or []
        executable = argv[0] if isinstance(argv, list) and argv else ""
        cwd = _resolve_scope_path(arguments.get("cwd") or str(context.project_dir), context)
        return f"{executable}:{cwd}"
    if "cwd" in arguments:
        return str(_resolve_scope_path(arguments["cwd"], context))
    return None


def _target_path(tool_name: str, arguments: dict[str, Any], context: ToolExecutionContext):
    if "path" in arguments and arguments["path"] is not None:
        return _resolve_scope_path(arguments["path"], context)
    if tool_name == "code.search":
        return context.project_dir.resolve()
    return None


def _sensitive_read_target_hit(
    tool_name: str,
    arguments: dict[str, Any],
    target_path: Any,
    context: ToolExecutionContext,
) -> bool:
    if tool_name not in {"code.read_file", "code.list_dir", "code.search"} or target_path is None:
        return False
    path = Path(str(target_path))
    patterns = context.config.tools.sensitive_paths
    if is_sensitive_path(path, patterns=patterns):
        return True
    if not path.is_dir():
        return False
    if tool_name == "code.search":
        return _directory_contains_sensitive_path(path, patterns=patterns, recursive=True)
    if tool_name == "code.list_dir":
        return _directory_contains_sensitive_path(path, patterns=patterns, recursive=bool(arguments.get("recursive", False)))
    return False


def _directory_contains_sensitive_path(path: Path, *, patterns: list[str], recursive: bool) -> bool:
    try:
        iterator = path.rglob("*") if recursive else path.iterdir()
        for child in iterator:
            if is_sensitive_path(child, patterns=patterns):
                return True
    except OSError:
        return False
    return False


def _resolve_scope_path(value: Any, context: ToolExecutionContext):
    from pathlib import Path

    raw = Path(str(value)).expanduser()
    if not raw.is_absolute():
        raw = context.effective_cwd / raw
    return raw.resolve()


def _hook_summary(decision: Any) -> dict[str, Any] | None:
    if not decision.hook and not decision.error and not decision.metadata and not decision.logs and not decision.denied:
        return None
    return {
        "action": "deny" if decision.denied else "allow",
        "reason": decision.reason,
        "metadata": decision.metadata,
        "logs": decision.logs,
        "error": decision.error.model_dump(mode="json") if decision.error else None,
    }


def _network_host(arguments: dict[str, Any]) -> str | None:
    for key in ("url", "uri", "endpoint", "base_url", "host", "hostname", "network_host"):
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            return _host_from_string(value)
    for value in arguments.values():
        if isinstance(value, str) and "://" in value:
            return _host_from_string(value)
    return None


def _host_from_string(value: str) -> str | None:
    text = value.strip()
    parsed = urlparse(text if "://" in text else f"//{text}")
    host = parsed.hostname
    return host.lower() if host else None


def _network_host_allowed(host: str | None, allowed_hosts: list[str], allowed_domains: list[str]) -> bool:
    if host is None:
        return False
    normalized = host.lower().rstrip(".")
    if normalized in {item.lower().rstrip(".") for item in allowed_hosts}:
        return True
    for domain in allowed_domains:
        normalized_domain = domain.lower().lstrip(".").rstrip(".")
        if normalized == normalized_domain or normalized.endswith(f".{normalized_domain}"):
            return True
    return False


def _validate_schema_subset(schema: Any, *, path: str = "$") -> None:
    if not isinstance(schema, dict):
        raise ValueError(f"tool schema at {path} must be an object")
    unsupported = {"$ref", "oneOf", "anyOf", "allOf", "if", "then", "else", "patternProperties"}
    for key in unsupported:
        if key in schema:
            raise ValueError(f"unsupported tool schema keyword at {path}: {key}")
    schema_type = schema.get("type")
    allowed_types = {"object", "string", "number", "integer", "boolean", "array", "null"}
    if isinstance(schema_type, list):
        invalid = [item for item in schema_type if item not in allowed_types]
        if invalid:
            raise ValueError(f"unsupported schema type at {path}: {invalid[0]}")
    elif schema_type is not None and schema_type not in allowed_types:
        raise ValueError(f"unsupported schema type at {path}: {schema_type}")
    properties = schema.get("properties") or {}
    if properties and not isinstance(properties, dict):
        raise ValueError(f"schema properties at {path} must be an object")
    for name, child in properties.items():
        _validate_schema_subset(child, path=f"{path}.properties.{name}")
    items = schema.get("items")
    if isinstance(items, dict):
        _validate_schema_subset(items, path=f"{path}.items")
    elif isinstance(items, list):
        raise ValueError(f"tuple-style array schemas are not supported at {path}.items")


def _validate_arguments(arguments: Any, schema: dict[str, Any], *, path: str = "$") -> None:
    schema_type = schema.get("type")
    allowed_types = schema_type if isinstance(schema_type, list) else ([schema_type] if schema_type is not None else [])
    if "null" in allowed_types and arguments is None:
        return
    non_null_types = [item for item in allowed_types if item != "null"]
    if non_null_types and not any(_matches_json_type(arguments, item) for item in non_null_types):
        raise ValueError(f"{path} must be {_type_label(non_null_types)}")
    if schema.get("enum") is not None and arguments not in schema["enum"]:
        raise ValueError(f"{path} must be one of {schema['enum']}")
    effective_type = non_null_types[0] if non_null_types else None
    if effective_type == "object" or (effective_type is None and isinstance(arguments, dict)):
        if not isinstance(arguments, dict):
            raise ValueError(f"{path} must be object")
        required = schema.get("required") or []
        for key in required:
            if key not in arguments:
                raise ValueError(f"{path}.{key} is required")
        properties = schema.get("properties") or {}
        if properties and schema.get("additionalProperties", False) is not True:
            unknown = sorted(key for key in arguments if key not in properties)
            if unknown:
                raise ValueError(f"{path} contains unknown field: {unknown[0]}")
        for key, value in arguments.items():
            if key in properties:
                _validate_arguments(value, properties[key], path=f"{path}.{key}")
    elif effective_type == "array" or (effective_type is None and isinstance(arguments, list)):
        if not isinstance(arguments, list):
            raise ValueError(f"{path} must be array")
        items_schema = schema.get("items")
        if isinstance(items_schema, dict):
            for index, value in enumerate(arguments):
                _validate_arguments(value, items_schema, path=f"{path}[{index}]")


def _matches_json_type(value: Any, schema_type: str) -> bool:
    if schema_type == "object":
        return isinstance(value, dict)
    if schema_type == "array":
        return isinstance(value, list)
    if schema_type == "string":
        return isinstance(value, str)
    if schema_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if schema_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if schema_type == "boolean":
        return isinstance(value, bool)
    if schema_type == "null":
        return value is None
    return True


def _type_label(types: list[str]) -> str:
    return " or ".join(types)
