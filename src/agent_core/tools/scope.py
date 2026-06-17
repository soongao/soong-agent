from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from agent_core.permissions import is_sensitive_path
from agent_core.tools.execution import ToolExecutionContext
from agent_core.types.tools import ToolDefinition


def effective_definition_for_call(definition: ToolDefinition, arguments: dict[str, Any]) -> ToolDefinition:
    if definition.name != "code.run_command" or not is_readonly_command(arguments):
        return definition
    tags = set(definition.tags)
    tags.discard("dangerous")
    tags.add("readonly")
    return definition.model_copy(update={"permission": "readonly", "tags": tags})


def is_readonly_command(arguments: dict[str, Any]) -> bool:
    argv = arguments.get("argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(item, str) for item in argv):
        return False
    command = Path(argv[0]).name
    if command == "pwd":
        return len(argv) == 1
    if command != "ls":
        return False
    for item in argv[1:]:
        if item == ".":
            continue
        if item.startswith("-") and "R" not in item:
            continue
        return False
    return True


def target_scope(
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
        cwd = resolve_scope_path(arguments.get("cwd") or str(context.project_dir), context)
        return f"{executable}:{cwd}"
    if "cwd" in arguments:
        return str(resolve_scope_path(arguments["cwd"], context))
    return None


def target_path(tool_name: str, arguments: dict[str, Any], context: ToolExecutionContext):
    if "path" in arguments and arguments["path"] is not None:
        return resolve_scope_path(arguments["path"], context)
    if tool_name == "code.search":
        return context.project_dir.resolve()
    return None


def sensitive_read_target_hit(
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
        return directory_contains_sensitive_path(path, patterns=patterns, recursive=True)
    if tool_name == "code.list_dir":
        return directory_contains_sensitive_path(path, patterns=patterns, recursive=bool(arguments.get("recursive", False)))
    return False


def directory_contains_sensitive_path(path: Path, *, patterns: list[str], recursive: bool) -> bool:
    try:
        iterator = path.rglob("*") if recursive else path.iterdir()
        for child in iterator:
            if is_sensitive_path(child, patterns=patterns):
                return True
    except OSError:
        return False
    return False


def resolve_scope_path(value: Any, context: ToolExecutionContext):
    raw = Path(str(value)).expanduser()
    if not raw.is_absolute():
        raw = context.effective_cwd / raw
    return raw.resolve()


def network_host(arguments: dict[str, Any]) -> str | None:
    for key in ("url", "uri", "endpoint", "base_url", "host", "hostname", "network_host"):
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            return host_from_string(value)
    for value in arguments.values():
        if isinstance(value, str) and "://" in value:
            return host_from_string(value)
    return None


def host_from_string(value: str) -> str | None:
    text = value.strip()
    parsed = urlparse(text if "://" in text else f"//{text}")
    host = parsed.hostname
    return host.lower() if host else None


def network_host_allowed(host: str | None, allowed_hosts: list[str], allowed_domains: list[str]) -> bool:
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
