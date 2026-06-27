from __future__ import annotations

import asyncio
import logging
import os
import shutil
from pathlib import Path
from typing import Any

from agent_core.api.runtime_helpers.agents.worker_executor import WorkerExecutorContext, WorkerExecutorResult
from agent_core.errors import AgentCoreError
from agent_core.errors.codes import ErrorCode

from agent_hub.backend.database import HubDatabase
from agent_hub.backend.permissions import PermissionBridge
from agent_hub.backend.workers.executors.opencode.jsonrpc import JsonRpcProcess

logger = logging.getLogger(__name__)


class OpenCodeWorkerExecutor:
    def __init__(
        self,
        *,
        db: HubDatabase,
        permission_bridge: PermissionBridge,
        project_dir: Path,
    ) -> None:
        self._db = db
        self._permission_bridge = permission_bridge
        self._project_dir = project_dir.resolve()

    async def run(self, runtime: Any, context: WorkerExecutorContext) -> WorkerExecutorResult:
        config = dict(context.executor_config or {})
        cwd = Path(str(config.get("cwd") or self._project_dir)).expanduser().resolve()
        if not cwd.exists():
            raise AgentCoreError(ErrorCode.VALIDATION_ERROR, f"opencode cwd does not exist: {cwd}")
        command = _opencode_command(config, cwd=cwd)
        message_parts: list[str] = []
        session_ref: dict[str, str | None] = {"session_id": None}
        collecting_response = {"enabled": False}

        async def on_notification(method: str, params: dict[str, Any]) -> None:
            if method != "session/update":
                return
            update = params.get("update")
            if not isinstance(update, dict):
                return
            if not collecting_response["enabled"]:
                return
            session_update = update.get("sessionUpdate")
            if session_update == "agent_message_chunk":
                text = _content_text(update.get("content"))
                if text:
                    message_parts.append(text)
                    await runtime._emit_child_run_event(
                        stream=context.worker_stream,
                        mirror_handle=context.parent_handle,
                        session_id=context.session_id,
                        agent_id=context.worker_agent_id,
                        run_id=context.worker_run_id,
                        event_type="worker_text_delta",
                        payload={
                            "text": text,
                            "worker_id": context.worker.worker_id,
                            "worker_agent_id": context.worker_agent_id,
                            "child_run_id": context.worker_run_id,
                        },
                    )
            elif session_update in {"tool_call", "tool_call_update", "agent_thought_chunk", "usage_update", "plan"}:
                await runtime._emit_child_run_event(
                    stream=context.worker_stream,
                    mirror_handle=context.parent_handle,
                    session_id=context.session_id,
                    agent_id=context.worker_agent_id,
                    run_id=context.worker_run_id,
                    event_type=f"opencode_{session_update}",
                    payload={"worker_id": context.worker.worker_id, "update": update},
                )

        async def on_reverse_request(method: str, params: dict[str, Any]) -> Any:
            if method == "session/request_permission":
                option_id = await self._request_permission(
                    context=context,
                    external_session_id=session_ref["session_id"],
                    params=params,
                )
                return {"outcome": {"outcome": "selected", "optionId": option_id}}
            if method == "fs/read_text_file":
                return self._read_text_file(params)
            if method == "fs/write_text_file":
                return self._write_text_file(params)
            if method.startswith("terminal/"):
                raise RuntimeError("terminal requests are not supported by Agent Hub opencode worker")
            raise RuntimeError(f"unsupported ACP reverse request: {method}")

        process = JsonRpcProcess(
            command=command,
            reverse_request_handler=on_reverse_request,
            notification_handler=on_notification,
            log_name="opencode acp",
        )
        try:
            await process.start()
            await process.request(
                "initialize",
                {
                    "protocolVersion": 1,
                    "clientCapabilities": {
                        "auth": {"terminal": False},
                        "fs": {"readTextFile": True, "writeTextFile": True},
                        "terminal": False,
                    },
                    "clientInfo": {"name": "agenthub", "version": "0.1.0"},
                },
            )
            external_session_id = await self._load_or_create_session(process, context=context, cwd=cwd)
            session_ref["session_id"] = external_session_id
            collecting_response["enabled"] = True
            result = await process.request(
                "session/prompt",
                {
                    "sessionId": external_session_id,
                    "prompt": [{"type": "text", "text": _prompt_text(context)}],
                },
            )
            stop_reason = result.get("stopReason") if isinstance(result, dict) else None
            final_text = "".join(message_parts).strip()
            if not final_text:
                final_text = "OpenCode worker completed."
            return WorkerExecutorResult(
                text=final_text,
                data={"text": final_text, "external_session_id": external_session_id, "stop_reason": stop_reason},
                metadata={"external_executor": "opencode", "external_session_id": external_session_id, "stop_reason": stop_reason},
            )
        except AgentCoreError:
            raise
        except Exception as exc:
            raise AgentCoreError(ErrorCode.WORKER_NOT_AVAILABLE, f"opencode worker failed: {exc}") from exc
        finally:
            await process.close()

    async def close(self) -> None:
        return None

    async def _load_or_create_session(self, process: JsonRpcProcess, *, context: WorkerExecutorContext, cwd: Path) -> str:
        stored = await self._db.get_external_worker_session(
            core_session_id=context.session_id,
            worker_id=context.worker.worker_id,
            executor_type="opencode",
        )
        session_params = {"cwd": str(cwd), "mcpServers": []}
        if stored is not None:
            external_session_id = str(stored["external_session_id"])
            try:
                await process.request("session/load", {"sessionId": external_session_id, **session_params})
                return external_session_id
            except Exception:
                logger.warning(
                    "failed to load opencode session, creating a new one core_session_id=%s worker_id=%s external_session_id=%s",
                    context.session_id,
                    context.worker.worker_id,
                    external_session_id,
                    exc_info=True,
                )
        created = await process.request("session/new", session_params)
        external_session_id = _session_id_from_result(created)
        if not external_session_id:
            raise RuntimeError("opencode session/new returned no session id")
        await self._db.upsert_external_worker_session(
            core_session_id=context.session_id,
            worker_id=context.worker.worker_id,
            executor_type="opencode",
            external_session_id=external_session_id,
            metadata={"cwd": str(cwd)},
        )
        return external_session_id

    async def _request_permission(
        self,
        *,
        context: WorkerExecutorContext,
        external_session_id: str | None,
        params: dict[str, Any],
    ) -> str:
        tool_call = params.get("toolCall")
        tool_name = "opencode"
        args_summary = "OpenCode requests permission."
        target_scope = None
        if isinstance(tool_call, dict):
            if tool_call.get("title"):
                tool_name = str(tool_call["title"])
            elif tool_call.get("toolCallId"):
                tool_name = str(tool_call["toolCallId"])
            args_summary = _tool_call_summary(tool_call)
            locations = tool_call.get("locations")
            if isinstance(locations, list) and locations:
                location = locations[0]
                if isinstance(location, dict) and location.get("path"):
                    target_scope = str(location["path"])
        option_id = await self._permission_bridge.external_permission_callback(
            core_session_id=context.session_id,
            core_run_id=context.worker_run_id,
            tool_name=f"opencode.{tool_name}",
            permission=_permission_kind(params),
            target_scope=target_scope,
            args_summary=args_summary,
            metadata={
                "source": "opencode_acp",
                "external_session_id": external_session_id,
                "options": params.get("options") if isinstance(params.get("options"), list) else [],
                "tool_call": tool_call if isinstance(tool_call, dict) else {},
            },
        )
        allowed = {str(option.get("optionId")) for option in params.get("options", []) if isinstance(option, dict) and option.get("optionId")}
        if allowed and option_id not in allowed:
            return _match_acp_option(option_id, allowed)
        return option_id

    def _read_text_file(self, params: dict[str, Any]) -> dict[str, Any]:
        path = _resolve_path(params.get("path"), base=self._project_dir)
        limit = params.get("limit")
        line = params.get("line")
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        if isinstance(line, int) and line > 0:
            lines = lines[line:]
        if isinstance(limit, int) and limit >= 0:
            lines = lines[:limit]
        return {"content": {"type": "text", "text": "\n".join(lines)}}

    def _write_text_file(self, params: dict[str, Any]) -> dict[str, Any]:
        path = _resolve_path(params.get("path"), base=self._project_dir)
        content = params.get("content")
        if not isinstance(content, str):
            raise RuntimeError("fs/write_text_file content must be a string")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return {}


def _opencode_command(config: dict[str, Any], *, cwd: Path) -> list[str]:
    raw = config.get("command")
    if isinstance(raw, list) and raw:
        return [str(item) for item in raw]
    if isinstance(raw, str) and raw.strip():
        args = [raw.strip(), "acp"]
        if "--cwd" not in args:
            args.extend(["--cwd", str(cwd)])
        return args
    binary = str(config.get("binary") or shutil.which("opencode") or "")
    if not binary:
        raise AgentCoreError(ErrorCode.WORKER_NOT_AVAILABLE, "opencode command is not available on PATH")
    args = [binary, "acp"]
    if "--cwd" not in args:
        args.extend(["--cwd", str(cwd)])
    for extra in config.get("args") or []:
        args.append(str(extra))
    return args


def _prompt_text(context: WorkerExecutorContext) -> str:
    for block in context.worker_start_node.content:
        text = getattr(block, "text", None)
        if text:
            return str(text)
    return context.instruction


def _content_text(content: Any) -> str:
    if isinstance(content, dict):
        if content.get("type") == "text" and content.get("text"):
            return str(content["text"])
        if content.get("text"):
            return str(content["text"])
    return ""


def _session_id_from_result(result: Any) -> str | None:
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        for key in ("sessionId", "id"):
            value = result.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def _tool_call_summary(tool_call: dict[str, Any]) -> str:
    title = str(tool_call.get("title") or "OpenCode tool")
    raw_input = tool_call.get("rawInput")
    if raw_input:
        return f"{title}: {raw_input}"
    content = tool_call.get("content")
    if isinstance(content, list):
        texts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                nested = item.get("content")
                if isinstance(nested, dict) and nested.get("text"):
                    texts.append(str(nested["text"]))
        if texts:
            return f"{title}: {' '.join(texts)[:500]}"
    return title


def _permission_kind(params: dict[str, Any]) -> str:
    tool_call = params.get("toolCall")
    if isinstance(tool_call, dict):
        kind = str(tool_call.get("kind") or "")
        if kind in {"edit", "delete", "move", "execute"}:
            return "write"
        if kind in {"read", "search", "fetch", "think"}:
            return "readonly"
    return "write"


def _match_acp_option(requested: str, allowed: set[str]) -> str:
    if requested in {"allow_always", "always"}:
        preferences = ("allow_always", "always", "allow_once", "once")
    elif requested in {"allow_once", "once"}:
        preferences = ("allow_once", "once", "allow_always", "always")
    else:
        preferences = ("reject_once", "reject", "reject_always")
    for option_id in preferences:
        if option_id in allowed:
            return option_id
    reject = next((option_id for option_id in sorted(allowed) if "reject" in option_id), None)
    if reject:
        return reject
    return sorted(allowed)[0]


def _resolve_path(value: Any, *, base: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeError("path is required")
    path = Path(os.path.expanduser(value))
    if not path.is_absolute():
        path = base / path
    return path.resolve()
