from __future__ import annotations

import logging
import shlex
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


class CodexMcpWorkerExecutor:
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
            raise AgentCoreError(ErrorCode.VALIDATION_ERROR, f"codex mcp cwd does not exist: {cwd}")

        command = _codex_mcp_command(config)
        message_parts: list[str] = []
        final_text_ref: dict[str, str] = {"text": ""}
        thread_ref: dict[str, str | None] = {"thread_id": None}

        async def on_notification(method: str, params: dict[str, Any]) -> None:
            if method != "codex/event":
                return
            await self._handle_codex_event(
                runtime,
                context=context,
                params=params,
                message_parts=message_parts,
                final_text_ref=final_text_ref,
                thread_ref=thread_ref,
            )

        async def on_reverse_request(method: str, params: dict[str, Any]) -> Any:
            if method == "elicitation/create":
                return await self._handle_elicitation(context=context, params=params)
            raise RuntimeError(f"unsupported Codex MCP reverse request: {method}")

        process = JsonRpcProcess(
            command=command,
            reverse_request_handler=on_reverse_request,
            notification_handler=on_notification,
            log_name="codex mcp-server",
        )
        try:
            await process.start()
            await process.request(
                "initialize",
                {
                    "protocolVersion": config.get("protocolVersion", "2024-11-05"),
                    "capabilities": {"elicitation": {}},
                    "clientInfo": {"name": "agenthub", "version": "0.1.0"},
                },
            )
            await process.notify("notifications/initialized", {})

            stored_thread_id = await self._stored_thread_id(context)
            tool_name = "codex-reply" if stored_thread_id else "codex"
            arguments = _codex_tool_arguments(
                config,
                prompt=_prompt_text(context),
                cwd=cwd,
                thread_id=stored_thread_id,
            )
            result = await process.request("tools/call", {"name": tool_name, "arguments": arguments})
            result_text = _result_text(result).strip()
            thread_id = _result_thread_id(result) or thread_ref["thread_id"] or stored_thread_id
            if thread_id:
                await self._db.upsert_external_worker_session(
                    core_session_id=context.session_id,
                    worker_id=context.worker.worker_id,
                    executor_type="codex_mcp",
                    external_session_id=thread_id,
                    metadata={"cwd": str(cwd)},
                )

            final_text = (result_text or final_text_ref["text"] or "".join(message_parts)).strip()
            if not final_text:
                final_text = "Codex MCP worker completed."
            data: dict[str, Any] = {"text": final_text}
            metadata: dict[str, Any] = {"external_executor": "codex_mcp"}
            if thread_id:
                data["external_thread_id"] = thread_id
                metadata["external_thread_id"] = thread_id
            return WorkerExecutorResult(text=final_text, data=data, metadata=metadata)
        except AgentCoreError:
            raise
        except Exception as exc:
            raise AgentCoreError(ErrorCode.WORKER_NOT_AVAILABLE, f"codex mcp worker failed: {exc}") from exc
        finally:
            await process.close()

    async def close(self) -> None:
        return None

    async def _stored_thread_id(self, context: WorkerExecutorContext) -> str | None:
        stored = await self._db.get_external_worker_session(
            core_session_id=context.session_id,
            worker_id=context.worker.worker_id,
            executor_type="codex_mcp",
        )
        if stored is None:
            return None
        thread_id = stored.get("external_session_id")
        return str(thread_id) if thread_id else None

    async def _handle_codex_event(
        self,
        runtime: Any,
        *,
        context: WorkerExecutorContext,
        params: dict[str, Any],
        message_parts: list[str],
        final_text_ref: dict[str, str],
        thread_ref: dict[str, str | None],
    ) -> None:
        meta = params.get("_meta") if isinstance(params.get("_meta"), dict) else {}
        if isinstance(meta.get("threadId"), str) and meta["threadId"]:
            thread_ref["thread_id"] = meta["threadId"]
        message = params.get("msg") if isinstance(params.get("msg"), dict) else {}
        event_type = str(message.get("type") or "")
        if not event_type:
            return

        if event_type == "agent_message_content_delta":
            text = _delta_text(message)
            if text:
                message_parts.append(text)
                await _emit_worker_event(runtime, context, "worker_text_delta", {"text": text})
            return

        if event_type in {"agent_message", "item_completed", "raw_response_item"}:
            item = message.get("item") if isinstance(message.get("item"), dict) else {}
            text = _message_text(message) or _item_text(item)
            if text and _is_final_agent_item(message, item):
                final_text_ref["text"] = text
            if event_type == "agent_message" and text and not message_parts:
                await _emit_worker_event(runtime, context, "worker_text_delta", {"text": text})
            return

        if event_type in {"exec_approval_request", "task_started", "task_complete", "token_count"} or event_type.startswith("mcp_"):
            payload = {
                "worker_id": context.worker.worker_id,
                "worker_agent_id": context.worker_agent_id,
                "child_run_id": context.worker_run_id,
                "codex_event": message,
            }
            await _emit_worker_event(runtime, context, f"codex_{event_type}", payload)

    async def _handle_elicitation(self, *, context: WorkerExecutorContext, params: dict[str, Any]) -> dict[str, Any]:
        if params.get("codex_elicitation") != "exec-approval":
            return {"decision": "denied"}
        command = params.get("codex_command")
        cwd = params.get("codex_cwd")
        command_text = _command_text(command)
        args_summary = str(params.get("message") or command_text or "Codex requests permission.")
        decision = await self._permission_bridge.external_permission_callback(
            core_session_id=context.session_id,
            core_run_id=context.worker_run_id,
            tool_name="codex.exec",
            permission="write",
            target_scope=str(cwd) if cwd else None,
            args_summary=args_summary,
            metadata={
                "source": "codex_mcp",
                "codex_elicitation": params.get("codex_elicitation"),
                "codex_call_id": params.get("codex_call_id"),
                "codex_command": command if isinstance(command, list) else [],
                "codex_cwd": cwd,
                "codex_parsed_cmd": params.get("codex_parsed_cmd") if isinstance(params.get("codex_parsed_cmd"), list) else [],
            },
        )
        return {"decision": "approved" if decision.startswith("allow") else "denied"}


def _codex_mcp_command(config: dict[str, Any]) -> list[str]:
    raw = config.get("command")
    if isinstance(raw, list) and raw:
        return [str(item) for item in raw]
    if isinstance(raw, str) and raw.strip():
        args = shlex.split(raw)
        if "mcp-server" not in args:
            args.append("mcp-server")
        return args
    binary = str(config.get("binary") or shutil.which("codex") or "")
    if not binary:
        raise AgentCoreError(ErrorCode.WORKER_NOT_AVAILABLE, "codex command is not available on PATH")
    args = [binary, "mcp-server"]
    for extra in config.get("server_args") or []:
        args.append(str(extra))
    return args


def _codex_tool_arguments(config: dict[str, Any], *, prompt: str, cwd: Path, thread_id: str | None) -> dict[str, Any]:
    arguments: dict[str, Any] = {"prompt": prompt, "cwd": str(cwd)}
    if thread_id:
        arguments["threadId"] = thread_id
    model = config.get("model")
    if isinstance(model, str) and model.strip():
        arguments["model"] = model.strip()
    profile = config.get("profile")
    if isinstance(profile, str) and profile.strip():
        arguments["profile"] = profile.strip()
    sandbox = config.get("sandbox")
    if isinstance(sandbox, str) and sandbox.strip():
        arguments["sandbox"] = sandbox.strip()
    approval_policy = config.get("approval_policy") or config.get("ask_for_approval")
    if isinstance(approval_policy, str) and approval_policy.strip():
        arguments["approval-policy"] = approval_policy.strip()
    if isinstance(config.get("config"), dict):
        arguments["config"] = dict(config["config"])
    return arguments


def _prompt_text(context: WorkerExecutorContext) -> str:
    for block in getattr(context.worker_start_node, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            return str(text)
    sections = [context.instruction.strip(), "", "Agent Hub worker metadata:", f"- task_id: {context.task_id}"]
    if context.dispatch_context:
        sections.extend(["", "Dispatch context:", context.dispatch_context.strip()])
    return "\n".join(section for section in sections if section is not None).strip()


async def _emit_worker_event(runtime: Any, context: WorkerExecutorContext, event_type: str, payload: dict[str, Any]) -> None:
    await runtime._emit_child_run_event(
        stream=context.worker_stream,
        mirror_handle=context.parent_handle,
        session_id=context.session_id,
        agent_id=context.worker_agent_id,
        run_id=context.worker_run_id,
        event_type=event_type,
        payload={
            "worker_id": context.worker.worker_id,
            "worker_agent_id": context.worker_agent_id,
            "child_run_id": context.worker_run_id,
            **payload,
        },
    )


def _result_text(result: Any) -> str:
    if not isinstance(result, dict):
        return ""
    structured = result.get("structuredContent")
    if isinstance(structured, dict) and structured.get("content"):
        return str(structured["content"])
    content = result.get("content")
    if isinstance(content, list):
        texts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text" and item.get("text"):
                texts.append(str(item["text"]))
        if texts:
            return "".join(texts)
    return ""


def _result_thread_id(result: Any) -> str | None:
    if not isinstance(result, dict):
        return None
    structured = result.get("structuredContent")
    if isinstance(structured, dict) and structured.get("threadId"):
        return str(structured["threadId"])
    return None


def _delta_text(message: dict[str, Any]) -> str:
    for key in ("delta", "text", "content"):
        value = message.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _message_text(message: dict[str, Any]) -> str:
    value = message.get("message")
    return str(value) if isinstance(value, str) and value else ""


def _item_text(item: dict[str, Any]) -> str:
    content = item.get("content")
    if not isinstance(content, list):
        return ""
    texts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        text = block.get("text")
        if isinstance(text, str):
            texts.append(text)
    return "".join(texts)


def _is_final_agent_item(message: dict[str, Any], item: dict[str, Any]) -> bool:
    if message.get("type") == "agent_message":
        return True
    item_type = str(item.get("type") or "")
    phase = item.get("phase")
    if item_type == "AgentMessage":
        return phase in {None, "final_answer"}
    if item_type == "message":
        return phase == "final_answer" or item.get("role") == "assistant"
    return False


def _command_text(command: Any) -> str:
    if isinstance(command, list):
        return " ".join(str(part) for part in command)
    if isinstance(command, str):
        return command
    return ""
