from __future__ import annotations

import ast
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from agent_core.providers.base import ModelMessage, ModelRole
from agent_core.types.content import TextBlock, ToolCallBlock, ToolResultBlock
from agent_core.types.tools import ToolCall, ToolDefinition


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"failed to parse JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"BFCL case at {path}:{line_number} must be a JSON object")
            cases.append(value)
    return cases


def case_id(case: dict[str, Any], index: int) -> str:
    for key in ("id", "case_id", "question_id"):
        value = case.get(key)
        if value is not None:
            return str(value)
    return f"case_{index}"


def case_messages(case: dict[str, Any]) -> list[ModelMessage]:
    raw_messages = case.get("messages")
    if raw_messages is None:
        raw_messages = case.get("conversation")
    if raw_messages is None:
        raw_messages = case.get("turns")
    if raw_messages is not None:
        parsed = _parse_messages(raw_messages)
        if parsed:
            return parsed

    question = case.get("question")
    if isinstance(question, str):
        return [ModelMessage(role=ModelRole.USER, content=[TextBlock(text=question)])]
    if isinstance(question, list):
        parsed = _parse_messages(question)
        if parsed:
            return parsed

    prompt = case.get("prompt")
    if isinstance(prompt, str):
        return [ModelMessage(role=ModelRole.USER, content=[TextBlock(text=prompt)])]

    raise ValueError("BFCL case does not contain messages, conversation, turns, question, or prompt")


def case_tools(case: dict[str, Any]) -> list[ToolDefinition]:
    raw_tools = case.get("tools")
    if raw_tools is None:
        raw_tools = case.get("functions")
    if raw_tools is None:
        raw_tools = case.get("function")
    if raw_tools is None:
        return []
    if isinstance(raw_tools, str):
        raw_tools = _loads_json_or_python(raw_tools)
    if isinstance(raw_tools, dict):
        raw_tools = [raw_tools]
    if not isinstance(raw_tools, list):
        raise ValueError("BFCL tools/function/functions field must be a list or object")
    return [_tool_definition(raw_tool) for raw_tool in raw_tools]


def calls_to_bfcl_result(calls: Iterable[ToolCall]) -> str:
    return "; ".join(_call_expression(call.name, call.arguments) for call in calls)


def calls_to_assistant_content(calls: Iterable[ToolCall], *, text: str = "") -> str:
    payload = [{"name": call.name, "arguments": call.arguments} for call in calls]
    if payload:
        return json.dumps(payload, ensure_ascii=False)
    return text


def prediction_record(
    *,
    case: dict[str, Any],
    index: int,
    calls: list[ToolCall],
    assistant_text: str,
    error: str | None = None,
) -> dict[str, Any]:
    content = calls_to_assistant_content(calls, text=assistant_text)
    record: dict[str, Any] = {
        "id": case_id(case, index),
        "result": calls_to_bfcl_result(calls),
        "inference_log": [
            {"role": "user", "content": _original_prompt_for_log(case)},
            {"role": "assistant", "content": content},
        ],
    }
    if error:
        record["error"] = error
    return record


def debug_record(
    *,
    case: dict[str, Any],
    index: int,
    calls: list[ToolCall],
    assistant_text: str,
    events: list[dict[str, Any]],
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "id": case_id(case, index),
        "tool_calls": [call.model_dump(mode="json") for call in calls],
        "assistant_text": assistant_text,
        "result": calls_to_bfcl_result(calls),
        "events": events,
        "error": error,
    }


def _parse_messages(raw_messages: Any) -> list[ModelMessage]:
    if isinstance(raw_messages, str):
        raw_messages = _loads_json_or_python(raw_messages)
    if not isinstance(raw_messages, list):
        return []
    if raw_messages and all(isinstance(item, list) for item in raw_messages):
        raw_messages = raw_messages[-1]

    messages: list[ModelMessage] = []
    for raw in raw_messages:
        if not isinstance(raw, dict):
            continue
        role = _model_role(str(raw.get("role") or "user"))
        content = raw.get("content")
        tool_calls = raw.get("tool_calls") or raw.get("function_call")
        if role == ModelRole.ASSISTANT and tool_calls:
            blocks = _tool_call_blocks(tool_calls)
            text = _content_text(content)
            if text:
                blocks.insert(0, TextBlock(text=text))
            messages.append(ModelMessage(role=role, content=blocks))
            continue
        if role == ModelRole.TOOL:
            tool_call_id = str(raw.get("tool_call_id") or raw.get("id") or raw.get("name") or "tool_call")
            tool_name = str(raw.get("name") or raw.get("tool_name") or "")
            messages.append(
                ModelMessage(
                    role=role,
                    content=[
                        ToolResultBlock(
                            tool_call_id=tool_call_id,
                            content=[TextBlock(text=_content_text(content))],
                            metadata={"tool_name": tool_name},
                        )
                    ],
                )
            )
            continue
        messages.append(ModelMessage(role=role, content=[TextBlock(text=_content_text(content))]))
    return messages


def _model_role(role: str) -> ModelRole:
    normalized = role.lower()
    if normalized == "assistant":
        return ModelRole.ASSISTANT
    if normalized in {"tool", "function"}:
        return ModelRole.TOOL
    if normalized == "system":
        return ModelRole.SYSTEM
    return ModelRole.USER


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def _tool_call_blocks(raw_tool_calls: Any) -> list[ToolCallBlock]:
    if isinstance(raw_tool_calls, dict):
        raw_tool_calls = [raw_tool_calls]
    if not isinstance(raw_tool_calls, list):
        return []
    blocks: list[ToolCallBlock] = []
    for index, raw in enumerate(raw_tool_calls):
        if not isinstance(raw, dict):
            continue
        function = raw.get("function") if isinstance(raw.get("function"), dict) else raw
        name = str(function.get("name") or raw.get("name") or "")
        args = function.get("arguments") or raw.get("arguments") or {}
        if isinstance(args, str):
            args = _loads_json_or_python(args)
        blocks.append(
            ToolCallBlock(
                tool_call_id=str(raw.get("id") or raw.get("tool_call_id") or f"bfcl_history_call_{index}"),
                name=name,
                arguments=args if isinstance(args, dict) else {},
            )
        )
    return blocks


def _tool_definition(raw_tool: Any) -> ToolDefinition:
    if not isinstance(raw_tool, dict):
        raise ValueError("BFCL tool entry must be an object")
    function = raw_tool.get("function") if isinstance(raw_tool.get("function"), dict) else raw_tool
    name = function.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("BFCL tool is missing function name")
    parameters = function.get("parameters")
    if parameters is None:
        parameters = function.get("input_schema")
    if parameters is None:
        parameters = function.get("parameter")
    if isinstance(parameters, str):
        parameters = _loads_json_or_python(parameters)
    if parameters is None:
        parameters = {"type": "object", "properties": {}}
    if not isinstance(parameters, dict):
        raise ValueError(f"BFCL tool {name} parameters must be an object")
    description = function.get("description")
    return ToolDefinition(
        name=name,
        description=description if isinstance(description, str) else "",
        input_schema=parameters,
        permission="readonly",
        tags={"bfcl"},
    )


def _loads_json_or_python(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return ast.literal_eval(value)


def _call_expression(name: str, arguments: dict[str, Any]) -> str:
    args = ", ".join(f"{key}={_format_arg(value)}" for key, value in arguments.items())
    return f"{name}({args})"


def _format_arg(value: Any) -> str:
    if isinstance(value, str):
        return value
    return repr(value)


def _original_prompt_for_log(case: dict[str, Any]) -> Any:
    for key in ("question", "messages", "conversation", "turns", "prompt"):
        if key in case:
            return case[key]
    return ""
