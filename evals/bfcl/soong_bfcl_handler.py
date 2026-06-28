from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import Any

from agent_core.config.loader import load_runtime_config, resolve_model_config
from agent_core.providers.base import ModelMessage, ModelRequest, ModelRole, Usage
from agent_core.providers.registry import default_provider_registry
from agent_core.types.content import TextBlock, ToolCallBlock, ToolResultBlock
from agent_core.types.tools import ToolCall, ToolDefinition

from bfcl_eval.constants.enums import ModelStyle
from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
from bfcl_eval.model_handler.base_handler import BaseHandler
from bfcl_eval.model_handler.utils import convert_to_function_call, convert_to_tool


@dataclass
class SoongBFCLResponse:
    text: str
    tool_calls: list[ToolCall]
    usage: Usage | None


class SoongAgentBFCLHandler(BaseHandler):
    """BFCL handler that routes function-calling prompts through soong-agent providers."""

    def __init__(
        self,
        model_name: str,
        temperature: float,
        registry_name: str,
        is_fc_model: bool,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        self.model_style = ModelStyle.OPENAI_COMPLETIONS

        config, _paths = load_runtime_config(
            home_dir=os.getenv("SOONG_BFCL_HOME_DIR") or None,
            config_path=os.getenv("SOONG_BFCL_CONFIG") or None,
            project_dir=os.getenv("SOONG_BFCL_PROJECT_DIR") or None,
        )
        profile = os.getenv("SOONG_BFCL_MODEL_PROFILE") or None
        self._model_config = resolve_model_config(config, profile)
        self._provider_registry = default_provider_registry()
        self._soong_model_name = os.getenv("SOONG_BFCL_MODEL") or self._model_config.name
        max_tokens = os.getenv("SOONG_BFCL_MAX_OUTPUT_TOKENS")
        self._max_output_tokens = int(max_tokens) if max_tokens else self._model_config.max_output_tokens

    def decode_ast(self, result: Any, language: Any, has_tool_call_tag: bool) -> list[dict[str, Any]]:
        decoded_output: list[dict[str, Any]] = []
        for invoked_function in result:
            if not isinstance(invoked_function, dict) or not invoked_function:
                raise ValueError(f"invalid function-call result item: {invoked_function!r}")
            name = list(invoked_function.keys())[0]
            params = invoked_function[name]
            if isinstance(params, str):
                params = json.loads(params)
            if not isinstance(params, dict):
                raise ValueError(f"invalid function-call arguments for {name}: {params!r}")
            decoded_output.append({name: params})
        return decoded_output

    def decode_execute(self, result: Any, has_tool_call_tag: bool) -> list[str]:
        return convert_to_function_call(result)

    def _pre_query_processing_FC(self, inference_data: dict[str, Any], test_entry: dict[str, Any]) -> dict[str, Any]:
        inference_data["messages"] = []
        inference_data["tools"] = []
        return inference_data

    def _compile_tools(self, inference_data: dict[str, Any], test_entry: dict[str, Any]) -> dict[str, Any]:
        compiled_tools = convert_to_tool(
            test_entry.get("function", []),
            GORILLA_TO_OPENAPI,
            self.model_style,
        )
        definitions: list[ToolDefinition] = []
        for raw_tool in compiled_tools:
            function = raw_tool.get("function", raw_tool)
            definitions.append(
                ToolDefinition(
                    name=function["name"],
                    description=function.get("description", ""),
                    input_schema=function.get("parameters", {"type": "object", "properties": {}}),
                    permission="readonly",
                    tags={"bfcl"},
                )
            )
        inference_data["tools"] = definitions
        inference_data["tool_log"] = compiled_tools
        return inference_data

    def add_first_turn_message_FC(self, inference_data: dict[str, Any], first_turn_message: list[dict[str, Any]]) -> dict[str, Any]:
        inference_data["messages"].extend(_bfcl_messages(first_turn_message))
        return inference_data

    def _add_next_turn_user_message_FC(self, inference_data: dict[str, Any], user_message: list[dict[str, Any]]) -> dict[str, Any]:
        inference_data["messages"].extend(_bfcl_messages(user_message))
        return inference_data

    def _add_assistant_message_FC(self, inference_data: dict[str, Any], model_response_data: dict[str, Any]) -> dict[str, Any]:
        inference_data["messages"].append(model_response_data["model_responses_message_for_chat_history"])
        return inference_data

    def _add_execution_results_FC(
        self,
        inference_data: dict[str, Any],
        execution_results: list[str],
        model_response_data: dict[str, Any],
    ) -> dict[str, Any]:
        call_names = model_response_data.get("tool_call_names", [])
        for index, execution_result in enumerate(execution_results):
            tool_call_id = model_response_data["tool_call_ids"][index]
            tool_name = call_names[index] if index < len(call_names) else ""
            inference_data["messages"].append(
                ModelMessage(
                    role=ModelRole.TOOL,
                    content=[
                        ToolResultBlock(
                            tool_call_id=tool_call_id,
                            content=[TextBlock(text=execution_result)],
                            metadata={"tool_name": tool_name},
                        )
                    ],
                )
            )
        return inference_data

    def _query_FC(self, inference_data: dict[str, Any]) -> tuple[SoongBFCLResponse, float]:
        inference_data["inference_input_log"] = {
            "messages": [message.model_dump(mode="json") for message in inference_data["messages"]],
            "tools": inference_data.get("tool_log", []),
            "soong_model": self._soong_model_name,
            "soong_provider": self._model_config.provider,
        }

        started_at = time.time()
        response = asyncio.run(
            self._query_soong_provider(
                messages=inference_data["messages"],
                tools=inference_data["tools"],
            )
        )
        return response, time.time() - started_at

    def _parse_query_response_FC(self, api_response: SoongBFCLResponse) -> dict[str, Any]:
        model_responses: Any
        assistant_blocks: list[Any] = []
        tool_call_ids: list[str] = []
        tool_call_names: list[str] = []

        if api_response.tool_calls:
            model_responses = []
            for call in api_response.tool_calls:
                model_responses.append({call.name: json.dumps(call.arguments, ensure_ascii=False)})
                tool_call_ids.append(call.tool_call_id)
                tool_call_names.append(call.name)
                assistant_blocks.append(
                    ToolCallBlock(
                        tool_call_id=call.tool_call_id,
                        name=call.name,
                        arguments=call.arguments,
                    )
                )
            if api_response.text:
                assistant_blocks.insert(0, TextBlock(text=api_response.text))
        else:
            model_responses = api_response.text
            assistant_blocks = [TextBlock(text=api_response.text)] if api_response.text else []

        usage = api_response.usage
        return {
            "model_responses": model_responses,
            "model_responses_message_for_chat_history": ModelMessage(
                role=ModelRole.ASSISTANT,
                content=assistant_blocks,
            ),
            "tool_call_ids": tool_call_ids,
            "tool_call_names": tool_call_names,
            "input_token": usage.input_tokens if usage and usage.input_tokens is not None else 0,
            "output_token": usage.output_tokens if usage and usage.output_tokens is not None else 0,
        }

    async def _query_soong_provider(self, *, messages: list[ModelMessage], tools: list[ToolDefinition]) -> SoongBFCLResponse:
        provider = self._provider_registry.create(self._model_config.provider, self._model_config)
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        usage: Usage | None = None
        try:
            request = ModelRequest(
                model=self._soong_model_name,
                messages=messages,
                tools=tools,
                temperature=self.temperature,
                max_output_tokens=self._max_output_tokens,
            )
            async for event in provider.stream(request):
                if event.event_type == "model_text_delta" and event.text_delta:
                    text_parts.append(event.text_delta)
                elif event.event_type == "model_completed":
                    tool_calls = event.tool_calls
                    usage = event.usage
                    if not text_parts and event.content:
                        text_parts.extend(
                            getattr(block, "text", "")
                            for block in event.content
                            if getattr(block, "type", None) == "text"
                        )
                elif event.event_type == "model_failed":
                    message = event.error.message if event.error else "model_failed"
                    raise RuntimeError(message)
        finally:
            await provider.close()
        return SoongBFCLResponse(text="".join(text_parts), tool_calls=tool_calls, usage=usage)


def _bfcl_messages(raw_messages: list[dict[str, Any]]) -> list[ModelMessage]:
    messages: list[ModelMessage] = []
    for raw in raw_messages:
        role = _role(raw.get("role"))
        content = _content_text(raw.get("content"))
        if role == ModelRole.TOOL:
            tool_call_id = str(raw.get("tool_call_id") or raw.get("id") or raw.get("name") or "bfcl_tool")
            tool_name = str(raw.get("name") or raw.get("tool_name") or "")
            messages.append(
                ModelMessage(
                    role=ModelRole.TOOL,
                    content=[
                        ToolResultBlock(
                            tool_call_id=tool_call_id,
                            content=[TextBlock(text=content)],
                            metadata={"tool_name": tool_name},
                        )
                    ],
                )
            )
            continue
        if role == ModelRole.ASSISTANT and raw.get("tool_calls"):
            blocks: list[Any] = []
            if content:
                blocks.append(TextBlock(text=content))
            tool_calls = raw.get("tool_calls")
            if isinstance(tool_calls, dict):
                tool_calls = [tool_calls]
            for index, tool_call in enumerate(tool_calls or []):
                function = tool_call.get("function", tool_call)
                args = function.get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args)
                blocks.append(
                    ToolCallBlock(
                        tool_call_id=str(tool_call.get("id") or tool_call.get("tool_call_id") or f"bfcl_history_call_{index}"),
                        name=str(function.get("name") or tool_call.get("name") or ""),
                        arguments=args if isinstance(args, dict) else {},
                    )
                )
            messages.append(ModelMessage(role=role, content=blocks))
            continue
        messages.append(ModelMessage(role=role, content=[TextBlock(text=content)]))
    return messages


def _role(raw_role: Any) -> ModelRole:
    normalized = str(raw_role or "user").lower()
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
