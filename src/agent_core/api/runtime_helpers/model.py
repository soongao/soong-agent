from __future__ import annotations

from collections.abc import Awaitable, Callable
import json
from typing import Any

from agent_core.config.models import AgentCoreConfig
from agent_core.errors import AgentCoreError
from agent_core.errors.codes import ErrorCode
from agent_core.providers import ModelMessage, ModelRequest, ProviderAdapter, SystemBlock
from agent_core.providers.base import ModelRole
from agent_core.types import ErrorPayload, TextBlock, ToolDefinition


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
    if provider == "openai":
        return {"openai": {"response_format": {"type": "json_schema", "json_schema": {"name": "structured_json", "schema": schema}}}}
    return {}
