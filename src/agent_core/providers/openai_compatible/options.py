from __future__ import annotations

from agent_core.errors.codes import ErrorCode
from agent_core.providers.base import ModelEvent, ModelRequest
from agent_core.types.common import ErrorPayload


def openai_provider_options(request: ModelRequest) -> tuple[dict[str, object], str | None]:
    if not request.provider_options:
        return {}, None
    unknown_namespaces = sorted(key for key in request.provider_options if key != "openai")
    if unknown_namespaces:
        return {}, f"unknown provider_options namespace for openai: {', '.join(unknown_namespaces)}"
    options = request.provider_options.get("openai") or {}
    if not isinstance(options, dict):
        return {}, "provider_options.openai must be an object"
    allowed_keys = {"response_format", "seed", "parallel_tool_calls"}
    unknown_keys = sorted(key for key in options if key not in allowed_keys)
    if unknown_keys:
        return {}, f"unsupported openai provider_options: {', '.join(unknown_keys)}"
    return dict(options), None


def failed_config_event(message: str) -> ModelEvent:
    return ModelEvent(event_type="model_failed", error=ErrorPayload(code=ErrorCode.CONFIG_ERROR, message=message))
