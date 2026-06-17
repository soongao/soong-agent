from __future__ import annotations

import json
from typing import Any


def message_text(content: list[Any]) -> str:
    parts: list[str] = []
    for block in content:
        if getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", ""))
        elif getattr(block, "type", None) == "json":
            parts.append(json.dumps(getattr(block, "data", None), ensure_ascii=False))
        elif getattr(block, "type", None) == "artifact_ref":
            parts.append(f"[artifact:{getattr(block, 'artifact_id', '')}] {getattr(block, 'summary', '') or ''}".strip())
    return "\n".join(part for part in parts if part)


def tool_result_text(block: Any) -> str:
    if getattr(block, "is_error", False) and getattr(block, "error", None) is not None:
        error = getattr(block, "error")
        return json.dumps({"error": error.model_dump(mode="json")}, ensure_ascii=False)
    text = message_text(getattr(block, "content", []) or [])
    if text:
        return text
    return json.dumps(getattr(block, "metadata", {}) or {}, ensure_ascii=False)
