from __future__ import annotations

import json
from typing import Any

from agent_core.tools.execution import ToolExecutionContext
from agent_core.types.content import ArtifactRefBlock, JsonBlock
from agent_core.types.tools import ToolResult


def artifact_json_if_large(
    context: ToolExecutionContext,
    *,
    tool_name: str,
    data: dict[str, Any],
    filename: str,
    summary: str,
) -> dict[str, Any] | ToolResult:
    text = json.dumps(data, ensure_ascii=False, indent=2)
    limit = context.config.tools.stdout_limit_bytes
    if len(text.encode("utf-8")) <= limit:
        return data
    artifact = context.artifact_manager.write_text(
        session_id=context.session_id,
        text=text,
        filename=filename,
        mime_type="application/json",
        summary=summary,
    )
    return ToolResult(
        tool_call_id="",
        tool_name=tool_name,
        content=[
            JsonBlock(
                data={
                    "truncated": True,
                    "artifact_id": artifact.artifact_id,
                    "summary": summary,
                }
            ),
            ArtifactRefBlock(artifact_id=artifact.artifact_id, summary=summary, mime_type="application/json"),
        ],
        metadata={"artifact_ids": [artifact.artifact_id], "output_artifact_id": artifact.artifact_id},
    )
