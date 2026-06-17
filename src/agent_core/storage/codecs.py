from __future__ import annotations

import json
import sqlite3
from typing import Any

from agent_core.types.runtime import Node, RuntimeEvent


def model_dump(value: Any) -> dict[str, Any]:
    return value.model_dump(mode="json") if hasattr(value, "model_dump") else value


def node_from_row(row: sqlite3.Row) -> Node:
    from pydantic import TypeAdapter

    from agent_core.types.content import ContentBlock

    adapter = TypeAdapter(list[ContentBlock])
    return Node(
        node_id=row["node_id"],
        parent_id=row["parent_id"],
        agent_id=row["agent_id"],
        run_id=row["run_id"],
        role=row["role"],
        node_type=row["node_type"],
        content=adapter.validate_python(json.loads(row["content_json"])),
        metadata=json.loads(row["metadata_json"] or "{}"),
        token_count=row["token_count"],
        created_at=row["created_at"],
    )


def event_from_row(row: sqlite3.Row, session_id: str) -> RuntimeEvent:
    return RuntimeEvent(
        event_id=row["event_id"],
        seq=row["seq"],
        run_seq=row["run_seq"],
        session_id=session_id,
        agent_id=row["agent_id"],
        run_id=row["run_id"],
        level=row["level"],
        event_type=row["event_type"],
        node_id=row["node_id"],
        tool_call_id=row["tool_call_id"],
        payload=json.loads(row["payload_json"] or "{}"),
        created_at=row["created_at"],
    )
