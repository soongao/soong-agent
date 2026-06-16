from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any

from agent_core.storage.ids import sanitize_session_id
from agent_core.storage.migrations import ensure_session_tables, migrate
from agent_core.types.common import utc_iso, utc_now
from agent_core.types.content import ContentBlock
from agent_core.types.runtime import Node, RuntimeEvent


class SQLiteStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = asyncio.Lock()
        migrate(self._conn)

    async def close(self) -> None:
        async with self._lock:
            self._conn.close()

    async def ensure_session(self, *, session_id: str, cwd: str, root_agent_id: str) -> bool:
        session_id = sanitize_session_id(session_id)
        async with self._lock:
            ensure_session_tables(self._conn, session_id)
            now = utc_iso()
            cursor = self._conn.execute(
                """
                INSERT OR IGNORE INTO sessions(
                    session_id, cwd, root_agent_id, active_node_id, parent_session_id,
                    status, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, NULL, NULL, 'active', '{}', ?, ?)
                """,
                (session_id, cwd, root_agent_id, now, now),
            )
            self._conn.commit()
            return cursor.rowcount == 1

    async def ensure_agent(
        self,
        *,
        agent_id: str,
        session_id: str,
        agent_type: str,
        status: str = "idle",
        parent_agent_id: str | None = None,
        created_by_run_id: str | None = None,
        fork_from_node_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        async with self._lock:
            now = utc_iso()
            existing = self._conn.execute(
                "SELECT session_id FROM agents WHERE agent_id=?",
                (agent_id,),
            ).fetchone()
            if existing is not None and existing["session_id"] != session_id:
                raise ValueError(f"agent_id already belongs to another session: {agent_id}")
            self._conn.execute(
                """
                INSERT INTO agents(
                    agent_id, session_id, parent_agent_id, agent_type, created_by_run_id,
                    fork_from_node_id, status, result_json, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?)
                ON CONFLICT(agent_id) DO UPDATE SET
                    status=excluded.status,
                    parent_agent_id=COALESCE(agents.parent_agent_id, excluded.parent_agent_id),
                    created_by_run_id=COALESCE(agents.created_by_run_id, excluded.created_by_run_id),
                    fork_from_node_id=COALESCE(agents.fork_from_node_id, excluded.fork_from_node_id),
                    metadata_json=CASE
                        WHEN excluded.metadata_json != '{}' THEN excluded.metadata_json
                        ELSE agents.metadata_json
                    END,
                    updated_at=excluded.updated_at
                """,
                (
                    agent_id,
                    session_id,
                    parent_agent_id,
                    agent_type,
                    created_by_run_id,
                    fork_from_node_id,
                    status,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    now,
                    now,
                ),
            )
            self._conn.commit()

    async def update_agent(
        self,
        *,
        agent_id: str,
        status: str | None = None,
        result: dict[str, Any] | None = None,
        metadata_updates: dict[str, Any] | None = None,
    ) -> None:
        async with self._lock:
            row = self._conn.execute(
                "SELECT metadata_json FROM agents WHERE agent_id=?",
                (agent_id,),
            ).fetchone()
            if row is None:
                return
            metadata = json.loads(row["metadata_json"] or "{}")
            if metadata_updates:
                metadata.update(metadata_updates)
            self._conn.execute(
                """
                UPDATE agents
                SET status=COALESCE(?, status),
                    result_json=COALESCE(?, result_json),
                    metadata_json=?,
                    updated_at=?
                WHERE agent_id=?
                """,
                (
                    status,
                    json.dumps(result, ensure_ascii=False) if result is not None else None,
                    json.dumps(metadata, ensure_ascii=False),
                    utc_iso(),
                    agent_id,
                ),
            )
            self._conn.commit()

    async def create_run(self, *, run_id: str, session_id: str, agent_id: str, status: str) -> None:
        async with self._lock:
            now = utc_iso()
            self._conn.execute(
                """
                INSERT INTO runs(
                    run_id, session_id, agent_id, status, start_node_id, end_node_id,
                    end_reason, turn_count, usage_json, error_json, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, NULL, NULL, NULL, 0, NULL, NULL, '{}', ?, ?)
                """,
                (run_id, session_id, agent_id, status, now, now),
            )
            self._conn.commit()

    async def update_run(
        self,
        *,
        run_id: str,
        status: str,
        start_node_id: str | None = None,
        end_node_id: str | None = None,
        end_reason: str | None = None,
        error: dict[str, Any] | None = None,
    ) -> None:
        async with self._lock:
            now = utc_iso()
            self._conn.execute(
                """
                UPDATE runs
                SET status=?, start_node_id=COALESCE(?, start_node_id),
                    end_node_id=COALESCE(?, end_node_id), end_reason=COALESCE(?, end_reason),
                    error_json=COALESCE(?, error_json), updated_at=?
                WHERE run_id=?
                """,
                (
                    status,
                    start_node_id,
                    end_node_id,
                    end_reason,
                    json.dumps(error) if error is not None else None,
                    now,
                    run_id,
                ),
            )
            self._conn.commit()

    async def add_node(
        self,
        *,
        session_id: str,
        parent_id: str | None,
        agent_id: str,
        run_id: str | None,
        role: str,
        node_type: str,
        content: list[ContentBlock],
        metadata: dict[str, Any] | None = None,
        token_count: int | None = None,
        make_active: bool = False,
    ) -> Node:
        session_id = sanitize_session_id(session_id)
        async with self._lock:
            ensure_session_tables(self._conn, session_id)
            table = f"nodes_{session_id}"
            row = self._conn.execute(f"SELECT COALESCE(MAX(node_seq), 0) + 1 AS next_seq FROM {table}").fetchone()
            node_seq = int(row["next_seq"])
            from agent_core.storage.ids import new_id

            node_id = new_id("node")
            now_dt = utc_now()
            content_json = json.dumps([_model_dump(block) for block in content], ensure_ascii=False)
            self._conn.execute(
                f"""
                INSERT INTO {table}(
                    node_id, node_seq, parent_id, agent_id, run_id, role, node_type,
                    content_json, metadata_json, token_count, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    node_id,
                    node_seq,
                    parent_id,
                    agent_id,
                    run_id,
                    role,
                    node_type,
                    content_json,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    token_count,
                    utc_iso(now_dt),
                ),
            )
            if make_active:
                self._conn.execute(
                    "UPDATE sessions SET active_node_id=?, updated_at=? WHERE session_id=?",
                    (node_id, utc_iso(now_dt), session_id),
                )
            self._conn.commit()
            return Node(
                node_id=node_id,
                parent_id=parent_id,
                agent_id=agent_id,
                run_id=run_id,
                role=role,
                node_type=node_type,
                content=content,
                metadata=metadata or {},
                token_count=token_count,
                created_at=now_dt,
            )

    async def add_event(self, event: RuntimeEvent) -> RuntimeEvent:
        session_id = sanitize_session_id(event.session_id)
        async with self._lock:
            ensure_session_tables(self._conn, session_id)
            table = f"events_{session_id}"
            row = self._conn.execute(f"SELECT COALESCE(MAX(seq), 0) + 1 AS next_seq FROM {table}").fetchone()
            seq = int(row["next_seq"])
            run_seq = event.run_seq
            if event.run_id and run_seq is None:
                row = self._conn.execute(
                    f"SELECT COALESCE(MAX(run_seq), 0) + 1 AS next_run_seq FROM {table} WHERE run_id=?",
                    (event.run_id,),
                ).fetchone()
                run_seq = int(row["next_run_seq"])
            stored = event.model_copy(update={"seq": seq, "run_seq": run_seq})
            self._conn.execute(
                f"""
                INSERT INTO {table}(
                    event_id, seq, run_seq, agent_id, run_id, level, event_type,
                    node_id, tool_call_id, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    stored.event_id,
                    stored.seq,
                    stored.run_seq,
                    stored.agent_id,
                    stored.run_id,
                    stored.level,
                    stored.event_type,
                    stored.node_id,
                    stored.tool_call_id,
                    json.dumps(stored.payload, ensure_ascii=False),
                    utc_iso(stored.created_at),
                ),
            )
            self._conn.commit()
            return stored

    async def active_node_id(self, session_id: str) -> str | None:
        async with self._lock:
            row = self._conn.execute("SELECT active_node_id FROM sessions WHERE session_id=?", (session_id,)).fetchone()
            return row["active_node_id"] if row else None

    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        session_id = sanitize_session_id(session_id)
        async with self._lock:
            row = self._conn.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,)).fetchone()
        return dict(row) if row else None

    async def node_exists(self, session_id: str, node_id: str) -> bool:
        session_id = sanitize_session_id(session_id)
        async with self._lock:
            ensure_session_tables(self._conn, session_id)
            row = self._conn.execute(f"SELECT 1 FROM nodes_{session_id} WHERE node_id=? LIMIT 1", (node_id,)).fetchone()
            return row is not None

    async def get_node(self, session_id: str, node_id: str) -> Node | None:
        session_id = sanitize_session_id(session_id)
        async with self._lock:
            ensure_session_tables(self._conn, session_id)
            row = self._conn.execute(f"SELECT * FROM nodes_{session_id} WHERE node_id=?", (node_id,)).fetchone()
        return _node_from_row(row) if row else None

    async def set_active_node(self, session_id: str, node_id: str) -> None:
        session_id = sanitize_session_id(session_id)
        async with self._lock:
            ensure_session_tables(self._conn, session_id)
            self._conn.execute(
                "UPDATE sessions SET active_node_id=?, updated_at=? WHERE session_id=?",
                (node_id, utc_iso(), session_id),
            )
            self._conn.commit()

    async def session_metadata(self, session_id: str) -> dict[str, Any]:
        session_id = sanitize_session_id(session_id)
        async with self._lock:
            row = self._conn.execute("SELECT metadata_json FROM sessions WHERE session_id=?", (session_id,)).fetchone()
        if not row:
            return {}
        try:
            return json.loads(row["metadata_json"] or "{}")
        except json.JSONDecodeError:
            return {}

    async def update_session_metadata(self, session_id: str, updates: dict[str, Any]) -> dict[str, Any]:
        session_id = sanitize_session_id(session_id)
        async with self._lock:
            row = self._conn.execute("SELECT metadata_json FROM sessions WHERE session_id=?", (session_id,)).fetchone()
            try:
                metadata = json.loads(row["metadata_json"] or "{}") if row else {}
            except json.JSONDecodeError:
                metadata = {}
            metadata.update(updates)
            self._conn.execute(
                "UPDATE sessions SET metadata_json=?, updated_at=? WHERE session_id=?",
                (json.dumps(metadata, ensure_ascii=False), utc_iso(), session_id),
            )
            self._conn.commit()
        return metadata

    async def memory_source_nodes_since(self, session_id: str, after_node_seq: int) -> list[tuple[int, Node]]:
        session_id = sanitize_session_id(session_id)
        async with self._lock:
            ensure_session_tables(self._conn, session_id)
            session = self._conn.execute("SELECT root_agent_id FROM sessions WHERE session_id=?", (session_id,)).fetchone()
            if not session:
                return []
            table = f"nodes_{session_id}"
            rows = self._conn.execute(
                f"""
                SELECT * FROM {table}
                WHERE node_seq > ?
                  AND role = 'user'
                  AND node_type = 'message'
                  AND agent_id = ?
                ORDER BY node_seq
                """,
                (after_node_seq, session["root_agent_id"]),
            ).fetchall()
        return [(int(row["node_seq"]), _node_from_row(row)) for row in rows]

    async def has_active_runs(self, session_id: str) -> bool:
        session_id = sanitize_session_id(session_id)
        async with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM runs WHERE session_id=? AND status IN ('queued','pending','running') LIMIT 1",
                (session_id,),
            ).fetchone()
            return row is not None

    async def delete_session(self, session_id: str) -> None:
        session_id = sanitize_session_id(session_id)
        async with self._lock:
            self._conn.execute("DELETE FROM runs WHERE session_id=?", (session_id,))
            self._conn.execute("DELETE FROM agents WHERE session_id=?", (session_id,))
            self._conn.execute("DELETE FROM artifacts WHERE session_id=?", (session_id,))
            self._conn.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))
            self._conn.execute(f"DROP TABLE IF EXISTS nodes_{session_id}")
            self._conn.execute(f"DROP TABLE IF EXISTS events_{session_id}")
            self._conn.commit()

    async def get_node_path(self, node_id: str) -> list[Node]:
        async with self._lock:
            session_rows = self._conn.execute("SELECT session_id FROM sessions").fetchall()
            found_session = None
            found = None
            for row in session_rows:
                session_id = row["session_id"]
                ensure_session_tables(self._conn, session_id)
                node = self._conn.execute(f"SELECT * FROM nodes_{session_id} WHERE node_id=?", (node_id,)).fetchone()
                if node:
                    found_session = session_id
                    found = node
                    break
            if not found_session or not found:
                return []
            path_rows = []
            current = found
            while current:
                path_rows.append(current)
                parent_id = current["parent_id"]
                if not parent_id:
                    break
                current = self._conn.execute(
                    f"SELECT * FROM nodes_{found_session} WHERE node_id=?",
                    (parent_id,),
                ).fetchone()
        return list(reversed([_node_from_row(row) for row in path_rows]))

    async def list_artifacts(self, session_id: str | None = None) -> list[dict[str, Any]]:
        async with self._lock:
            if session_id:
                rows = self._conn.execute("SELECT * FROM artifacts WHERE session_id=?", (session_id,)).fetchall()
            else:
                rows = self._conn.execute("SELECT * FROM artifacts").fetchall()
        return [dict(row) for row in rows]

    async def add_artifact(
        self,
        *,
        artifact_id: str,
        session_id: str,
        path: str,
        filename: str | None = None,
        mime_type: str | None = None,
        size_bytes: int | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        node_id: str | None = None,
        tool_call_id: str | None = None,
        summary: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        async with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO artifacts(
                    artifact_id, session_id, agent_id, run_id, node_id, tool_call_id,
                    path, filename, mime_type, size_bytes, summary, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact_id,
                    session_id,
                    agent_id,
                    run_id,
                    node_id,
                    tool_call_id,
                    path,
                    filename,
                    mime_type,
                    size_bytes,
                    summary,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    utc_iso(),
                ),
            )
            self._conn.commit()

    async def delete_artifact(self, artifact_id: str) -> None:
        async with self._lock:
            self._conn.execute("DELETE FROM artifacts WHERE artifact_id=?", (artifact_id,))
            self._conn.commit()

    async def get_artifact(self, artifact_id: str) -> dict[str, Any] | None:
        async with self._lock:
            row = self._conn.execute("SELECT * FROM artifacts WHERE artifact_id=?", (artifact_id,)).fetchone()
        return dict(row) if row else None

    async def replay_session(self, session_id: str, *, from_seq: int | None = None, to_seq: int | None = None):
        session_id = sanitize_session_id(session_id)
        async with self._lock:
            ensure_session_tables(self._conn, session_id)
            nodes_rows = self._conn.execute(f"SELECT * FROM nodes_{session_id} ORDER BY node_seq").fetchall()
            clauses = []
            params: list[Any] = []
            if from_seq is not None:
                clauses.append("seq >= ?")
                params.append(from_seq)
            if to_seq is not None:
                clauses.append("seq <= ?")
                params.append(to_seq)
            where = "WHERE " + " AND ".join(clauses) if clauses else ""
            event_rows = self._conn.execute(f"SELECT * FROM events_{session_id} {where} ORDER BY seq", params).fetchall()
        return [_node_from_row(row) for row in nodes_rows], [_event_from_row(row, session_id) for row in event_rows]

    async def replay_run(self, session_id: str, run_id: str):
        session_id = sanitize_session_id(session_id)
        async with self._lock:
            ensure_session_tables(self._conn, session_id)
            node_rows = self._conn.execute(
                f"SELECT * FROM nodes_{session_id} WHERE run_id=? ORDER BY node_seq", (run_id,)
            ).fetchall()
            event_rows = self._conn.execute(
                f"SELECT * FROM events_{session_id} WHERE run_id=? ORDER BY seq", (run_id,)
            ).fetchall()
        return [_node_from_row(row) for row in node_rows], [_event_from_row(row, session_id) for row in event_rows]

    async def find_run_session(self, run_id: str) -> str | None:
        async with self._lock:
            row = self._conn.execute("SELECT session_id FROM runs WHERE run_id=?", (run_id,)).fetchone()
            return row["session_id"] if row else None


def _model_dump(value: Any) -> dict[str, Any]:
    return value.model_dump(mode="json") if hasattr(value, "model_dump") else value


def _node_from_row(row: sqlite3.Row) -> Node:
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


def _event_from_row(row: sqlite3.Row, session_id: str) -> RuntimeEvent:
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
