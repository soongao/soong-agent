from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any

from agent_core.agents.dynamic import WorkerConfigView
from agent_core.storage.codecs import event_from_row, model_dump, node_from_row
from agent_core.storage.ids import sanitize_session_id
from agent_core.storage.migrations import ensure_session_tables, migrate
from agent_core.types.agents import AgentDefinition
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

    async def create_run(
        self,
        *,
        run_id: str,
        session_id: str,
        agent_id: str,
        status: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        async with self._lock:
            now = utc_iso()
            self._conn.execute(
                """
                INSERT INTO runs(
                    run_id, session_id, agent_id, status, start_node_id, end_node_id,
                    end_reason, turn_count, usage_json, error_json, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, NULL, NULL, NULL, 0, NULL, NULL, ?, ?, ?)
                """,
                (run_id, session_id, agent_id, status, json.dumps(metadata or {}, ensure_ascii=False), now, now),
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
            content_json = json.dumps([model_dump(block) for block in content], ensure_ascii=False)
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

    async def list_sessions(self, *, limit: int = 20, offset: int = 0) -> list[dict[str, Any]]:
        async with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM sessions
                ORDER BY updated_at DESC, created_at DESC
                LIMIT ? OFFSET ?
                """,
                (max(limit, 0), max(offset, 0)),
            ).fetchall()
        return [dict(row) for row in rows]

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
        return node_from_row(row) if row else None

    async def list_session_nodes(self, session_id: str, *, limit: int = 20, offset: int = 0) -> list[Node]:
        session_id = sanitize_session_id(session_id)
        async with self._lock:
            ensure_session_tables(self._conn, session_id)
            rows = self._conn.execute(
                f"""
                SELECT * FROM nodes_{session_id}
                ORDER BY node_seq DESC
                LIMIT ? OFFSET ?
                """,
                (max(limit, 0), max(offset, 0)),
            ).fetchall()
        return [node_from_row(row) for row in rows]

    async def list_branchable_nodes(self, session_id: str, *, limit: int = 100, offset: int = 0) -> list[Node]:
        session_id = sanitize_session_id(session_id)
        async with self._lock:
            ensure_session_tables(self._conn, session_id)
            rows = self._conn.execute(
                f"""
                SELECT * FROM nodes_{session_id}
                WHERE role = 'user' AND node_type = 'message'
                ORDER BY node_seq DESC
                LIMIT ? OFFSET ?
                """,
                (max(limit, 0), max(offset, 0)),
            ).fetchall()
        return [node_from_row(row) for row in rows]

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
        return [(int(row["node_seq"]), node_from_row(row)) for row in rows]

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
        return list(reversed([node_from_row(row) for row in path_rows]))

    async def get_session_node_path(self, session_id: str, node_id: str) -> list[Node]:
        session_id = sanitize_session_id(session_id)
        async with self._lock:
            ensure_session_tables(self._conn, session_id)
            table = f"nodes_{session_id}"
            current = self._conn.execute(f"SELECT * FROM {table} WHERE node_id=?", (node_id,)).fetchone()
            if current is None:
                return []
            path_rows = []
            while current:
                path_rows.append(current)
                parent_id = current["parent_id"]
                if not parent_id:
                    break
                current = self._conn.execute(f"SELECT * FROM {table} WHERE node_id=?", (parent_id,)).fetchone()
        return list(reversed([node_from_row(row) for row in path_rows]))

    async def fork_session_from_path(
        self,
        *,
        source_session_id: str,
        new_session_id: str,
        cwd: str,
        root_agent_id: str,
        agent_type: str,
        nodes: list[Node],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        source_session_id = sanitize_session_id(source_session_id)
        new_session_id = sanitize_session_id(new_session_id)
        async with self._lock:
            existing = self._conn.execute("SELECT 1 FROM sessions WHERE session_id=?", (new_session_id,)).fetchone()
            if existing is not None:
                raise ValueError(f"session already exists: {new_session_id}")
            ensure_session_tables(self._conn, new_session_id)
            now = utc_iso()
            fork_metadata = dict(metadata or {})
            fork_metadata.setdefault("forked_from_session_id", source_session_id)
            fork_metadata.setdefault("forked_from_node_id", nodes[-1].node_id if nodes else None)
            self._conn.execute(
                """
                INSERT INTO sessions(
                    session_id, cwd, root_agent_id, active_node_id, parent_session_id,
                    status, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, NULL, ?, 'active', ?, ?, ?)
                """,
                (
                    new_session_id,
                    cwd,
                    root_agent_id,
                    source_session_id,
                    json.dumps(fork_metadata, ensure_ascii=False),
                    now,
                    now,
                ),
            )
            self._conn.execute(
                """
                INSERT INTO agents(
                    agent_id, session_id, parent_agent_id, agent_type, created_by_run_id,
                    fork_from_node_id, status, result_json, metadata_json, created_at, updated_at
                ) VALUES (?, ?, NULL, ?, NULL, ?, 'idle', NULL, ?, ?, ?)
                """,
                (
                    root_agent_id,
                    new_session_id,
                    agent_type,
                    nodes[-1].node_id if nodes else None,
                    json.dumps({"forked_from_session_id": source_session_id}, ensure_ascii=False),
                    now,
                    now,
                ),
            )
            table = f"nodes_{new_session_id}"
            id_map: dict[str, str] = {}
            for index, node in enumerate(nodes, start=1):
                from agent_core.storage.ids import new_id

                copied_node_id = new_id("node")
                id_map[node.node_id] = copied_node_id
                copied_parent_id = id_map.get(node.parent_id or "")
                metadata_json = dict(node.metadata)
                metadata_json["forked_from_session_id"] = source_session_id
                metadata_json["forked_from_node_id"] = node.node_id
                self._conn.execute(
                    f"""
                    INSERT INTO {table}(
                        node_id, node_seq, parent_id, agent_id, run_id, role, node_type,
                        content_json, metadata_json, token_count, created_at
                    ) VALUES (?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        copied_node_id,
                        index,
                        copied_parent_id,
                        root_agent_id,
                        node.role,
                        node.node_type,
                        json.dumps([model_dump(block) for block in node.content], ensure_ascii=False),
                        json.dumps(metadata_json, ensure_ascii=False),
                        node.token_count,
                        utc_iso(node.created_at),
                    ),
                )
            active_node_id = id_map[nodes[-1].node_id] if nodes else None
            self._conn.execute(
                "UPDATE sessions SET active_node_id=?, updated_at=? WHERE session_id=?",
                (active_node_id, now, new_session_id),
            )
            self._conn.commit()
        return {"active_node_id": active_node_id, "copied_nodes": len(nodes)}

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

    async def upsert_dynamic_agent_definition(self, definition: AgentDefinition, *, source: str = "dynamic") -> None:
        async with self._lock:
            now = utc_iso()
            existing = self._conn.execute(
                "SELECT created_at FROM agent_definitions_dynamic WHERE agent_definition_id=?",
                (definition.agent_definition_id,),
            ).fetchone()
            metadata = dict(definition.metadata)
            metadata.setdefault("stored_source", source)
            model_profile = definition.model_profile if isinstance(definition.model_profile, str) else None
            model_json = (
                json.dumps(definition.model_profile, ensure_ascii=False, sort_keys=True)
                if isinstance(definition.model_profile, dict)
                else None
            )
            self._conn.execute(
                """
                INSERT INTO agent_definitions_dynamic(
                    agent_definition_id, name, description, model_profile, model_json,
                    system_prompt, suggested_tools_json, tags_json, enabled, deleted_at,
                    source, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, NULL, ?, ?, ?, ?)
                ON CONFLICT(agent_definition_id) DO UPDATE SET
                    name=excluded.name,
                    description=excluded.description,
                    model_profile=excluded.model_profile,
                    model_json=excluded.model_json,
                    system_prompt=excluded.system_prompt,
                    suggested_tools_json=excluded.suggested_tools_json,
                    tags_json=excluded.tags_json,
                    enabled=1,
                    deleted_at=NULL,
                    source=excluded.source,
                    metadata_json=excluded.metadata_json,
                    updated_at=excluded.updated_at
                """,
                (
                    definition.agent_definition_id,
                    definition.name,
                    definition.description,
                    model_profile,
                    model_json,
                    definition.body,
                    json.dumps(definition.suggested_tools, ensure_ascii=False),
                    json.dumps(definition.tags, ensure_ascii=False),
                    source,
                    json.dumps(metadata, ensure_ascii=False),
                    existing["created_at"] if existing is not None else now,
                    now,
                ),
            )
            self._conn.commit()

    async def list_dynamic_agent_definitions(
        self,
        *,
        include_disabled: bool = False,
        include_deleted: bool = False,
    ) -> list[AgentDefinition]:
        async with self._lock:
            clauses = []
            if not include_disabled:
                clauses.append("enabled=1")
            if not include_deleted:
                clauses.append("deleted_at IS NULL")
            where = "WHERE " + " AND ".join(clauses) if clauses else ""
            rows = self._conn.execute(
                f"""
                SELECT * FROM agent_definitions_dynamic
                {where}
                ORDER BY agent_definition_id
                """
            ).fetchall()
        return [_dynamic_agent_definition_from_row(row) for row in rows]

    async def upsert_dynamic_worker_config(self, worker: WorkerConfigView, *, source: str = "dynamic") -> WorkerConfigView:
        async with self._lock:
            now = utc_iso()
            existing = self._conn.execute(
                "SELECT created_at FROM worker_configs_dynamic WHERE worker_id=?",
                (worker.worker_id,),
            ).fetchone()
            created_at = existing["created_at"] if existing is not None else now
            self._conn.execute(
                """
                INSERT INTO worker_configs_dynamic(
                    worker_id, worker_pool_id, agent_definition_id, name, description,
                    system_prompt, model_profile, model_json, allowed_tools_json,
                    enabled, deleted_at, source, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(worker_id) DO UPDATE SET
                    worker_pool_id=excluded.worker_pool_id,
                    agent_definition_id=excluded.agent_definition_id,
                    name=excluded.name,
                    description=excluded.description,
                    system_prompt=excluded.system_prompt,
                    model_profile=excluded.model_profile,
                    model_json=excluded.model_json,
                    allowed_tools_json=excluded.allowed_tools_json,
                    enabled=excluded.enabled,
                    deleted_at=excluded.deleted_at,
                    source=excluded.source,
                    metadata_json=excluded.metadata_json,
                    updated_at=excluded.updated_at
                """,
                (
                    worker.worker_id,
                    worker.worker_pool_id,
                    worker.agent_definition_id,
                    worker.name,
                    worker.description,
                    worker.system_prompt,
                    worker.model_profile,
                    json.dumps(worker.model, ensure_ascii=False, sort_keys=True) if worker.model is not None else None,
                    json.dumps(worker.allowed_tools, ensure_ascii=False) if worker.allowed_tools is not None else None,
                    1 if worker.enabled else 0,
                    worker.deleted_at,
                    source,
                    json.dumps(worker.metadata, ensure_ascii=False),
                    created_at,
                    now,
                ),
            )
            self._conn.commit()
        stored = await self.get_dynamic_worker_config(worker.worker_id, include_disabled=True, include_deleted=True)
        assert stored is not None
        return stored

    async def get_dynamic_worker_config(
        self,
        worker_id: str,
        *,
        include_disabled: bool = True,
        include_deleted: bool = True,
    ) -> WorkerConfigView | None:
        async with self._lock:
            clauses = ["worker_id=?"]
            if not include_disabled:
                clauses.append("enabled=1")
            if not include_deleted:
                clauses.append("deleted_at IS NULL")
            row = self._conn.execute(
                f"SELECT * FROM worker_configs_dynamic WHERE {' AND '.join(clauses)}",
                (worker_id,),
            ).fetchone()
        return _dynamic_worker_config_from_row(row) if row is not None else None

    async def list_dynamic_worker_configs(
        self,
        *,
        include_disabled: bool = True,
        include_deleted: bool = False,
    ) -> list[WorkerConfigView]:
        async with self._lock:
            clauses = []
            if not include_disabled:
                clauses.append("enabled=1")
            if not include_deleted:
                clauses.append("deleted_at IS NULL")
            where = "WHERE " + " AND ".join(clauses) if clauses else ""
            rows = self._conn.execute(
                f"""
                SELECT * FROM worker_configs_dynamic
                {where}
                ORDER BY worker_id
                """
            ).fetchall()
        return [_dynamic_worker_config_from_row(row) for row in rows]

    async def mark_dynamic_worker_enabled(self, worker_id: str, enabled: bool) -> WorkerConfigView | None:
        async with self._lock:
            self._conn.execute(
                """
                UPDATE worker_configs_dynamic
                SET enabled=?, updated_at=?
                WHERE worker_id=? AND deleted_at IS NULL
                """,
                (1 if enabled else 0, utc_iso(), worker_id),
            )
            self._conn.commit()
        return await self.get_dynamic_worker_config(worker_id, include_disabled=True, include_deleted=True)

    async def soft_delete_dynamic_worker_config(self, worker_id: str) -> WorkerConfigView | None:
        async with self._lock:
            now = utc_iso()
            self._conn.execute(
                """
                UPDATE worker_configs_dynamic
                SET enabled=0, deleted_at=COALESCE(deleted_at, ?), updated_at=?
                WHERE worker_id=?
                """,
                (now, now, worker_id),
            )
            self._conn.commit()
        return await self.get_dynamic_worker_config(worker_id, include_disabled=True, include_deleted=True)

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
        return [node_from_row(row) for row in nodes_rows], [event_from_row(row, session_id) for row in event_rows]

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
        return [node_from_row(row) for row in node_rows], [event_from_row(row, session_id) for row in event_rows]

    async def find_run_session(self, run_id: str) -> str | None:
        async with self._lock:
            row = self._conn.execute("SELECT session_id FROM runs WHERE run_id=?", (run_id,)).fetchone()
            return row["session_id"] if row else None


def _loads_json(value: str | None, default: Any) -> Any:
    if value is None or value == "":
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _dynamic_agent_definition_from_row(row: sqlite3.Row) -> AgentDefinition:
    model_json = _loads_json(row["model_json"], None)
    model_profile = model_json if isinstance(model_json, dict) else row["model_profile"]
    metadata = _loads_json(row["metadata_json"], {})
    if not isinstance(metadata, dict):
        metadata = {}
    metadata.update(
        {
            "enabled": bool(row["enabled"]),
            "deleted_at": row["deleted_at"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
    )
    return AgentDefinition(
        agent_definition_id=row["agent_definition_id"],
        name=row["name"],
        description=row["description"],
        body=row["system_prompt"] or "",
        model_profile=model_profile,
        suggested_tools=[str(item) for item in _loads_json(row["suggested_tools_json"], [])],
        tags=[str(item) for item in _loads_json(row["tags_json"], [])],
        overrides=None,
        source="dynamic",
        metadata=metadata,
    )


def _dynamic_worker_config_from_row(row: sqlite3.Row) -> WorkerConfigView:
    metadata = _loads_json(row["metadata_json"], {})
    if not isinstance(metadata, dict):
        metadata = {}
    return WorkerConfigView(
        worker_id=row["worker_id"],
        worker_pool_id=row["worker_pool_id"],
        agent_definition_id=row["agent_definition_id"],
        name=row["name"] or row["worker_id"],
        description=row["description"] or "",
        system_prompt=row["system_prompt"],
        model_profile=row["model_profile"],
        model=_loads_json(row["model_json"], None),
        allowed_tools=_loads_json(row["allowed_tools_json"], None),
        enabled=bool(row["enabled"]),
        deleted_at=row["deleted_at"],
        source=row["source"],
        metadata=metadata,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
