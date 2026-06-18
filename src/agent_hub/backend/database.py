from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import aiosqlite

from agent_core.storage import new_id
from agent_core.types.common import utc_iso
from agent_hub.backend.models import ConversationView, MessageView


class HubDatabase:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._conn: aiosqlite.Connection | None = None

    async def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self.path)
        self._conn.row_factory = aiosqlite.Row
        await self.migrate()

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    @property
    def conn(self) -> aiosqlite.Connection:
        assert self._conn is not None, "HubDatabase is not open"
        return self._conn

    async def migrate(self) -> None:
        conn = self.conn
        await conn.execute("PRAGMA foreign_keys=ON")
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
              conversation_id TEXT PRIMARY KEY,
              core_session_id TEXT NOT NULL UNIQUE,
              title TEXT NOT NULL DEFAULT 'New conversation',
              status TEXT NOT NULL DEFAULT 'active',
              active_core_node_id TEXT,
              last_message_preview TEXT NOT NULL DEFAULT '',
              metadata_json TEXT NOT NULL DEFAULT '{}',
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            )
            """
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
              message_id TEXT PRIMARY KEY,
              conversation_id TEXT NOT NULL,
              parent_message_id TEXT,
              sender_type TEXT NOT NULL,
              sender_id TEXT,
              sender_name TEXT NOT NULL,
              target_type TEXT,
              target_id TEXT,
              original_text TEXT NOT NULL DEFAULT '',
              display_text TEXT NOT NULL DEFAULT '',
              status TEXT NOT NULL DEFAULT 'completed',
              core_session_id TEXT,
              core_run_id TEXT,
              core_node_id TEXT,
              child_run_id TEXT,
              task_id TEXT,
              worker_id TEXT,
              queue_id TEXT,
              metadata_json TEXT NOT NULL DEFAULT '{}',
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              FOREIGN KEY(conversation_id) REFERENCES conversations(conversation_id)
            )
            """
        )
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_messages_conversation_created
            ON messages(conversation_id, created_at)
            """
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS permission_requests (
              permission_request_id TEXT PRIMARY KEY,
              conversation_id TEXT NOT NULL,
              core_session_id TEXT NOT NULL,
              core_run_id TEXT,
              tool_name TEXT NOT NULL,
              permission TEXT NOT NULL,
              target_scope TEXT,
              args_summary TEXT NOT NULL DEFAULT '',
              status TEXT NOT NULL DEFAULT 'pending',
              decision TEXT,
              metadata_json TEXT NOT NULL DEFAULT '{}',
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            )
            """
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS worker_snapshots (
              snapshot_id TEXT PRIMARY KEY,
              worker_id TEXT NOT NULL,
              worker_json TEXT NOT NULL,
              created_at TEXT NOT NULL
            )
            """
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS external_worker_sessions (
              core_session_id TEXT NOT NULL,
              worker_id TEXT NOT NULL,
              executor_type TEXT NOT NULL,
              external_session_id TEXT NOT NULL,
              metadata_json TEXT NOT NULL DEFAULT '{}',
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              PRIMARY KEY(core_session_id, worker_id, executor_type)
            )
            """
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_workers (
              conversation_id TEXT NOT NULL,
              worker_id TEXT NOT NULL,
              created_at TEXT NOT NULL,
              PRIMARY KEY(conversation_id, worker_id),
              FOREIGN KEY(conversation_id) REFERENCES conversations(conversation_id)
            )
            """
        )
        await conn.commit()

    async def create_conversation(self, *, core_session_id: str, title: str = "New conversation") -> ConversationView:
        now = utc_iso()
        conversation_id = new_id("conv")
        await self.conn.execute(
            """
            INSERT INTO conversations(
                conversation_id, core_session_id, title, status, active_core_node_id,
                last_message_preview, metadata_json, created_at, updated_at
            ) VALUES (?, ?, ?, 'active', NULL, '', '{}', ?, ?)
            """,
            (conversation_id, core_session_id, title or "New conversation", now, now),
        )
        await self.conn.commit()
        row = await self.get_conversation(conversation_id)
        assert row is not None
        return row

    async def get_conversation(self, conversation_id: str) -> ConversationView | None:
        cursor = await self.conn.execute("SELECT * FROM conversations WHERE conversation_id=?", (conversation_id,))
        row = await cursor.fetchone()
        return _conversation_from_row(row) if row is not None else None

    async def get_conversation_by_core_session(self, core_session_id: str) -> ConversationView | None:
        cursor = await self.conn.execute("SELECT * FROM conversations WHERE core_session_id=?", (core_session_id,))
        row = await cursor.fetchone()
        return _conversation_from_row(row) if row is not None else None

    async def list_conversations(self) -> list[ConversationView]:
        cursor = await self.conn.execute(
            """
            SELECT * FROM conversations
            WHERE status != 'deleted'
            ORDER BY updated_at DESC, created_at DESC
            """
        )
        rows = await cursor.fetchall()
        return [_conversation_from_row(row) for row in rows]

    async def update_conversation(self, conversation_id: str, **updates: Any) -> ConversationView | None:
        allowed = {"title", "status", "active_core_node_id", "last_message_preview"}
        fields = {key: value for key, value in updates.items() if key in allowed}
        if not fields:
            return await self.get_conversation(conversation_id)
        fields["updated_at"] = utc_iso()
        assignments = ", ".join(f"{key}=?" for key in fields)
        await self.conn.execute(
            f"UPDATE conversations SET {assignments} WHERE conversation_id=?",
            (*fields.values(), conversation_id),
        )
        await self.conn.commit()
        return await self.get_conversation(conversation_id)

    async def soft_delete_conversation(self, conversation_id: str) -> ConversationView | None:
        return await self.update_conversation(conversation_id, status="deleted")

    async def add_conversation_worker(self, conversation_id: str, worker_id: str) -> dict[str, Any]:
        now = utc_iso()
        await self.conn.execute(
            """
            INSERT OR IGNORE INTO conversation_workers(conversation_id, worker_id, created_at)
            VALUES (?, ?, ?)
            """,
            (conversation_id, worker_id, now),
        )
        await self.conn.commit()
        return {"conversation_id": conversation_id, "worker_id": worker_id, "created_at": now}

    async def remove_conversation_worker(self, conversation_id: str, worker_id: str) -> None:
        await self.conn.execute(
            "DELETE FROM conversation_workers WHERE conversation_id=? AND worker_id=?",
            (conversation_id, worker_id),
        )
        await self.conn.commit()

    async def list_conversation_worker_ids(self, conversation_id: str) -> list[str]:
        cursor = await self.conn.execute(
            """
            SELECT worker_id FROM conversation_workers
            WHERE conversation_id=?
            ORDER BY created_at ASC, worker_id ASC
            """,
            (conversation_id,),
        )
        return [str(row["worker_id"]) for row in await cursor.fetchall()]

    async def create_message(
        self,
        *,
        conversation_id: str,
        sender_type: str,
        sender_name: str,
        original_text: str = "",
        display_text: str = "",
        status: str = "completed",
        sender_id: str | None = None,
        target_type: str | None = None,
        target_id: str | None = None,
        core_session_id: str | None = None,
        core_run_id: str | None = None,
        core_node_id: str | None = None,
        child_run_id: str | None = None,
        task_id: str | None = None,
        worker_id: str | None = None,
        queue_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MessageView:
        now = utc_iso()
        message_id = new_id("msg")
        await self.conn.execute(
            """
            INSERT INTO messages(
                message_id, conversation_id, parent_message_id, sender_type, sender_id,
                sender_name, target_type, target_id, original_text, display_text,
                status, core_session_id, core_run_id, core_node_id, child_run_id,
                task_id, worker_id, queue_id, metadata_json, created_at, updated_at
            ) VALUES (?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message_id,
                conversation_id,
                sender_type,
                sender_id,
                sender_name,
                target_type,
                target_id,
                original_text,
                display_text,
                status,
                core_session_id,
                core_run_id,
                core_node_id,
                child_run_id,
                task_id,
                worker_id,
                queue_id,
                json.dumps(metadata or {}, ensure_ascii=False),
                now,
                now,
            ),
        )
        await self.conn.commit()
        row = await self.get_message(message_id)
        assert row is not None
        return row

    async def get_message(self, message_id: str) -> MessageView | None:
        cursor = await self.conn.execute("SELECT * FROM messages WHERE message_id=?", (message_id,))
        row = await cursor.fetchone()
        return _message_from_row(row) if row is not None else None

    async def list_messages(self, conversation_id: str, *, limit: int = 100) -> list[MessageView]:
        cursor = await self.conn.execute(
            """
            SELECT * FROM messages
            WHERE conversation_id=?
            ORDER BY created_at ASC
            """,
            (conversation_id,),
        )
        rows = await cursor.fetchall()
        return _sort_messages([_message_from_row(row) for row in rows])[: max(limit, 1)]

    async def update_message(self, message_id: str, **updates: Any) -> MessageView | None:
        allowed = {"status", "display_text", "core_run_id", "core_node_id", "child_run_id", "task_id", "worker_id", "queue_id"}
        fields = {key: value for key, value in updates.items() if key in allowed}
        if not fields:
            return await self.get_message(message_id)
        fields["updated_at"] = utc_iso()
        assignments = ", ".join(f"{key}=?" for key in fields)
        await self.conn.execute(
            f"UPDATE messages SET {assignments} WHERE message_id=?",
            (*fields.values(), message_id),
        )
        await self.conn.commit()
        return await self.get_message(message_id)

    async def update_messages_by_run(self, core_run_id: str, **updates: Any) -> list[MessageView]:
        allowed = {"status", "display_text"}
        fields = {key: value for key, value in updates.items() if key in allowed}
        if not fields:
            return []
        fields["updated_at"] = utc_iso()
        assignments = ", ".join(f"{key}=?" for key in fields)
        await self.conn.execute(
            f"UPDATE messages SET {assignments} WHERE core_run_id=?",
            (*fields.values(), core_run_id),
        )
        await self.conn.commit()
        cursor = await self.conn.execute(
            """
            SELECT * FROM messages
            WHERE core_run_id=?
            ORDER BY created_at ASC
            """,
            (core_run_id,),
        )
        return [_message_from_row(row) for row in await cursor.fetchall()]

    async def update_messages_by_queue_id(self, queue_id: str, **updates: Any) -> list[MessageView]:
        allowed = {"status", "display_text", "core_run_id", "core_node_id", "child_run_id", "task_id", "worker_id", "queue_id"}
        fields = {key: value for key, value in updates.items() if key in allowed}
        if not fields:
            return []
        fields["updated_at"] = utc_iso()
        assignments = ", ".join(f"{key}=?" for key in fields)
        await self.conn.execute(
            f"UPDATE messages SET {assignments} WHERE queue_id=?",
            (*fields.values(), queue_id),
        )
        await self.conn.commit()
        cursor = await self.conn.execute(
            """
            SELECT * FROM messages
            WHERE queue_id=?
            ORDER BY created_at ASC
            """,
            (queue_id,),
        )
        return [_message_from_row(row) for row in await cursor.fetchall()]

    async def latest_message_for_run(
        self,
        *,
        conversation_id: str,
        core_run_id: str | None,
        sender_type: str,
    ) -> MessageView | None:
        if core_run_id is None:
            return None
        cursor = await self.conn.execute(
            """
            SELECT * FROM messages
            WHERE conversation_id=? AND core_run_id=? AND sender_type=?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (conversation_id, core_run_id, sender_type),
        )
        row = await cursor.fetchone()
        return _message_from_row(row) if row is not None else None

    async def latest_queued_worker_message(
        self,
        *,
        conversation_id: str,
        core_run_id: str | None,
        task_id: str | None,
        worker_id: str | None,
    ) -> MessageView | None:
        if not core_run_id or not task_id or not worker_id:
            return None
        cursor = await self.conn.execute(
            """
            SELECT * FROM messages
            WHERE conversation_id=?
              AND sender_type='worker'
              AND status='queued'
              AND core_run_id=?
              AND task_id=?
              AND worker_id=?
            ORDER BY created_at ASC
            LIMIT 1
            """,
            (conversation_id, core_run_id, task_id, worker_id),
        )
        row = await cursor.fetchone()
        return _message_from_row(row) if row is not None else None

    async def latest_worker_message(
        self,
        *,
        conversation_id: str,
        child_run_id: str | None,
        worker_id: str | None,
    ) -> MessageView | None:
        if not child_run_id and not worker_id:
            return None
        clauses = ["conversation_id=?", "sender_type='worker'"]
        params: list[Any] = [conversation_id]
        if child_run_id:
            clauses.append("(child_run_id=? OR core_run_id=?)")
            params.extend([child_run_id, child_run_id])
        if worker_id:
            clauses.append("worker_id=?")
            params.append(worker_id)
        cursor = await self.conn.execute(
            f"""
            SELECT * FROM messages
            WHERE {' AND '.join(clauses)}
            ORDER BY created_at DESC
            LIMIT 1
            """,
            params,
        )
        row = await cursor.fetchone()
        return _message_from_row(row) if row is not None else None

    async def snapshot_worker(self, worker_id: str, worker_json: dict[str, Any]) -> str:
        snapshot_id = new_id("worker_snapshot")
        await self.conn.execute(
            """
            INSERT INTO worker_snapshots(snapshot_id, worker_id, worker_json, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (snapshot_id, worker_id, json.dumps(worker_json, ensure_ascii=False), utc_iso()),
        )
        await self.conn.commit()
        return snapshot_id

    async def get_external_worker_session(
        self,
        *,
        core_session_id: str,
        worker_id: str,
        executor_type: str,
    ) -> dict[str, Any] | None:
        cursor = await self.conn.execute(
            """
            SELECT * FROM external_worker_sessions
            WHERE core_session_id=? AND worker_id=? AND executor_type=?
            """,
            (core_session_id, worker_id, executor_type),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return {
            "core_session_id": row["core_session_id"],
            "worker_id": row["worker_id"],
            "executor_type": row["executor_type"],
            "external_session_id": row["external_session_id"],
            "metadata": _loads_json(row["metadata_json"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    async def upsert_external_worker_session(
        self,
        *,
        core_session_id: str,
        worker_id: str,
        executor_type: str,
        external_session_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = utc_iso()
        await self.conn.execute(
            """
            INSERT INTO external_worker_sessions(
                core_session_id, worker_id, executor_type, external_session_id,
                metadata_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(core_session_id, worker_id, executor_type) DO UPDATE SET
                external_session_id=excluded.external_session_id,
                metadata_json=excluded.metadata_json,
                updated_at=excluded.updated_at
            """,
            (
                core_session_id,
                worker_id,
                executor_type,
                external_session_id,
                json.dumps(metadata or {}, ensure_ascii=False),
                now,
                now,
            ),
        )
        await self.conn.commit()
        row = await self.get_external_worker_session(
            core_session_id=core_session_id,
            worker_id=worker_id,
            executor_type=executor_type,
        )
        assert row is not None
        return row


def _loads_json(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _conversation_from_row(row: aiosqlite.Row) -> ConversationView:
    return ConversationView(
        conversation_id=row["conversation_id"],
        core_session_id=row["core_session_id"],
        title=row["title"],
        status=row["status"],
        active_core_node_id=row["active_core_node_id"],
        last_message_preview=row["last_message_preview"],
        metadata=_loads_json(row["metadata_json"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _message_from_row(row: aiosqlite.Row) -> MessageView:
    return MessageView(
        message_id=row["message_id"],
        conversation_id=row["conversation_id"],
        parent_message_id=row["parent_message_id"],
        sender_type=row["sender_type"],
        sender_id=row["sender_id"],
        sender_name=row["sender_name"],
        target_type=row["target_type"],
        target_id=row["target_id"],
        original_text=row["original_text"],
        display_text=row["display_text"],
        status=row["status"],
        core_session_id=row["core_session_id"],
        core_run_id=row["core_run_id"],
        core_node_id=row["core_node_id"],
        child_run_id=row["child_run_id"],
        task_id=row["task_id"],
        worker_id=row["worker_id"],
        queue_id=row["queue_id"],
        metadata=_loads_json(row["metadata_json"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _sort_messages(messages: list[MessageView]) -> list[MessageView]:
    group_start: dict[str, str] = {}
    for message in messages:
        group = _message_group_key(message)
        current = group_start.get(group)
        created_at = str(message.created_at)
        if current is None or created_at < current:
            group_start[group] = created_at
    return sorted(
        messages,
        key=lambda message: (
            group_start[_message_group_key(message)],
            _message_sender_rank(message.sender_type),
            str(message.updated_at if message.sender_type != "user" else message.created_at),
            str(message.created_at),
            message.message_id,
        ),
    )


def _message_group_key(message: MessageView) -> str:
    if message.core_run_id:
        return f"run:{message.core_run_id}"
    return f"message:{message.message_id}"


def _message_sender_rank(sender_type: str) -> int:
    return {
        "user": 0,
        "worker": 1,
        "orchestrator": 2,
        "system": 3,
    }.get(sender_type, 4)
