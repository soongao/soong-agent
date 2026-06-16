from __future__ import annotations

import sqlite3

from agent_core.types.common import utc_iso


SCHEMA_VERSION = 1


def migrate(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            cwd TEXT NOT NULL,
            root_agent_id TEXT NOT NULL,
            active_node_id TEXT,
            parent_session_id TEXT,
            status TEXT NOT NULL,
            metadata_json TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS agents (
            agent_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            parent_agent_id TEXT,
            agent_type TEXT NOT NULL,
            created_by_run_id TEXT,
            fork_from_node_id TEXT,
            status TEXT NOT NULL,
            result_json TEXT,
            metadata_json TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            status TEXT NOT NULL,
            start_node_id TEXT,
            end_node_id TEXT,
            end_reason TEXT,
            turn_count INTEGER,
            usage_json TEXT,
            error_json TEXT,
            metadata_json TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS artifacts (
            artifact_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            agent_id TEXT,
            run_id TEXT,
            node_id TEXT,
            tool_call_id TEXT,
            path TEXT NOT NULL,
            filename TEXT,
            mime_type TEXT,
            size_bytes INTEGER,
            summary TEXT,
            metadata_json TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (?, ?)",
        (SCHEMA_VERSION, utc_iso()),
    )
    conn.commit()


def ensure_session_tables(conn: sqlite3.Connection, session_id: str) -> None:
    nodes_table = f"nodes_{session_id}"
    events_table = f"events_{session_id}"
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {nodes_table} (
            node_id TEXT PRIMARY KEY,
            node_seq INTEGER NOT NULL,
            parent_id TEXT NULL,
            agent_id TEXT NOT NULL,
            run_id TEXT NULL,
            role TEXT NOT NULL,
            node_type TEXT NOT NULL,
            content_json TEXT NOT NULL,
            metadata_json TEXT,
            token_count INTEGER,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {events_table} (
            event_id TEXT PRIMARY KEY,
            seq INTEGER NOT NULL,
            run_seq INTEGER,
            agent_id TEXT,
            run_id TEXT,
            level TEXT NOT NULL,
            event_type TEXT NOT NULL,
            node_id TEXT,
            tool_call_id TEXT,
            payload_json TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
