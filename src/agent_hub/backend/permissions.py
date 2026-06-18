from __future__ import annotations

import asyncio
import logging
from typing import Any

from agent_core.storage import new_id
from agent_core.types.common import utc_iso
from agent_core.types.permissions import PermissionDecision, PermissionDecisionKind, PermissionRequest
from agent_hub.backend.database import HubDatabase
from agent_hub.backend.errors import raise_hub_error
from agent_hub.backend.events import HubEventHub

logger = logging.getLogger(__name__)


class PermissionBridge:
    def __init__(self, db: HubDatabase, events: HubEventHub) -> None:
        self._db = db
        self._events = events
        self._pending: dict[str, asyncio.Future[PermissionDecision]] = {}
        self._session_conversations: dict[str, str] = {}

    def bind_session(self, *, core_session_id: str, conversation_id: str) -> None:
        self._session_conversations[core_session_id] = conversation_id

    async def permission_callback(self, request: PermissionRequest) -> PermissionDecision:
        conversation_id = self._session_conversations.get(request.session_id)
        if conversation_id is None:
            return PermissionDecision(decision=PermissionDecisionKind.DENY, reason="conversation not found")
        permission_request_id = new_id("perm")
        now = utc_iso()
        await self._db.conn.execute(
            """
            INSERT INTO permission_requests(
                permission_request_id, conversation_id, core_session_id, core_run_id,
                tool_name, permission, target_scope, args_summary, status,
                decision, metadata_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', NULL, '{}', ?, ?)
            """,
            (
                permission_request_id,
                conversation_id,
                request.session_id,
                request.run_id,
                request.tool_name,
                request.permission,
                request.target_scope,
                request.args_summary,
                now,
                now,
            ),
        )
        await self._db.conn.commit()
        future: asyncio.Future[PermissionDecision] = asyncio.get_running_loop().create_future()
        self._pending[permission_request_id] = future
        await self._events.publish(
            "permission_requested",
            conversation_id=conversation_id,
            payload={
                "permission_request_id": permission_request_id,
                "tool_name": request.tool_name,
                "permission": request.permission,
                "target_scope": request.target_scope,
                "args_summary": request.args_summary,
                "suggested_decision": request.suggested_decision,
            },
        )
        logger.info(
            "permission requested permission_request_id=%s conversation_id=%s session_id=%s run_id=%s tool=%s permission=%s",
            permission_request_id,
            conversation_id,
            request.session_id,
            request.run_id,
            request.tool_name,
            request.permission,
        )
        try:
            return await future
        finally:
            self._pending.pop(permission_request_id, None)

    async def decide(self, permission_request_id: str, decision: str) -> dict[str, Any]:
        row = await (
            await self._db.conn.execute(
                "SELECT * FROM permission_requests WHERE permission_request_id=?",
                (permission_request_id,),
            )
        ).fetchone()
        if row is None:
            raise_hub_error(404, "permission_not_found", f"Permission request not found: {permission_request_id}")
        if row["status"] != "pending":
            raise_hub_error(
                409,
                "permission_already_resolved",
                f"Permission request already resolved: {permission_request_id}",
                {"status": row["status"], "decision": row["decision"]},
            )
        kind = {
            "allow_once": PermissionDecisionKind.ALLOW_ONCE,
            "allow_for_session": PermissionDecisionKind.ALLOW_FOR_SESSION,
            "deny": PermissionDecisionKind.DENY,
        }[decision]
        status = "allowed" if decision.startswith("allow") else "denied"
        await self._db.conn.execute(
            """
            UPDATE permission_requests
            SET status=?, decision=?, updated_at=?
            WHERE permission_request_id=?
            """,
            (status, decision, utc_iso(), permission_request_id),
        )
        await self._db.conn.commit()
        future = self._pending.get(permission_request_id)
        if future is not None and not future.done():
            future.set_result(PermissionDecision(decision=kind))
        await self._events.publish(
            "permission_resolved",
            conversation_id=row["conversation_id"],
            payload={"permission_request_id": permission_request_id, "status": status, "decision": decision},
        )
        logger.info(
            "permission decided permission_request_id=%s conversation_id=%s status=%s decision=%s",
            permission_request_id,
            row["conversation_id"],
            status,
            decision,
        )
        return {"permission_request_id": permission_request_id, "status": status, "decision": decision}

    async def shutdown(self) -> None:
        for permission_request_id, future in list(self._pending.items()):
            row = await (
                await self._db.conn.execute(
                    "SELECT conversation_id FROM permission_requests WHERE permission_request_id=?",
                    (permission_request_id,),
                )
            ).fetchone()
            if not future.done():
                future.set_result(PermissionDecision(decision=PermissionDecisionKind.DENY, reason="backend shutdown"))
            await self._db.conn.execute(
                """
                UPDATE permission_requests
                SET status='cancelled', decision='deny', updated_at=?
                WHERE permission_request_id=?
                """,
                (utc_iso(), permission_request_id),
            )
            if row is not None:
                await self._events.publish(
                    "permission_resolved",
                    conversation_id=row["conversation_id"],
                    payload={"permission_request_id": permission_request_id, "status": "cancelled", "decision": "deny"},
                )
                logger.info(
                    "permission cancelled on shutdown permission_request_id=%s conversation_id=%s",
                    permission_request_id,
                    row["conversation_id"],
                )
        await self._db.conn.commit()
        self._pending.clear()
