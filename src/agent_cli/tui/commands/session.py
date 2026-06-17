from __future__ import annotations

from agent_core.storage import new_id

from agent_cli.tui.formatting import compact_time, short_id
from agent_cli.tui.slash import parse_positive_int


class SessionCommandsMixin:
    async def _new_session(self, requested_session_id: str = "") -> None:
        self._close_assistant_stream()
        previous = self.session_id
        self.session_id = requested_session_id.strip() or new_id("sess")
        self.current_handle = None
        self.event_task = None
        self._handles.clear()
        self._event_tasks.clear()
        self._hide_branch_candidates()
        self._run_count = 0
        self._set_status("idle")
        await self._write_message("system", f"new session: {self.session_id}\nprevious session: {previous}")

    async def _mode_command(self, argument: str) -> None:
        value = argument.strip().lower()
        if not value:
            await self._write_message("system", f"mode: {self.mode}")
            return
        if value not in {"normal", "orchestrator"}:
            await self._write_message("warning", "usage: /mode [normal|orchestrator]")
            return
        self.mode = value
        active = self._active_handle()
        self._set_status(active.status.value if active is not None else "idle")
        await self._write_message("system", f"mode set: {self.mode}")

    async def _show_sessions(self, argument: str) -> None:
        limit = parse_positive_int(argument.strip()) or 10
        await self._ensure_runtime()
        assert self.runtime is not None
        sessions = await self.runtime.list_sessions(limit=limit)
        if not sessions:
            await self._write_message("system", "no sessions found")
            return
        rows = ["sessions"]
        for session in sessions:
            marker = "*" if session.session_id == self.session_id else " "
            rows.append(
                f"{marker} {session.session_id} "
                f"updated={compact_time(session.updated_at)} "
                f"active={short_id(session.active_node_id)} "
                f"cwd={session.cwd}"
            )
        await self._write_message("system", "\n".join(rows))

    async def _use_session(self, argument: str) -> None:
        if self._has_active_run():
            await self._write_message("warning", "cannot switch sessions while a run is active; use /cancel first")
            return
        requested = argument.strip()
        if not requested:
            await self._write_message("warning", "usage: /use <session_id>")
            return
        await self._ensure_runtime()
        assert self.runtime is not None
        session = await self._resolve_session(requested)
        if session is None:
            await self._write_message("warning", f"session not found or ambiguous: {requested}")
            return
        previous = self.session_id
        self.session_id = session.session_id
        self.mode = "orchestrator" if session.root_agent_id.startswith("agent_orchestrator") else "normal"
        self.current_handle = None
        self.event_task = None
        self._handles.clear()
        self._event_tasks.clear()
        self._hide_branch_candidates()
        self._run_count = 0
        self._set_status("idle")
        await self._write_message("system", f"using session: {self.session_id}\nprevious session: {previous}\nmode: {self.mode}")

    async def _resolve_session(self, text: str):
        await self._ensure_runtime()
        assert self.runtime is not None
        sessions = await self.runtime.list_sessions(limit=200)
        exact = [session for session in sessions if session.session_id == text]
        if exact:
            return exact[0]
        matches = [session for session in sessions if session.session_id.startswith(text)]
        return matches[0] if len(matches) == 1 else None

    def _session_info_text(self) -> str:
        project_path = self._project_path_text()
        model = getattr(getattr(self.runtime, "config", None), "model", None)
        provider = getattr(model, "provider", "not loaded")
        model_name = getattr(model, "name", "not loaded")
        active = self._active_handle()
        state = active.status.value if active is not None else "idle"
        return "\n".join(
            [
                f"session_id: {self.session_id}",
                f"mode: {self.mode}",
                f"state: {state}",
                f"runs: {self._run_count}",
                f"provider: {provider}",
                f"model: {model_name}",
                f"project: {project_path}",
                f"autoscroll: {'on' if self._auto_scroll else 'off'}",
            ]
        )
