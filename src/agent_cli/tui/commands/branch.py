from __future__ import annotations

from agent_cli.tui.formatting import format_node_row
from agent_cli.tui.slash import parse_positive_int
from agent_cli.tui.types import BranchCandidate


class BranchCommandsMixin:
    async def _show_active_node(self) -> None:
        await self._ensure_runtime()
        assert self.runtime is not None
        nodes = await self.runtime.list_session_nodes(self.session_id, limit=200)
        active = next((node for node in nodes if node.active), None)
        if active is None:
            await self._write_message("system", "active node: none")
            return
        await self._write_message("system", "active node\n" + format_node_row(active, selected=True))

    async def _show_session_nodes(self, argument: str = "") -> None:
        limit = parse_positive_int(argument.strip()) or 20
        await self._write_session_nodes(limit=limit)

    async def _branch_command(self, argument: str) -> bool:
        requested = argument.strip()
        if not requested:
            await self._show_branch_candidates()
            return True
        if self._has_active_run():
            await self._write_message("warning", "cannot switch active node while a run is active; use /cancel first")
            return False
        await self._ensure_runtime()
        assert self.runtime is not None
        node_id = await self._resolve_node_id(requested)
        if node_id is None:
            await self._write_message("warning", f"node not found or ambiguous: {requested}")
            return False
        await self._switch_branch_node(node_id)
        return False

    async def _show_branch_candidates(self) -> None:
        await self._ensure_runtime()
        assert self.runtime is not None
        nodes = await self.runtime.list_session_nodes(self.session_id, limit=200)
        candidates = [
            BranchCandidate(node_id=node.node_id, preview=node.content_preview, active=node.active)
            for node in nodes
            if node.role == "user" and node.node_type == "message"
        ]
        if not candidates:
            await self._write_message("system", "branch candidates\nno user message nodes found")
            return
        self._hide_slash_suggestions()
        self._branch_overlay_locked = True
        self._branch_candidates = candidates
        active_index = next((index for index, candidate in enumerate(candidates) if candidate.active), 0)
        self._branch_selected_index = active_index
        self._branch_window_start = 0
        self._render_branch_candidates()
        await self._write_message("system", "select a branch source with Up/Down, then press Enter")

    async def _switch_branch_node(self, node_id: str) -> None:
        await self._ensure_runtime()
        assert self.runtime is not None
        result = await self.runtime.switch_node(self.session_id, node_id)
        if result.error is not None:
            await self._write_message("error", result.error.message)
            return
        await self._write_message("system", f"active node switched: {result.node_id}\nnext prompt will branch from this node")

    async def _fork_session_command(self, argument: str) -> None:
        if self._has_active_run():
            await self._write_message("warning", "cannot fork session while a run is active; use /cancel first")
            return
        await self._ensure_runtime()
        assert self.runtime is not None
        requested = argument.strip()
        node_id = None
        if requested:
            node_id = await self._resolve_node_id(requested)
            if node_id is None:
                await self._write_message("warning", f"node not found or ambiguous: {requested}")
                return
        result = await self.runtime.fork_session(self.session_id, node_id=node_id, mode=self.mode)
        if result.error is not None:
            await self._write_message("error", result.error.message)
            return
        previous = self.session_id
        assert result.session_id is not None
        self.session_id = result.session_id
        self.current_handle = None
        self.event_task = None
        self._handles.clear()
        self._event_tasks.clear()
        self._hide_branch_candidates()
        self._run_count = 0
        self._set_status("idle")
        await self._write_message(
            "system",
            "\n".join(
                [
                    f"forked session: {self.session_id}",
                    f"source session: {previous}",
                    f"source node: {result.source_node_id}",
                    f"active node: {result.active_node_id}",
                    f"copied nodes: {result.copied_nodes}",
                ]
            ),
        )

    async def _write_session_nodes(self, *, limit: int, title: str = "session nodes") -> None:
        await self._ensure_runtime()
        assert self.runtime is not None
        nodes = await self.runtime.list_session_nodes(self.session_id, limit=limit)
        if not nodes:
            await self._write_message("system", f"{title}\nno nodes found")
            return
        rows = [title]
        rows.extend(format_node_row(node) for node in nodes)
        await self._write_message("system", "\n".join(rows))

    async def _resolve_node_id(self, text: str) -> str | None:
        await self._ensure_runtime()
        assert self.runtime is not None
        if await self.runtime.get_node_path(text):
            return text
        nodes = await self.runtime.list_session_nodes(self.session_id, limit=200)
        exact = [node.node_id for node in nodes if node.node_id == text]
        if exact:
            return exact[0]
        matches = [node.node_id for node in nodes if node.node_id.startswith(text)]
        return matches[0] if len(matches) == 1 else None
