from __future__ import annotations

import asyncio

from agent_core.types.permissions import PermissionDecision, PermissionDecisionKind, PermissionRequest

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.events import Key
from textual.widgets import Button, Static, TextArea


class PromptTextArea(TextArea):
    async def _on_key(self, event: Key) -> None:
        if event.key not in {"up", "down"}:
            self.app._history_browsing = False  # type: ignore[attr-defined]
        if event.key == "enter":
            if await self.app._accept_branch_candidate():  # type: ignore[attr-defined]
                event.stop()
                event.prevent_default()
                return
            if self.app._accept_slash_suggestion(require_change=True):  # type: ignore[attr-defined]
                event.stop()
                event.prevent_default()
                return
            event.stop()
            event.prevent_default()
            await self.app._submit_prompt()  # type: ignore[attr-defined]
            return
        if event.key in {"alt+enter", "ctrl+j"}:
            event.stop()
            event.prevent_default()
            self.insert("\n")
            return
        if event.key == "tab" and self.app._complete_slash_command():  # type: ignore[attr-defined]
            event.stop()
            event.prevent_default()
            return
        await super()._on_key(event)

    def action_cursor_up(self, select: bool = False) -> None:
        if not select and self.app._move_branch_selection(-1):  # type: ignore[attr-defined]
            return
        if not select and self.app._move_slash_selection(-1):  # type: ignore[attr-defined]
            return
        if not select and self.cursor_location[0] == 0:
            self.app._show_history(-1)  # type: ignore[attr-defined]
            return
        super().action_cursor_up(select)

    def action_cursor_down(self, select: bool = False) -> None:
        if not select and self.app._move_branch_selection(1):  # type: ignore[attr-defined]
            return
        if not select and self.app._move_slash_selection(1):  # type: ignore[attr-defined]
            return
        last_row = max(len(self.text.splitlines() or [""]) - 1, 0)
        if not select and self.cursor_location[0] >= last_row:
            self.app._show_history(1)  # type: ignore[attr-defined]
            return
        super().action_cursor_down(select)


class InlinePermissionPrompt(Container):
    DEFAULT_CSS = """
    InlinePermissionPrompt {
        height: auto;
        margin-bottom: 1;
        padding: 1 1;
        border: solid $warning;
    }

    InlinePermissionPrompt .permission-title {
        color: $warning;
        text-style: bold;
    }

    InlinePermissionPrompt .permission-detail {
        color: $text-muted;
    }

    InlinePermissionPrompt .permission-actions {
        height: auto;
        margin-top: 1;
    }

    InlinePermissionPrompt Button {
        margin-right: 1;
    }
    """

    def __init__(self, request: PermissionRequest, future: asyncio.Future[PermissionDecision]) -> None:
        super().__init__(classes="permission-prompt")
        self.request = request
        self.future = future

    def compose(self) -> ComposeResult:
        yield Static(f"PERMISSION REQUIRED: {self.request.tool_name} ({self.request.permission})", classes="permission-title")
        yield Static(f"Target: {self.request.target_scope or self.request.cwd}", classes="permission-detail")
        yield Static(f"Args: {self.request.args_summary}", classes="permission-detail")
        with Horizontal(classes="permission-actions"):
            yield Button("Allow Once", id="allow_once", variant="success")
            yield Button("Allow Session", id="allow_session", variant="primary")
            yield Button("Deny", id="deny", variant="error")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if self.future.done():
            return
        if event.button.id == "allow_once":
            decision = PermissionDecision(decision=PermissionDecisionKind.ALLOW_ONCE)
        elif event.button.id == "allow_session":
            decision = PermissionDecision(decision=PermissionDecisionKind.ALLOW_FOR_SESSION)
        else:
            decision = PermissionDecision(decision=PermissionDecisionKind.DENY)
        self.future.set_result(decision)
        await self.remove()
