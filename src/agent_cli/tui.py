from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Literal

from agent_core.api import AgentRuntime, RunHandle
from agent_core.errors import AgentCoreError
from agent_core.storage import new_id
from agent_core.types.permissions import PermissionDecision, PermissionDecisionKind, PermissionRequest
from agent_core.types.runtime import RunStatus

from agent_cli.plain import EXIT_COMMANDS
from agent_cli.render import event_to_lines

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Header, Input, Label, Static


class PermissionScreen(ModalScreen[PermissionDecision]):
    CSS = """
    PermissionScreen {
        align: center middle;
    }

    #permission-dialog {
        width: 72;
        height: auto;
        border: thick $primary;
        background: $surface;
        padding: 1 2;
    }

    #permission-actions {
        height: auto;
        margin-top: 1;
    }

    #permission-actions Button {
        margin-right: 1;
    }
    """

    def __init__(self, request: PermissionRequest) -> None:
        super().__init__()
        self.request = request

    def compose(self) -> ComposeResult:
        with Container(id="permission-dialog"):
            yield Label(f"Permission required for {self.request.tool_name} ({self.request.permission})")
            yield Label(f"Target: {self.request.target_scope or self.request.cwd}")
            yield Label(f"Args: {self.request.args_summary}")
            with Horizontal(id="permission-actions"):
                yield Button("Allow Once", id="allow_once", variant="success")
                yield Button("Allow Session", id="allow_session", variant="primary")
                yield Button("Deny", id="deny", variant="error")

    @on(Button.Pressed)
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "allow_once":
            self.dismiss(PermissionDecision(decision=PermissionDecisionKind.ALLOW_ONCE))
        elif event.button.id == "allow_session":
            self.dismiss(PermissionDecision(decision=PermissionDecisionKind.ALLOW_FOR_SESSION))
        else:
            self.dismiss(PermissionDecision(decision=PermissionDecisionKind.DENY))


class SoongAgentTui(App[int]):
    CSS = """
    Screen {
        layout: vertical;
    }

    #status {
        height: 1;
        padding: 0 1;
        background: $boost;
    }

    #transcript {
        height: 1fr;
        border: solid $primary;
        padding: 1 1;
    }

    .message {
        height: auto;
        margin-bottom: 1;
        padding: 0 1;
    }

    .user-message {
        color: $accent;
        text-style: bold;
    }

    .assistant-message {
        color: $success;
    }

    .system-message,
    .event-message {
        color: $text-muted;
    }

    .warning-message {
        color: $warning;
    }

    .error-message {
        color: $error;
        text-style: bold;
    }

    #prompt {
        height: 3;
        margin-top: 1;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "cancel_or_quit", "Cancel/Quit", show=True),
    ]

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.args = args
        self.session_id = args.session_id or new_id("sess")
        self.mode: Literal["normal", "orchestrator"] = "orchestrator" if args.orchestrator else "normal"
        self.runtime: AgentRuntime | None = None
        self.current_handle: RunHandle | None = None
        self.event_task: asyncio.Task[None] | None = None
        self._assistant_stream_widget: Static | None = None
        self._assistant_stream_text = ""

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static(self._status_text("idle"), id="status")
        yield VerticalScroll(id="transcript")
        yield Input(placeholder="Type a message. /exit quits.", id="prompt")
        yield Footer()

    async def on_mount(self) -> None:
        self.query_one("#prompt", Input).focus()
        await self._write_message("system", "soong-agent TUI ready. Use /exit to quit.")

    async def on_unmount(self) -> None:
        if self.event_task is not None and not self.event_task.done():
            self.event_task.cancel()
            await asyncio.gather(self.event_task, return_exceptions=True)
        if self.runtime is not None:
            await self.runtime.close()

    @on(Input.Submitted, "#prompt")
    async def on_prompt_submitted(self, event: Input.Submitted) -> None:
        message = event.value.strip()
        prompt = self.query_one("#prompt", Input)
        prompt.value = ""
        if not message:
            return
        if message in EXIT_COMMANDS:
            self.exit(0)
            return
        if self.current_handle is not None and self.current_handle.status in {RunStatus.PENDING, RunStatus.RUNNING, RunStatus.QUEUED}:
            await self._write_message("warning", "run already active; wait or press Ctrl-C to cancel")
            return
        await self._ensure_runtime()
        assert self.runtime is not None
        self._close_assistant_stream()
        await self._write_message("user", message)
        prompt.disabled = True
        self._set_status("running")
        try:
            handle = await self.runtime.start(message, session_id=self.session_id, mode=self.mode)
        except AgentCoreError as exc:
            await self._write_message("error", exc.message)
            prompt.disabled = False
            prompt.focus()
            self._set_status("error")
            return
        self.current_handle = handle
        self.event_task = asyncio.create_task(self._consume_events(handle))

    async def action_cancel_or_quit(self) -> None:
        if self.current_handle is not None and self.current_handle.status in {RunStatus.PENDING, RunStatus.RUNNING, RunStatus.QUEUED}:
            self._close_assistant_stream()
            await self._write_message("warning", "cancelling current run")
            await self.current_handle.cancel()
            return
        self.exit(0)

    async def permission_callback(self, request: PermissionRequest) -> PermissionDecision:
        return await self.push_screen_wait(PermissionScreen(request))

    async def _ensure_runtime(self) -> None:
        if self.runtime is not None:
            return
        self.runtime = AgentRuntime(
            project_dir=Path(self.args.path) if self.args.path else None,
            permission_callback=self.permission_callback,
        )
        await self.runtime.__aenter__()

    async def _consume_events(self, handle: RunHandle) -> None:
        prompt = self.query_one("#prompt", Input)
        try:
            async for event in handle.events(debug=bool(getattr(self.args, "debug_events", False))):
                if event.event_type == "model_text_delta":
                    await self._append_assistant_delta(str(event.payload.get("text", "")))
                    continue
                self._close_assistant_stream()
                for line in event_to_lines(event):
                    await self._write_event_line(line.text, level=line.level)
        finally:
            self._close_assistant_stream()
            prompt.disabled = False
            prompt.focus()
            self._set_status(handle.status.value)
            self.current_handle = None

    async def _write_event_line(self, text: str, *, level: str) -> None:
        role = "event"
        if level == "error":
            role = "error"
        elif level == "warning":
            role = "warning"
        await self._write_message(role, text)

    async def _write_message(self, role: str, text: str) -> None:
        self._close_assistant_stream()
        transcript = self.query_one("#transcript", VerticalScroll)
        widget = Static(_format_message(role, text), classes=f"message {role}-message", markup=False)
        await transcript.mount(widget)
        transcript.scroll_end(animate=False)

    async def _append_assistant_delta(self, text: str) -> None:
        if not text:
            return
        transcript = self.query_one("#transcript", VerticalScroll)
        if self._assistant_stream_widget is None:
            self._assistant_stream_text = ""
            self._assistant_stream_widget = Static("", classes="message assistant-message", markup=False)
            await transcript.mount(self._assistant_stream_widget)
        self._assistant_stream_text += text
        self._assistant_stream_widget.update(_format_message("assistant", self._assistant_stream_text))
        transcript.scroll_end(animate=False)

    def _close_assistant_stream(self) -> None:
        self._assistant_stream_widget = None
        self._assistant_stream_text = ""

    def _set_status(self, state: str) -> None:
        self.query_one("#status", Static).update(self._status_text(state))

    def _status_text(self, state: str) -> str:
        return f"session: {self.session_id} | mode: {self.mode} | state: {state}"


def _format_message(role: str, text: str) -> str:
    labels = {
        "user": "USER",
        "assistant": "ASSISTANT",
        "system": "SYSTEM",
        "event": "EVENT",
        "warning": "WARNING",
        "error": "ERROR",
    }
    label = labels.get(role, role.upper())
    body = text.rstrip()
    return f"{label}\n{body}" if body else label


async def run_tui_chat(args: argparse.Namespace) -> int:
    app = SoongAgentTui(args)
    result = await app.run_async()
    return int(result or 0)
