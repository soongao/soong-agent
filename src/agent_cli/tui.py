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

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, VerticalScroll
from textual.events import Key
from textual.widgets import Button, Footer, Header, Label, Markdown, Static, TextArea


class PromptTextArea(TextArea):
    async def _on_key(self, event: Key) -> None:
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            await self.app._submit_prompt()  # type: ignore[attr-defined]
            return
        if event.key in {"alt+enter", "ctrl+j"}:
            event.stop()
            event.prevent_default()
            self.insert("\n")
            return
        await super()._on_key(event)

    def action_cursor_up(self, select: bool = False) -> None:
        if not select and self.cursor_location[0] == 0:
            self.app._show_history(-1)  # type: ignore[attr-defined]
            return
        super().action_cursor_up(select)

    def action_cursor_down(self, select: bool = False) -> None:
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

    .assistant-markdown {
        margin-bottom: 1;
        padding: 0 1;
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
        height: 5;
        margin-top: 1;
    }

    #input-help {
        height: 1;
        padding: 0 1;
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "cancel_or_quit", "Cancel/Quit", show=True),
        Binding("ctrl+s", "toggle_auto_scroll", "Auto-scroll", show=True),
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
        self._history: list[str] = []
        self._history_index: int | None = None
        self._draft_message = ""
        self._auto_scroll = True
        self._turn_count = 0

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static(self._status_text("idle"), id="status")
        yield VerticalScroll(id="transcript")
        yield PromptTextArea(
            "",
            id="prompt",
            placeholder="Type a message. Enter submits. Alt+Enter or Ctrl+J inserts a newline.",
            show_line_numbers=False,
            soft_wrap=True,
            tab_behavior="focus",
        )
        yield Static("Enter submit | Alt+Enter/Ctrl+J newline | Up/Down history | Ctrl+S auto-scroll", id="input-help")
        yield Footer()

    async def on_mount(self) -> None:
        self.query_one("#prompt", PromptTextArea).focus()
        await self._write_message("system", "soong-agent TUI ready. Use /exit to quit.")

    async def on_unmount(self) -> None:
        if self.event_task is not None and not self.event_task.done():
            self.event_task.cancel()
            await asyncio.gather(self.event_task, return_exceptions=True)
        if self.runtime is not None:
            await self.runtime.close()

    async def _submit_prompt(self) -> None:
        prompt = self.query_one("#prompt", PromptTextArea)
        message = prompt.text.strip()
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
        self._record_history(message)
        prompt.clear()
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

    def action_toggle_auto_scroll(self) -> None:
        self._auto_scroll = not self._auto_scroll
        self._set_status(self.current_handle.status.value if self.current_handle is not None else "idle")

    async def permission_callback(self, request: PermissionRequest) -> PermissionDecision:
        await self._finalize_assistant_stream()
        loop = asyncio.get_running_loop()
        future: asyncio.Future[PermissionDecision] = loop.create_future()
        prompt = InlinePermissionPrompt(request, future)
        transcript = self.query_one("#transcript", VerticalScroll)
        await transcript.mount(prompt)
        self._scroll_transcript_end()
        return await future

    async def _ensure_runtime(self) -> None:
        if self.runtime is not None:
            return
        self.runtime = AgentRuntime(
            project_dir=Path(self.args.path) if self.args.path else None,
            permission_callback=self.permission_callback,
        )
        await self.runtime.__aenter__()

    async def _consume_events(self, handle: RunHandle) -> None:
        prompt = self.query_one("#prompt", PromptTextArea)
        try:
            async for event in handle.events(debug=bool(getattr(self.args, "debug_events", False))):
                if event.event_type == "model_text_delta":
                    await self._append_assistant_delta(str(event.payload.get("text", "")))
                    continue
                await self._finalize_assistant_stream()
                for line in event_to_lines(event):
                    await self._write_event_line(line.text, level=line.level)
        finally:
            await self._finalize_assistant_stream()
            prompt.disabled = False
            prompt.focus()
            if handle.status == RunStatus.COMPLETED:
                self._turn_count += 1
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
        await self._finalize_assistant_stream()
        transcript = self.query_one("#transcript", VerticalScroll)
        widget = Static(_format_message(role, text), classes=f"message {role}-message", markup=False)
        await transcript.mount(widget)
        self._scroll_transcript_end()

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
        self._scroll_transcript_end()

    def _close_assistant_stream(self) -> None:
        self._assistant_stream_widget = None
        self._assistant_stream_text = ""

    async def _finalize_assistant_stream(self) -> None:
        widget = self._assistant_stream_widget
        text = self._assistant_stream_text.strip()
        self._close_assistant_stream()
        if widget is None or not text:
            return
        await widget.remove()
        transcript = self.query_one("#transcript", VerticalScroll)
        markdown = Markdown(f"**ASSISTANT**\n\n{text}", classes="assistant-markdown")
        await transcript.mount(markdown)
        self._scroll_transcript_end()

    def _scroll_transcript_end(self) -> None:
        if self._auto_scroll:
            self.query_one("#transcript", VerticalScroll).scroll_end(animate=False)

    def _record_history(self, message: str) -> None:
        if not self._history or self._history[-1] != message:
            self._history.append(message)
        self._history_index = None
        self._draft_message = ""

    def _show_history(self, direction: int) -> None:
        if not self._history:
            return
        prompt = self.query_one("#prompt", PromptTextArea)
        if self._history_index is None:
            self._draft_message = prompt.text
            self._history_index = len(self._history) - 1 if direction < 0 else None
        elif direction < 0:
            self._history_index = max(0, self._history_index - 1)
        else:
            if self._history_index >= len(self._history) - 1:
                self._history_index = None
            else:
                self._history_index += 1
        text = self._draft_message if self._history_index is None else self._history[self._history_index]
        prompt.load_text(text)
        lines = text.splitlines() or [""]
        prompt.move_cursor((len(lines) - 1, len(lines[-1])))

    def _set_status(self, state: str) -> None:
        self.query_one("#status", Static).update(self._status_text(state))

    def _status_text(self, state: str) -> str:
        model = getattr(getattr(self.runtime, "config", None), "model", None)
        model_name = getattr(model, "name", "not loaded")
        scroll = "on" if self._auto_scroll else "off"
        return f"session: {self.session_id} | model: {model_name} | mode: {self.mode} | state: {state} | turns: {self._turn_count} | autoscroll: {scroll}"


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
