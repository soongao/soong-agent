from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Literal

from agent_core.api import AgentRuntime, RunHandle
from agent_core.errors import AgentCoreError
from agent_core.storage import new_id
from agent_core.types.permissions import PermissionDecision, PermissionRequest
from agent_core.types.runtime import RunStatus

from agent_cli.render import event_to_lines
from agent_cli.tui.commands import CommandMixin
from agent_cli.tui.formatting import format_message
from agent_cli.tui.suggestions import SuggestionMixin
from agent_cli.tui.types import BranchCandidate, SlashSuggestion
from agent_cli.tui.widgets import InlinePermissionPrompt, PromptTextArea

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Footer, Header, Markdown, Static


class SoongAgentTui(CommandMixin, SuggestionMixin, App[int]):
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

    #slash-suggestions {
        height: auto;
        max-height: 8;
        margin-top: 1;
        padding: 0 1;
        border: solid $accent;
        color: $text;
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
        self._handles: dict[str, RunHandle] = {}
        self._event_tasks: dict[str, asyncio.Task[None]] = {}
        self._display_lock = asyncio.Lock()
        self._assistant_stream_widget: Static | None = None
        self._assistant_stream_text = ""
        self._assistant_stream_run_id: str | None = None
        self._history: list[str] = []
        self._history_index: int | None = None
        self._draft_message = ""
        self._auto_scroll = True
        self._run_count = 0
        self._slash_suggestions: list[SlashSuggestion] = []
        self._slash_selected_index = 0
        self._slash_window_start = 0
        self._branch_candidates: list[BranchCandidate] = []
        self._branch_selected_index = 0
        self._branch_window_start = 0
        self._branch_overlay_locked = False
        self._history_browsing = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static(self._status_text("idle"), id="status")
        yield VerticalScroll(id="transcript")
        yield Static("", id="slash-suggestions", markup=False)
        yield PromptTextArea(
            "",
            id="prompt",
            placeholder="Type a message or /help. Enter submits. Alt+Enter or Ctrl+J inserts a newline.",
            show_line_numbers=False,
            soft_wrap=True,
            tab_behavior="focus",
        )
        yield Static("Enter submit | /help commands | Alt+Enter/Ctrl+J newline | Up/Down history | Ctrl+S auto-scroll", id="input-help")
        yield Footer()

    async def on_mount(self) -> None:
        self.query_one("#slash-suggestions", Static).display = False
        self.query_one("#prompt", PromptTextArea).focus()
        await self._write_message("system", "agentcli TUI ready. Use /help for commands.")

    async def on_unmount(self) -> None:
        tasks = [task for task in self._event_tasks.values() if not task.done()]
        if self.event_task is not None and not self.event_task.done() and self.event_task not in tasks:
            tasks.append(self.event_task)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if self.runtime is not None:
            await self.runtime.close()

    async def _submit_prompt(self) -> None:
        prompt = self.query_one("#prompt", PromptTextArea)
        message = prompt.text.strip()
        if not message:
            return
        display_message = message
        run_message = message
        if display_message.startswith("/"):
            slash_result = await self._handle_slash_command(display_message)
            if not slash_result.handled:
                return
            if slash_result.run_message is None:
                self._record_history(display_message)
                self._branch_overlay_locked = slash_result.keep_overlay
                prompt.clear()
                if slash_result.keep_overlay:
                    self._render_branch_candidates()
                else:
                    self._hide_slash_suggestions()
                    self._hide_branch_candidates()
                prompt.focus()
                return
            run_message = slash_result.run_message
        await self._ensure_runtime()
        assert self.runtime is not None
        self._record_history(display_message)
        prompt.clear()
        self._hide_slash_suggestions()
        self._hide_branch_candidates()
        await self._write_message("user", display_message)
        try:
            handle = await self.runtime.start(run_message, session_id=self.session_id, mode=self.mode)
        except AgentCoreError as exc:
            await self._write_message("error", exc.message)
            prompt.focus()
            self._set_status("error")
            return
        if handle.status == RunStatus.QUEUED:
            await self._write_message("system", f"queued: {display_message}")
        self.current_handle = handle
        self._handles[handle.run_id] = handle
        task = asyncio.create_task(self._consume_events(handle))
        self._event_tasks[handle.run_id] = task
        self.event_task = task
        prompt.focus()
        self._set_status(handle.status.value)

    async def action_cancel_or_quit(self) -> None:
        if self._has_active_run():
            await self._cancel_active_run()
            return
        self.exit(0)

    def action_toggle_auto_scroll(self) -> None:
        self._auto_scroll = not self._auto_scroll
        active = self._active_handle()
        self._set_status(active.status.value if active is not None else "idle")

    def on_text_area_changed(self, event) -> None:
        if event.text_area.id == "prompt":
            text = event.text_area.text.strip()
            if self._branch_overlay_locked and self._branch_candidates:
                if not text or text == "/branch":
                    self._render_branch_candidates()
                    return
                self._branch_overlay_locked = False
            else:
                self._branch_overlay_locked = False
            if self._history_browsing:
                self._hide_slash_suggestions()
                return
            if self._branch_candidates:
                self._hide_branch_candidates()
            self._update_slash_suggestions(event.text_area.text)

    async def _clear_transcript(self) -> None:
        await self._finalize_assistant_stream()
        transcript = self.query_one("#transcript", VerticalScroll)
        for child in list(transcript.children):
            await child.remove()

    async def _cancel_active_run(self) -> None:
        if not self._has_active_run():
            await self._write_message("warning", "no active run to cancel")
            return
        self._close_assistant_stream()
        await self._write_message("warning", "cancelling current run")
        handle = self._active_handle()
        if handle is None:
            await self._write_message("warning", "no active run to cancel")
            return
        await handle.cancel()

    def _has_active_run(self) -> bool:
        return self._active_handle() is not None

    def _active_handle(self) -> RunHandle | None:
        active_statuses = {RunStatus.PENDING, RunStatus.RUNNING, RunStatus.QUEUED}
        for handle in self._handles.values():
            if handle.status in active_statuses:
                return handle
        if self.current_handle is not None and self.current_handle.status in active_statuses:
            return self.current_handle
        return None

    def _queued_count(self) -> int:
        return sum(1 for handle in self._handles.values() if handle.status == RunStatus.QUEUED)

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
                self.current_handle = handle
                self._set_status(handle.status.value)
                if event.event_type == "model_text_delta":
                    await self._append_assistant_delta(handle.run_id, str(event.payload.get("text", "")))
                    continue
                await self._finalize_assistant_stream()
                for line in event_to_lines(event):
                    await self._write_event_line(line.text, level=line.level)
        finally:
            await self._finalize_assistant_stream()
            prompt.focus()
            if handle.status == RunStatus.COMPLETED:
                self._run_count += 1
            self._handles.pop(handle.run_id, None)
            self._event_tasks.pop(handle.run_id, None)
            active = self._active_handle()
            current_task = asyncio.current_task()
            self.current_handle = active if active is not None else handle
            self.event_task = self._event_tasks.get(active.run_id) if active is not None else current_task
            self._set_status(active.status.value if active is not None else handle.status.value)

    async def _write_event_line(self, text: str, *, level: str) -> None:
        role = "event"
        if level == "error":
            role = "error"
        elif level == "warning":
            role = "warning"
        await self._write_message(role, text)

    async def _write_message(self, role: str, text: str) -> None:
        async with self._display_lock:
            await self._finalize_assistant_stream_unlocked()
            transcript = self.query_one("#transcript", VerticalScroll)
            widget = Static(format_message(role, text), classes=f"message {role}-message", markup=False)
            await transcript.mount(widget)
            self._scroll_transcript_end()

    async def _append_assistant_delta(self, run_id: str, text: str) -> None:
        if not text:
            return
        async with self._display_lock:
            if self._assistant_stream_run_id is not None and self._assistant_stream_run_id != run_id:
                await self._finalize_assistant_stream_unlocked()
            transcript = self.query_one("#transcript", VerticalScroll)
            if self._assistant_stream_widget is None:
                self._assistant_stream_text = ""
                self._assistant_stream_run_id = run_id
                self._assistant_stream_widget = Static("", classes="message assistant-message", markup=False)
                await transcript.mount(self._assistant_stream_widget)
            self._assistant_stream_text += text
            widget = self._assistant_stream_widget
            if widget is not None:
                widget.update(format_message("assistant", self._assistant_stream_text))
            self._scroll_transcript_end()

    def _close_assistant_stream(self) -> None:
        self._assistant_stream_widget = None
        self._assistant_stream_text = ""
        self._assistant_stream_run_id = None

    async def _finalize_assistant_stream(self) -> None:
        async with self._display_lock:
            await self._finalize_assistant_stream_unlocked()

    async def _finalize_assistant_stream_unlocked(self) -> None:
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
        self._history_browsing = False

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
        self._history_browsing = self._history_index is not None
        prompt.load_text(text)
        lines = text.splitlines() or [""]
        prompt.move_cursor((len(lines) - 1, len(lines[-1])))
        if self._history_browsing:
            self._hide_slash_suggestions()
        else:
            self._update_slash_suggestions(text)

    def _set_status(self, state: str) -> None:
        self.query_one("#status", Static).update(self._status_text(state))

    def _status_text(self, state: str) -> str:
        model = getattr(getattr(self.runtime, "config", None), "model", None)
        model_name = getattr(model, "name", "not loaded")
        scroll = "on" if self._auto_scroll else "off"
        queued = self._queued_count()
        return f"session: {self.session_id} | model: {model_name} | mode: {self.mode} | state: {state} | queued: {queued} | runs: {self._run_count} | autoscroll: {scroll}"


async def run_tui_chat(args: argparse.Namespace) -> int:
    app = SoongAgentTui(args)
    result = await app.run_async()
    return int(result or 0)
