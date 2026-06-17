from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from agent_core.api import AgentRuntime, RunHandle
from agent_core.context.skills import build_skill_catalog
from agent_core.config.paths import resolve_home_dir, resolve_project_dir
from agent_core.errors import AgentCoreError
from agent_core.storage import new_id
from agent_core.types.permissions import PermissionDecision, PermissionDecisionKind, PermissionRequest
from agent_core.types.runtime import RunStatus

from agent_cli.render import event_to_lines

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, VerticalScroll
from textual.events import Key
from textual.widgets import Button, Footer, Header, Markdown, Static, TextArea


SLASH_SUGGESTION_VISIBLE_ROWS = 6
NODE_ID_DISPLAY_LENGTH = 28


@dataclass(frozen=True)
class SlashSuggestion:
    completion: str
    usage: str
    description: str


@dataclass(frozen=True)
class SlashCommandResult:
    handled: bool
    run_message: str | None = None
    keep_overlay: bool = False


@dataclass(frozen=True)
class BranchCandidate:
    node_id: str
    preview: str
    active: bool = False


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
        await self._write_message("system", "soong-agent TUI ready. Use /help for commands.")

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

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
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

    def _update_slash_suggestions(self, text: str) -> None:
        widget = self.query_one("#slash-suggestions", Static)
        command_text = text.lstrip()
        stripped = command_text.strip()
        if not command_text.startswith("/") or "\n" in command_text:
            self._hide_slash_suggestions()
            return
        if any(char.isspace() for char in command_text[1:]):
            self._hide_slash_suggestions()
            return
        suggestions = self._build_slash_suggestions(command_text)
        if not suggestions:
            self._slash_suggestions = []
            self._slash_selected_index = 0
            widget.display = True
            widget.update(f"No slash command matches {stripped!r}.")
            return
        self._slash_suggestions = suggestions
        self._slash_selected_index = 0
        self._slash_window_start = 0
        self._render_slash_suggestions()

    def _hide_slash_suggestions(self) -> None:
        widget = self.query_one("#slash-suggestions", Static)
        widget.display = False
        widget.update("")
        self._slash_suggestions = []
        self._slash_selected_index = 0
        self._slash_window_start = 0

    def _hide_branch_candidates(self) -> None:
        if not self.is_mounted:
            return
        self._branch_overlay_locked = False
        widget = self.query_one("#slash-suggestions", Static)
        if self._branch_candidates:
            widget.display = False
            widget.update("")
        self._branch_candidates = []
        self._branch_selected_index = 0
        self._branch_window_start = 0

    def _complete_slash_command(self) -> bool:
        return self._accept_slash_suggestion()

    def _accept_slash_suggestion(self, *, require_change: bool = False) -> bool:
        suggestion = self._selected_slash_suggestion()
        if suggestion is None:
            return False
        prompt = self.query_one("#prompt", PromptTextArea)
        completion = suggestion.completion
        if require_change and prompt.text.lstrip() == completion:
            return False
        prompt.load_text(completion)
        prompt.move_cursor((0, len(completion)))
        self._update_slash_suggestions(completion)
        return True

    def _move_slash_selection(self, direction: int) -> bool:
        if self._history_browsing:
            return False
        if not self._slash_suggestions or not self.query_one("#slash-suggestions", Static).display:
            return False
        self._slash_selected_index = (self._slash_selected_index + direction) % len(self._slash_suggestions)
        self._render_slash_suggestions()
        return True

    def _move_branch_selection(self, direction: int) -> bool:
        if self._history_browsing:
            return False
        if not self._branch_candidates or not self.query_one("#slash-suggestions", Static).display:
            return False
        self._branch_selected_index = (self._branch_selected_index + direction) % len(self._branch_candidates)
        self._render_branch_candidates()
        return True

    async def _accept_branch_candidate(self) -> bool:
        if not self._branch_candidates or not self.query_one("#slash-suggestions", Static).display:
            return False
        index = min(self._branch_selected_index, len(self._branch_candidates) - 1)
        candidate = self._branch_candidates[index]
        if self._has_active_run():
            await self._write_message("warning", "cannot switch active node while a run is active; use /cancel first")
            return True
        await self._switch_branch_node(candidate.node_id)
        prompt = self.query_one("#prompt", PromptTextArea)
        prompt.clear()
        self._hide_branch_candidates()
        prompt.focus()
        return True

    def _build_slash_suggestions(self, text: str) -> list[SlashSuggestion]:
        if " " in text.strip():
            return []
        command_matches = [
            SlashSuggestion(completion=f"/{name}", usage=usage, description=description)
            for name, usage, description in _matching_slash_commands(text.strip())
        ]
        return command_matches + self._matching_skill_suggestions(text)

    def _matching_skill_suggestions(self, text: str) -> list[SlashSuggestion]:
        if not text.startswith("/") or " " in text.strip():
            return []
        prefix = text[1:].strip().lower()
        skills = self._cached_skills_for_suggestions()
        matches = [skill for skill in skills if skill.get("name", "").lower().startswith(prefix)]
        command_names = {name for name, _usage, _description in SLASH_COMMANDS}
        return [
            SlashSuggestion(
                completion=f"/{skill['name']} ",
                usage=f"/{skill['name']} <message>",
                description=skill.get("description") or "run with skill",
            )
            for skill in matches
            if skill["name"] not in command_names
        ]

    def _selected_slash_suggestion(self) -> SlashSuggestion | None:
        if not self._slash_suggestions:
            return None
        index = min(self._slash_selected_index, len(self._slash_suggestions) - 1)
        return self._slash_suggestions[index]

    def _render_slash_suggestions(self) -> None:
        widget = self.query_one("#slash-suggestions", Static)
        if not self._slash_suggestions:
            widget.display = False
            widget.update("")
            return
        visible = self._visible_slash_suggestions()
        rows = []
        for index, suggestion in visible:
            marker = ">" if index == self._slash_selected_index else " "
            rows.append(f"{marker} {suggestion.usage:<18} {suggestion.description}")
        widget.display = True
        widget.update("\n".join(rows))

    def _render_branch_candidates(self) -> None:
        widget = self.query_one("#slash-suggestions", Static)
        if not self._branch_candidates:
            widget.display = False
            widget.update("")
            return
        visible = self._visible_branch_candidates()
        rows = ["branch candidates"]
        for index, candidate in visible:
            selector = ">" if index == self._branch_selected_index else " "
            active = "*" if candidate.active else " "
            rows.append(f"{selector}{active} {_short_id(candidate.node_id, NODE_ID_DISPLAY_LENGTH):<{NODE_ID_DISPLAY_LENGTH}} {candidate.preview}")
        widget.display = True
        widget.update("\n".join(rows))

    def _visible_slash_suggestions(self) -> list[tuple[int, SlashSuggestion]]:
        self._sync_slash_window()
        end = self._slash_window_start + SLASH_SUGGESTION_VISIBLE_ROWS
        return list(enumerate(self._slash_suggestions))[self._slash_window_start : end]

    def _sync_slash_window(self) -> None:
        if not self._slash_suggestions:
            self._slash_window_start = 0
            return
        max_start = max(len(self._slash_suggestions) - SLASH_SUGGESTION_VISIBLE_ROWS, 0)
        selected = min(max(self._slash_selected_index, 0), len(self._slash_suggestions) - 1)
        half_window = SLASH_SUGGESTION_VISIBLE_ROWS // 2
        self._slash_window_start = selected - half_window
        self._slash_window_start = min(max(self._slash_window_start, 0), max_start)

    def _visible_branch_candidates(self) -> list[tuple[int, BranchCandidate]]:
        self._sync_branch_window()
        end = self._branch_window_start + SLASH_SUGGESTION_VISIBLE_ROWS
        return list(enumerate(self._branch_candidates))[self._branch_window_start : end]

    def _sync_branch_window(self) -> None:
        if not self._branch_candidates:
            self._branch_window_start = 0
            return
        max_start = max(len(self._branch_candidates) - SLASH_SUGGESTION_VISIBLE_ROWS, 0)
        selected = min(max(self._branch_selected_index, 0), len(self._branch_candidates) - 1)
        half_window = SLASH_SUGGESTION_VISIBLE_ROWS // 2
        self._branch_window_start = selected - half_window
        self._branch_window_start = min(max(self._branch_window_start, 0), max_start)

    def _cached_skills_for_suggestions(self) -> list[dict[str, str]]:
        if self.runtime is not None and self.runtime.paths is not None:
            home_dir = self.runtime.paths.home_dir
        else:
            home_dir = resolve_home_dir()
        return build_skill_catalog(home_dir)

    async def _handle_slash_command(self, message: str) -> SlashCommandResult:
        command, argument = _parse_slash_command(message)
        if command in {"exit", "quit"}:
            self.exit(0)
            return SlashCommandResult(handled=True)
        if command == "help":
            await self._write_message("system", _slash_help_text())
            return SlashCommandResult(handled=True)
        if command == "clear":
            if self._has_active_run():
                await self._write_message("warning", "cannot clear the transcript while a run is active; use /cancel first")
                return SlashCommandResult(handled=True)
            await self._clear_transcript()
            await self._write_message("system", "transcript cleared")
            return SlashCommandResult(handled=True)
        if command == "new":
            if self._has_active_run():
                await self._write_message("warning", "cannot start a new session while a run is active; use /cancel first")
                return SlashCommandResult(handled=True)
            await self._new_session(argument)
            return SlashCommandResult(handled=True)
        if command == "mode":
            await self._mode_command(argument)
            return SlashCommandResult(handled=True)
        if command == "session":
            await self._write_message("system", self._session_info_text())
            return SlashCommandResult(handled=True)
        if command == "sessions":
            await self._show_sessions(argument)
            return SlashCommandResult(handled=True)
        if command == "use":
            await self._use_session(argument)
            return SlashCommandResult(handled=True)
        if command == "active":
            await self._show_active_node()
            return SlashCommandResult(handled=True)
        if command in {"session-nodes", "nodes"}:
            await self._show_session_nodes(argument)
            return SlashCommandResult(handled=True)
        if command == "branch":
            keep_overlay = await self._branch_command(argument)
            return SlashCommandResult(handled=True, keep_overlay=keep_overlay)
        if command == "fork-session":
            await self._fork_session_command(argument)
            return SlashCommandResult(handled=True)
        if command == "config":
            await self._write_message("system", self._config_info_text())
            return SlashCommandResult(handled=True)
        if command == "skills":
            await self._show_skills()
            return SlashCommandResult(handled=True)
        if command == "history":
            await self._write_message("system", self._history_text(argument))
            return SlashCommandResult(handled=True)
        if command == "autoscroll":
            self.action_toggle_auto_scroll()
            await self._write_message("system", f"autoscroll {'on' if self._auto_scroll else 'off'}")
            return SlashCommandResult(handled=True)
        if command == "cancel":
            await self._cancel_active_run()
            return SlashCommandResult(handled=True)
        if self._skill_name_exists(command):
            if not argument:
                await self._write_message("warning", f"usage: /{command} <message>")
                return SlashCommandResult(handled=True)
            if await self._load_skill_for_run(command):
                return SlashCommandResult(handled=True, run_message=argument)
            return SlashCommandResult(handled=True)
        await self._write_message("warning", f"unknown slash command: /{command}. Use /help for commands.")
        return SlashCommandResult(handled=True)

    async def _clear_transcript(self) -> None:
        await self._finalize_assistant_stream()
        transcript = self.query_one("#transcript", VerticalScroll)
        for child in list(transcript.children):
            await child.remove()

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
        self.mode = value  # type: ignore[assignment]
        active = self._active_handle()
        self._set_status(active.status.value if active is not None else "idle")
        await self._write_message("system", f"mode set: {self.mode}")

    async def _show_sessions(self, argument: str) -> None:
        limit = _parse_positive_int(argument.strip()) or 10
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
                f"updated={_compact_time(session.updated_at)} "
                f"active={_short_id(session.active_node_id)} "
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

    async def _show_active_node(self) -> None:
        await self._ensure_runtime()
        assert self.runtime is not None
        nodes = await self.runtime.list_session_nodes(self.session_id, limit=200)
        active = next((node for node in nodes if node.active), None)
        if active is None:
            await self._write_message("system", "active node: none")
            return
        await self._write_message("system", "active node\n" + _format_node_row(active, selected=True))

    async def _show_session_nodes(self, argument: str = "") -> None:
        limit = _parse_positive_int(argument.strip()) or 20
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
        rows.extend(_format_node_row(node) for node in nodes)
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

    async def _resolve_session(self, text: str):
        await self._ensure_runtime()
        assert self.runtime is not None
        sessions = await self.runtime.list_sessions(limit=200)
        exact = [session for session in sessions if session.session_id == text]
        if exact:
            return exact[0]
        matches = [session for session in sessions if session.session_id.startswith(text)]
        return matches[0] if len(matches) == 1 else None

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

    def _config_info_text(self) -> str:
        if self.runtime is not None:
            config_path = self.runtime.paths.config_path
            home_dir = self.runtime.paths.home_dir
        else:
            home_dir = resolve_home_dir()
            config_path = home_dir / "config.toml"
        return "\n".join(
            [
                f"config: {config_path}",
                f"home: {home_dir}",
                f"exists: {'yes' if config_path.exists() else 'no'}",
            ]
        )

    async def _show_skills(self) -> None:
        await self._ensure_runtime()
        assert self.runtime is not None
        skills = await self.runtime.list_skills()
        if not skills:
            await self._write_message("system", "no skills found")
            return
        rows = ["available skills"]
        rows.extend(f"{skill.name} - {skill.description or skill.path}" for skill in skills)
        rows.append("Use /<skill_name> <message> to load a skill and run the message.")
        await self._write_message("system", "\n".join(rows))

    async def _load_skill_for_run(self, name: str) -> bool:
        if self._has_active_run():
            await self._write_message("warning", "cannot load a skill while a run is active; use /cancel first")
            return False
        name = name.strip()
        if not name:
            return False
        await self._ensure_runtime()
        assert self.runtime is not None
        result = await self.runtime.load_skill(self.session_id, name, mode=self.mode)
        if result.error is not None:
            if str(result.error.code) == "skill_not_found":
                return False
            await self._write_message("error", result.error.message)
            return False
        return True

    def _skill_name_exists(self, name: str) -> bool:
        return any(skill.get("name") == name for skill in self._cached_skills_for_suggestions())

    def _history_text(self, argument: str = "") -> str:
        limit = _parse_positive_int(argument.strip()) or 10
        if not self._history:
            return "no prompt history"
        start = max(len(self._history) - limit, 0)
        rows = [f"{index + 1}. {entry}" for index, entry in enumerate(self._history[start:], start=start)]
        return "prompt history\n" + "\n".join(rows)

    def _project_path_text(self) -> str:
        if self.runtime is not None:
            return str(self.runtime.paths.project_dir)
        try:
            return str(resolve_project_dir(self.args.path))
        except AgentCoreError as exc:
            return exc.message

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
            widget = Static(_format_message(role, text), classes=f"message {role}-message", markup=False)
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
                widget.update(_format_message("assistant", self._assistant_stream_text))
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


def _compact_time(value) -> str:
    text = str(value or "")
    return text.replace("T", " ")[:19]


def _short_id(value: str | None, length: int = 18) -> str:
    if not value:
        return "-"
    return value if len(value) <= length else value[:length]


def _format_node_row(node, *, selected: bool = False) -> str:
    marker = "*" if selected or getattr(node, "active", False) else " "
    preview = getattr(node, "content_preview", "") or ""
    return (
        f"{marker} {_short_id(node.node_id, NODE_ID_DISPLAY_LENGTH):<{NODE_ID_DISPLAY_LENGTH}} "
        f"{node.role:<9} {node.node_type:<14} "
        f"{preview}"
    ).rstrip()


def _parse_slash_command(message: str) -> tuple[str, str]:
    command, _, argument = message[1:].partition(" ")
    return command.lower(), argument.strip()


def _parse_positive_int(text: str) -> int | None:
    if not text:
        return None
    try:
        value = int(text)
    except ValueError:
        return None
    return value if value > 0 else None


SLASH_COMMANDS: tuple[tuple[str, str, str], ...] = (
    ("help", "/help", "show slash command help"),
    ("clear", "/clear", "clear the transcript"),
    ("new", "/new [session_id]", "start a fresh session"),
    ("mode", "/mode [normal|orchestrator]", "show or set run mode"),
    ("session", "/session", "show current session details"),
    ("sessions", "/sessions [n]", "list recent sessions"),
    ("use", "/use <session_id>", "switch current session"),
    ("active", "/active", "show current active node"),
    ("session-nodes", "/session-nodes [n]", "list nodes in current session"),
    ("nodes", "/nodes [n]", "list nodes in current session"),
    ("branch", "/branch [node_id]", "list nodes or switch active node"),
    ("fork-session", "/fork-session [node_id]", "fork current path into a new session"),
    ("config", "/config", "show config and home paths"),
    ("skills", "/skills", "list available skills"),
    ("history", "/history [n]", "show recent prompts"),
    ("autoscroll", "/autoscroll", "toggle transcript autoscroll"),
    ("cancel", "/cancel", "cancel the active run"),
    ("exit", "/exit", "quit the TUI"),
    ("quit", "/quit", "quit the TUI"),
)


def _matching_slash_commands(text: str) -> list[tuple[str, str, str]]:
    command, _, _argument = text[1:].partition(" ")
    prefix = command.lower()
    return [entry for entry in SLASH_COMMANDS if entry[0].startswith(prefix)]


def _slash_help_text() -> str:
    rows = ["slash commands"]
    rows.extend(f"{usage} - {description}" for _name, usage, description in SLASH_COMMANDS)
    rows.append("/<skill_name> <message> - load a skill and run the message")
    return "\n".join(rows)


async def run_tui_chat(args: argparse.Namespace) -> int:
    app = SoongAgentTui(args)
    result = await app.run_async()
    return int(result or 0)
