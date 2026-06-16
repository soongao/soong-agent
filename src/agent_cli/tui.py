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


SLASH_SUGGESTION_VISIBLE_ROWS = 8


@dataclass(frozen=True)
class SlashSuggestion:
    completion: str
    usage: str
    description: str


class PromptTextArea(TextArea):
    async def _on_key(self, event: Key) -> None:
        if event.key not in {"up", "down"}:
            self.app._history_browsing = False  # type: ignore[attr-defined]
        if event.key == "enter":
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
        if not select and self.app._move_slash_selection(-1):  # type: ignore[attr-defined]
            return
        if not select and self.cursor_location[0] == 0:
            self.app._show_history(-1)  # type: ignore[attr-defined]
            return
        super().action_cursor_up(select)

    def action_cursor_down(self, select: bool = False) -> None:
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
        max-height: 9;
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
        self._assistant_stream_widget: Static | None = None
        self._assistant_stream_text = ""
        self._history: list[str] = []
        self._history_index: int | None = None
        self._draft_message = ""
        self._auto_scroll = True
        self._run_count = 0
        self._slash_suggestions: list[SlashSuggestion] = []
        self._slash_selected_index = 0
        self._slash_window_start = 0
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
        user_message = message
        if user_message.startswith("/") and await self._handle_slash_command(user_message):
            self._record_history(user_message)
            prompt.clear()
            self._hide_slash_suggestions()
            prompt.focus()
            return
        if self._has_active_run():
            await self._write_message("warning", "run already active; wait or press Ctrl-C to cancel")
            return
        await self._ensure_runtime()
        assert self.runtime is not None
        self._close_assistant_stream()
        self._record_history(user_message)
        prompt.clear()
        self._hide_slash_suggestions()
        await self._write_message("user", user_message)
        prompt.disabled = True
        self._set_status("running")
        try:
            handle = await self.runtime.start(user_message, session_id=self.session_id, mode=self.mode)
        except AgentCoreError as exc:
            await self._write_message("error", exc.message)
            prompt.disabled = False
            prompt.focus()
            self._set_status("error")
            return
        self.current_handle = handle
        self.event_task = asyncio.create_task(self._consume_events(handle))

    async def action_cancel_or_quit(self) -> None:
        if self._has_active_run():
            await self._cancel_active_run()
            return
        self.exit(0)

    def action_toggle_auto_scroll(self) -> None:
        self._auto_scroll = not self._auto_scroll
        self._set_status(self.current_handle.status.value if self.current_handle is not None else "idle")

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area.id == "prompt":
            if self._history_browsing:
                self._hide_slash_suggestions()
                return
            self._update_slash_suggestions(event.text_area.text)

    def _update_slash_suggestions(self, text: str) -> None:
        widget = self.query_one("#slash-suggestions", Static)
        command_text = text.lstrip()
        stripped = command_text.strip()
        if not command_text.startswith("/") or "\n" in command_text:
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
        self._sync_slash_window()
        self._render_slash_suggestions()
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
                completion=f"/{skill['name']}",
                usage=f"/{skill['name']}",
                description=skill.get("description") or "load skill",
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
        self._sync_slash_window()
        visible = self._visible_slash_suggestions()
        rows = ["Slash commands (Up/Down selects, Enter/Tab accepts)"]
        for index, suggestion in visible:
            marker = ">" if index == self._slash_selected_index else " "
            rows.append(f"{marker} {suggestion.usage:<18} {suggestion.description}")
        widget.display = True
        widget.update("\n".join(rows))

    def _visible_slash_suggestions(self) -> list[tuple[int, SlashSuggestion]]:
        end = self._slash_window_start + SLASH_SUGGESTION_VISIBLE_ROWS
        return list(enumerate(self._slash_suggestions))[self._slash_window_start : end]

    def _sync_slash_window(self) -> None:
        if not self._slash_suggestions:
            self._slash_window_start = 0
            return
        max_start = max(len(self._slash_suggestions) - SLASH_SUGGESTION_VISIBLE_ROWS, 0)
        if self._slash_selected_index < self._slash_window_start:
            self._slash_window_start = self._slash_selected_index
        elif self._slash_selected_index >= self._slash_window_start + SLASH_SUGGESTION_VISIBLE_ROWS:
            self._slash_window_start = self._slash_selected_index - SLASH_SUGGESTION_VISIBLE_ROWS + 1
        self._slash_window_start = min(max(self._slash_window_start, 0), max_start)

    def _cached_skills_for_suggestions(self) -> list[dict[str, str]]:
        if self.runtime is not None and self.runtime.paths is not None:
            home_dir = self.runtime.paths.home_dir
        else:
            home_dir = resolve_home_dir()
        return build_skill_catalog(home_dir)

    async def _handle_slash_command(self, message: str) -> bool:
        command, argument = _parse_slash_command(message)
        if command in {"exit", "quit"}:
            self.exit(0)
            return True
        if command == "help":
            await self._write_message("system", _slash_help_text())
            return True
        if command == "clear":
            if self._has_active_run():
                await self._write_message("warning", "cannot clear the transcript while a run is active; use /cancel first")
                return True
            await self._clear_transcript()
            await self._write_message("system", "transcript cleared")
            return True
        if command == "new":
            if self._has_active_run():
                await self._write_message("warning", "cannot start a new session while a run is active; use /cancel first")
                return True
            await self._new_session(argument)
            return True
        if command == "session":
            await self._write_message("system", self._session_info_text())
            return True
        if command == "config":
            await self._write_message("system", self._config_info_text())
            return True
        if command == "skills":
            await self._show_skills()
            return True
        if command == "history":
            await self._write_message("system", self._history_text(argument))
            return True
        if command == "autoscroll":
            self.action_toggle_auto_scroll()
            await self._write_message("system", f"autoscroll {'on' if self._auto_scroll else 'off'}")
            return True
        if command == "cancel":
            await self._cancel_active_run()
            return True
        if not argument and self._skill_name_exists(command) and await self._load_skill_command(command):
            return True
        await self._write_message("warning", f"unknown slash command: /{command}. Use /help for commands.")
        return True

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
        self._run_count = 0
        self._set_status("idle")
        await self._write_message("system", f"new session: {self.session_id}\nprevious session: {previous}")

    async def _cancel_active_run(self) -> None:
        if not self._has_active_run():
            await self._write_message("warning", "no active run to cancel")
            return
        self._close_assistant_stream()
        await self._write_message("warning", "cancelling current run")
        assert self.current_handle is not None
        await self.current_handle.cancel()

    def _session_info_text(self) -> str:
        project_path = self._project_path_text()
        model = getattr(getattr(self.runtime, "config", None), "model", None)
        provider = getattr(model, "provider", "not loaded")
        model_name = getattr(model, "name", "not loaded")
        state = self.current_handle.status.value if self.current_handle is not None else "idle"
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
        await self._write_message("system", "\n".join(rows))

    async def _load_skill_command(self, name: str) -> bool:
        if self._has_active_run():
            await self._write_message("warning", "cannot load a skill while a run is active; use /cancel first")
            return True
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
            return True
        if result.already_loaded:
            await self._write_message("system", f"skill already loaded: {result.name}")
            return True
        await self._write_message("system", f"skill loaded: {result.name}")
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
        return self.current_handle is not None and self.current_handle.status in {
            RunStatus.PENDING,
            RunStatus.RUNNING,
            RunStatus.QUEUED,
        }

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
                self._run_count += 1
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
        return f"session: {self.session_id} | model: {model_name} | mode: {self.mode} | state: {state} | runs: {self._run_count} | autoscroll: {scroll}"


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
    ("session", "/session", "show current session details"),
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
    return "\n".join(rows)


async def run_tui_chat(args: argparse.Namespace) -> int:
    app = SoongAgentTui(args)
    result = await app.run_async()
    return int(result or 0)
