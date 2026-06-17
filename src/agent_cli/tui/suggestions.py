from __future__ import annotations

from agent_core.context.skills import build_skill_catalog
from agent_core.config.paths import resolve_home_dir

from agent_cli.tui.formatting import NODE_ID_DISPLAY_LENGTH, short_id
from agent_cli.tui.slash import SLASH_COMMANDS, matching_slash_commands
from agent_cli.tui.types import BranchCandidate, SlashSuggestion
from agent_cli.tui.widgets import PromptTextArea

from textual.widgets import Static


SLASH_SUGGESTION_VISIBLE_ROWS = 6


class SuggestionMixin:
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
            for name, usage, description in matching_slash_commands(text.strip())
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
            rows.append(f"{selector}{active} {short_id(candidate.node_id, NODE_ID_DISPLAY_LENGTH):<{NODE_ID_DISPLAY_LENGTH}} {candidate.preview}")
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
