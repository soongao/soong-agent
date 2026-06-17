from __future__ import annotations

from agent_cli.tui.commands.branch import BranchCommandsMixin
from agent_cli.tui.commands.meta import MetaCommandsMixin
from agent_cli.tui.commands.session import SessionCommandsMixin
from agent_cli.tui.commands.skills import SkillCommandsMixin
from agent_cli.tui.slash import parse_slash_command, slash_help_text
from agent_cli.tui.types import SlashCommandResult


class CommandMixin(SessionCommandsMixin, BranchCommandsMixin, SkillCommandsMixin, MetaCommandsMixin):
    async def _handle_slash_command(self, message: str) -> SlashCommandResult:
        command, argument = parse_slash_command(message)
        if command in {"exit", "quit"}:
            self.exit(0)
            return SlashCommandResult(handled=True)
        if command == "help":
            await self._write_message("system", slash_help_text())
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
