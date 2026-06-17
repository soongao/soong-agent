from __future__ import annotations

from agent_core.config.paths import resolve_home_dir, resolve_project_dir
from agent_core.errors import AgentCoreError

from agent_cli.tui.slash import parse_positive_int


class MetaCommandsMixin:
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

    def _history_text(self, argument: str = "") -> str:
        limit = parse_positive_int(argument.strip()) or 10
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
