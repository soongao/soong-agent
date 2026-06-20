from __future__ import annotations

import os
import shlex
import shutil
from pathlib import Path
from typing import Any

from agent_core.api.runtime_helpers.agents.worker_executor import WorkerExecutorContext
from agent_core.errors import AgentCoreError
from agent_core.errors.codes import ErrorCode

from agent_hub.backend.workers.pty import PtySessionManager, PtyWorkerExecutorBase


class ClaudeCodePtyWorkerExecutor(PtyWorkerExecutorBase):
    def __init__(
        self,
        *,
        pty_manager: PtySessionManager,
        project_dir: Path,
    ) -> None:
        super().__init__(executor_type="claude_code_pty", pty_manager=pty_manager, project_dir=project_dir)

    def build_command(self, config: dict[str, Any], *, cwd: Path, initial_prompt: str | None = None) -> list[str]:
        return _claude_code_command(
            config,
            cwd=cwd,
            initial_prompt=initial_prompt if self.command_includes_initial_prompt(config) else None,
        )

    def prompt_text(self, context: WorkerExecutorContext, *, completion_marker: str) -> str:
        return _prompt_text(context, completion_marker=completion_marker)

    def default_completed_text(self, context: WorkerExecutorContext) -> str:
        return "Claude Code PTY worker completed."

    def format_turn_input(self, prompt: str, config: dict[str, Any]) -> str:
        if config.get("bracketed_paste", False) is True:
            return f"\x1b[200~{prompt}\x1b[201~"
        return prompt

    def command_includes_initial_prompt(self, config: dict[str, Any]) -> bool:
        return config.get("initial_prompt_arg", True) is not False

    def input_suffix(self, config: dict[str, Any]) -> str:
        return "\r"

    def active_input_suffix(self, config: dict[str, Any]) -> str:
        return "\r"

    def startup_delay_seconds(self, config: dict[str, Any]) -> float:
        raw = config.get("startup_delay_ms", 0)
        if isinstance(raw, int | float):
            return max(float(raw) / 1000.0, 0.0)
        return 0.0

    def startup_ready_pattern(self, config: dict[str, Any]) -> str | None:
        raw = config.get("startup_ready_pattern")
        if isinstance(raw, str) and raw:
            return raw
        return None

    def startup_ready_timeout_seconds(self, config: dict[str, Any]) -> float:
        raw = config.get("startup_ready_timeout_ms", 0)
        if isinstance(raw, int | float):
            return max(float(raw) / 1000.0, 0.0)
        return 0.0

    def environment(self, config: dict[str, Any], *, cwd: Path) -> dict[str, str] | None:
        env = dict(os.environ)
        env["TERM"] = str(config.get("term") or env.get("TERM") or "xterm-256color")
        if env["TERM"] == "dumb":
            env["TERM"] = "xterm-256color"
        env.setdefault("COLORTERM", "truecolor")
        return env

    def transform_output(self, text: str) -> str:
        return self.collapse_tui_repaint_text(super().transform_output(text))


def _claude_code_command(config: dict[str, Any], *, cwd: Path, initial_prompt: str | None = None) -> list[str]:
    raw = config.get("command")
    if isinstance(raw, list) and raw:
        args = [str(item) for item in raw]
        if initial_prompt:
            args.append(initial_prompt)
        return args
    if isinstance(raw, str) and raw.strip():
        args = shlex.split(raw)
        if initial_prompt:
            args.append(initial_prompt)
        return args
    binary = str(config.get("binary") or shutil.which("claude") or "")
    if not binary:
        raise AgentCoreError(ErrorCode.WORKER_NOT_AVAILABLE, "claude command is not available on PATH")
    args = [binary]
    if config.get("ax_screen_reader", True) is not False:
        args.append("--ax-screen-reader")
    model = config.get("model")
    if isinstance(model, str) and model.strip():
        args.extend(["--model", model.strip()])
    permission_mode = config.get("permission_mode")
    if isinstance(permission_mode, str) and permission_mode.strip():
        args.extend(["--permission-mode", permission_mode.strip()])
    for extra in config.get("args") or []:
        args.append(str(extra))
    if initial_prompt:
        args.append(initial_prompt)
    return args


def _prompt_text(context: WorkerExecutorContext, *, completion_marker: str) -> str:
    sections = [
        context.instruction.strip(),
        "",
        "Agent Hub worker metadata:",
        f"- task_id: {context.task_id}",
    ]
    if context.dispatch_context:
        sections.extend(["", "Dispatch context:", context.dispatch_context.strip()])
    sections.extend(
        [
            "",
            "When this worker turn is fully complete, print the completion marker on its own line.",
            "Build the marker by concatenating these parts without spaces, quotes, or code fences:",
            _completion_marker_parts(completion_marker),
            "Do not print the marker until you have finished the requested work.",
        ]
    )
    return "\n".join(sections).strip()


def _completion_marker_parts(completion_marker: str) -> str:
    prefix = "<<<AGENTHUB_DONE:"
    suffix = ">>>"
    if completion_marker.startswith(prefix) and completion_marker.endswith(suffix):
        run_id = completion_marker[len(prefix) : -len(suffix)]
        return "\n".join(
            [
                "- first part: three `<` characters, then `AGENTHUB_DONE`, then `:`",
                f"- second part: `{run_id}`",
                "- third part: three `>` characters",
            ]
        )
    midpoint = max(len(completion_marker) // 2, 1)
    return "\n".join(
        [
            f"- first part: `{completion_marker[:midpoint]}`",
            f"- second part: `{completion_marker[midpoint:]}`",
        ]
    )
