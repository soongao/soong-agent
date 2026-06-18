from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from agent_core.api.runtime_helpers.agents.worker_executor import WorkerExecutorContext, WorkerExecutorResult
from agent_core.errors import AgentCoreError
from agent_core.errors.codes import ErrorCode

from agent_hub.backend.workers.pty.manager import PtySessionKey, PtySessionManager

TERMINAL_CONTROL_RE = re.compile(
    "|".join(
        [
            r"\x1b\][^\x07]*(?:\x07|\x1b\\)",  # OSC, such as terminal title updates.
            r"\x1b[P^_].*?\x1b\\",  # DCS/PM/APC string controls.
            r"\x1b\[[0-?]*[ -/]*[@-~]",  # CSI sequences.
            r"\x1b[()][0-2AB]",  # Character set selection.
            r"\x1b[=>78]",  # Single-character terminal controls, including save/restore cursor.
            r"\x1b[@-Z\\-_]",  # Other single-character ESC controls.
        ]
    )
)
TUI_CHAR_LINE_RE = re.compile(r"^\s*(\S)\s*$")


class PtyWorkerExecutorBase(ABC):
    def __init__(
        self,
        *,
        executor_type: str,
        pty_manager: PtySessionManager,
        project_dir: Path,
    ) -> None:
        self.executor_type = executor_type
        self._pty_manager = pty_manager
        self._project_dir = project_dir.resolve()

    async def run(self, runtime: Any, context: WorkerExecutorContext) -> WorkerExecutorResult:
        config = dict(context.executor_config or {})
        cwd = self.resolve_cwd(config)
        marker = self.completion_marker(context)
        prompt = self.prompt_text(context, completion_marker=marker)
        session_key = PtySessionKey(
            core_session_id=context.session_id,
            worker_id=context.worker.worker_id,
            executor_type=self.executor_type,
        )
        existing_session = self._pty_manager.session(session_key)
        include_initial_prompt = existing_session is None and self.command_includes_initial_prompt(config)
        command = self.build_command(config, cwd=cwd, initial_prompt=prompt if include_initial_prompt else None)

        async def on_output(text: str) -> None:
            await runtime._emit_child_run_event(
                stream=context.worker_stream,
                mirror_handle=context.parent_handle,
                session_id=context.session_id,
                agent_id=context.worker_agent_id,
                run_id=context.worker_run_id,
                event_type="worker_text_delta",
                payload={
                    "text": text,
                    "worker_id": context.worker.worker_id,
                    "worker_agent_id": context.worker_agent_id,
                    "child_run_id": context.worker_run_id,
                },
            )

        session = await self._pty_manager.get_or_create(
            session_key,
            command=command,
            cwd=cwd,
            env=self.environment(config, cwd=cwd),
            output_transform=self.transform_output,
        )
        try:
            prompt_is_in_command = include_initial_prompt and session.turn_count == 0
            turn_input = "" if prompt_is_in_command else self.format_turn_input(prompt, config)
            final_text = await session.run_turn(
                worker_run_id=context.worker_run_id,
                prompt=turn_input,
                completion_marker=marker,
                output_callback=on_output,
                input_suffix="" if prompt_is_in_command else self.input_suffix(config),
                active_input_suffix=self.active_input_suffix(config),
                startup_delay_seconds=0.0 if prompt_is_in_command else self.startup_delay_seconds(config),
                startup_ready_pattern=None if prompt_is_in_command else self.startup_ready_pattern(config),
                startup_ready_timeout_seconds=0.0 if prompt_is_in_command else self.startup_ready_timeout_seconds(config),
            )
        except AgentCoreError:
            raise
        except Exception as exc:
            raise AgentCoreError(ErrorCode.WORKER_NOT_AVAILABLE, f"{self.executor_type} pty worker failed: {exc}") from exc
        if not final_text:
            final_text = self.default_completed_text(context)
        return WorkerExecutorResult(
            text=final_text,
            data=self.result_data(context, final_text=final_text, completion_marker=marker, cwd=cwd),
            metadata=self.result_metadata(context, final_text=final_text, completion_marker=marker, cwd=cwd),
        )

    async def close(self) -> None:
        return None

    def resolve_cwd(self, config: dict[str, Any]) -> Path:
        cwd = Path(str(config.get("cwd") or self._project_dir)).expanduser().resolve()
        if not cwd.exists():
            raise AgentCoreError(ErrorCode.VALIDATION_ERROR, f"{self.executor_type} cwd does not exist: {cwd}")
        return cwd

    def completion_marker(self, context: WorkerExecutorContext) -> str:
        return f"<<<AGENTHUB_DONE:{context.worker_run_id}>>>"

    def environment(self, config: dict[str, Any], *, cwd: Path) -> dict[str, str] | None:
        return None

    def input_suffix(self, config: dict[str, Any]) -> str:
        return "\n"

    def active_input_suffix(self, config: dict[str, Any]) -> str:
        return "\n"

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

    def format_turn_input(self, prompt: str, config: dict[str, Any]) -> str:
        return prompt

    def command_includes_initial_prompt(self, config: dict[str, Any]) -> bool:
        return False

    def transform_output(self, text: str) -> str:
        return TERMINAL_CONTROL_RE.sub("", text).replace("\r\n", "\n").replace("\r", "")

    def collapse_tui_repaint_text(self, text: str) -> str:
        return collapse_tui_repaint_text(text)

    def default_completed_text(self, context: WorkerExecutorContext) -> str:
        return f"{self.executor_type} PTY worker completed."

    def result_data(
        self,
        context: WorkerExecutorContext,
        *,
        final_text: str,
        completion_marker: str,
        cwd: Path,
    ) -> dict[str, Any]:
        return {"text": final_text, "completion_marker": completion_marker}

    def result_metadata(
        self,
        context: WorkerExecutorContext,
        *,
        final_text: str,
        completion_marker: str,
        cwd: Path,
    ) -> dict[str, Any]:
        return {
            "external_executor": self.executor_type,
            "completion_marker": completion_marker,
            "cwd": str(cwd),
        }

    @abstractmethod
    def build_command(self, config: dict[str, Any], *, cwd: Path, initial_prompt: str | None = None) -> list[str]:
        ...

    @abstractmethod
    def prompt_text(self, context: WorkerExecutorContext, *, completion_marker: str) -> str:
        ...


def collapse_tui_repaint_text(text: str) -> str:
    lines = text.splitlines()
    output: list[str] = []
    char_run: list[str] = []

    def flush_char_run() -> None:
        if not char_run:
            return
        chunk = "".join(char_run)
        if output and output[-1] and not output[-1].endswith(("\n", " ")):
            output.append("\n")
        output.append(chunk)
        char_run.clear()

    for line in lines:
        stripped = line.strip()
        match = TUI_CHAR_LINE_RE.match(line)
        if match:
            char_run.append(match.group(1))
            continue
        if not stripped:
            continue
        flush_char_run()
        output.append(stripped)
        output.append("\n")
    flush_char_run()
    return "\n".join(part.rstrip() for part in "".join(output).splitlines() if part.strip()).strip()
