from __future__ import annotations

from agent_core.types import WorkerConfigCreate


DEFAULT_HUB_WORKERS: tuple[WorkerConfigCreate, ...] = (
    WorkerConfigCreate(
        worker_id="code_reviewer",
        name="Code Reviewer",
        description="Reviews code for correctness, regressions, maintainability risks, and missing tests.",
        system_prompt=(
            "You are a senior code reviewer. Focus on concrete bugs, behavioral regressions, unsafe assumptions, "
            "and missing verification. Return concise findings with file references when possible."
        ),
        allowed_tools=["code.read_file", "code.list_dir", "code.search"],
        metadata={"agenthub_default_worker": True},
    ),
    WorkerConfigCreate(
        worker_id="doc_writer",
        name="Doc Writer",
        description="Writes and edits project documentation, plans, and concise implementation notes.",
        system_prompt=(
            "You are a documentation worker. Produce clear, structured Markdown that is specific to the repository. "
            "Keep prose concise, preserve technical details, and write files only when the task asks for an artifact."
        ),
        allowed_tools=["code.read_file", "code.list_dir", "code.search", "code.write_file", "code.edit_file"],
        metadata={"agenthub_default_worker": True},
    ),
    WorkerConfigCreate(
        worker_id="test_writer",
        name="Test Writer",
        description="Adds and fixes focused tests, then runs targeted verification commands.",
        system_prompt=(
            "You are a test worker. Add focused tests that cover the requested behavior, prefer existing test patterns, "
            "and run the narrowest useful verification command before reporting results."
        ),
        allowed_tools=["code.read_file", "code.list_dir", "code.search", "code.write_file", "code.edit_file", "code.run_command"],
        metadata={"agenthub_default_worker": True},
    ),
    WorkerConfigCreate(
        worker_id="opencode_worker",
        name="OpenCode Worker",
        description="Delegates coding tasks to the local OpenCode ACP agent while preserving an OpenCode session per Hub conversation.",
        system_prompt=(
            "You are an external OpenCode worker. Treat the orchestrator dispatch as the user's request, "
            "work in the current project, and return the result clearly."
        ),
        allowed_tools=["opencode.acp"],
        metadata={
            "agenthub_default_worker": True,
            "worker_executor": {
                "type": "opencode",
                "config": {},
            },
        },
    ),
    WorkerConfigCreate(
        worker_id="codex_pty_worker",
        name="Codex PTY Worker",
        description="Delegates tasks to the local Codex interactive CLI through a reusable PTY session with streamed output.",
        system_prompt=(
            "You are an external Codex PTY worker. Treat the orchestrator dispatch as the user's request. "
            "The Hub streams your terminal output to the user; if you need permission input, wait for the user's reply."
        ),
        allowed_tools=["codex.pty"],
        metadata={
            "agenthub_default_worker": True,
            "worker_executor": {
                "type": "codex_pty",
                "config": {},
            },
        },
    ),
    WorkerConfigCreate(
        worker_id="claude_code_pty_worker",
        name="Claude Code PTY Worker",
        description="Delegates tasks to the local Claude Code interactive CLI through a reusable PTY session with streamed output.",
        system_prompt=(
            "You are an external Claude Code PTY worker. Treat the orchestrator dispatch as the user's request. "
            "The Hub streams your terminal output to the user; if you need permission input, wait for the user's reply."
        ),
        allowed_tools=["claude_code.pty"],
        metadata={
            "agenthub_default_worker": True,
            "worker_executor": {
                "type": "claude_code_pty",
                "config": {},
            },
        },
    ),
)
