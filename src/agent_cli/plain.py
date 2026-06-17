from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Literal

from agent_core.api import AgentRuntime, RunHandle
from agent_core.errors import AgentCoreError
from agent_core.permissions import stdin_permission_callback
from agent_core.storage import new_id
from agent_core.types.runtime import RunStatus


PROMPT = "agentcli> "
EXIT_COMMANDS = {"/exit", "/quit"}


async def run_plain_chat(args: argparse.Namespace) -> int:
    session_id = args.session_id or new_id("sess")
    mode: Literal["normal", "orchestrator"] = "orchestrator" if args.orchestrator else "normal"
    runtime: AgentRuntime | None = None
    try:
        while True:
            try:
                line = await asyncio.to_thread(_readline)
            except KeyboardInterrupt:
                print()
                return 0
            if line == "":
                print()
                return 0
            message = line.strip()
            if not message:
                continue
            if message in EXIT_COMMANDS:
                return 0
            if runtime is None:
                runtime = AgentRuntime(
                    project_dir=Path(args.path) if args.path else None,
                    permission_callback=stdin_permission_callback,
                )
                await runtime.__aenter__()
            code = await _run_turn(runtime, message=message, session_id=session_id, mode=mode)
            print()
            if code != 0:
                return code
    except KeyboardInterrupt:
        print()
        return 0
    except AgentCoreError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        return 1
    finally:
        if runtime is not None:
            await runtime.close()


def _readline() -> str:
    print(PROMPT, end="", flush=True)
    return sys.stdin.readline()


async def _run_turn(
    runtime: AgentRuntime, *, message: str, session_id: str, mode: Literal["normal", "orchestrator"]
) -> int:
    handle = await runtime.start(message, session_id=session_id, mode=mode)
    await _stream_handle(handle)
    return 0 if handle.status == RunStatus.COMPLETED else 1


async def _stream_handle(handle: RunHandle) -> None:
    async for event in handle.events():
        if event.event_type == "model_text_delta":
            print(event.payload.get("text", ""), end="", flush=True)
        elif event.event_type == "loop_failed":
            print(f"\nerror: {event.payload.get('message')}", file=sys.stderr)
