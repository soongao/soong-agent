from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from agent_core.api import AgentRuntime
from agent_core.errors import AgentCoreError
from agent_core.permissions import stdin_permission_callback


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="soong-agent")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--path", type=str, default=None)
    run.add_argument("--orchestrator", action="store_true")
    run.add_argument("--session-id", type=str, default=None)
    run.add_argument("message", nargs="+")
    return parser


async def async_main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        message = " ".join(args.message)
        try:
            async with AgentRuntime(
                project_dir=Path(args.path) if args.path else None,
                permission_callback=stdin_permission_callback,
            ) as runtime:
                handle = await runtime.start(
                    message,
                    session_id=args.session_id,
                    mode="orchestrator" if args.orchestrator else "normal",
                )
                async for event in handle.events():
                    if event.event_type == "model_text_delta":
                        print(event.payload.get("text", ""), end="", flush=True)
                    elif event.event_type == "loop_failed":
                        print(f"\nerror: {event.payload.get('message')}", file=sys.stderr)
                print()
                return 0 if handle.status.value == "completed" else 1
        except AgentCoreError as exc:
            print(f"error: {exc.message}", file=sys.stderr)
            return 1
    return 1


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(async_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())

