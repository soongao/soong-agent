from __future__ import annotations

import argparse
import asyncio
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="soong-agent")
    sub = parser.add_subparsers(dest="command", required=True)
    chat = sub.add_parser("chat")
    chat.add_argument("--path", type=str, default=None)
    chat.add_argument("--orchestrator", action="store_true")
    chat.add_argument("--session-id", type=str, default=None)
    chat.add_argument("--plain", action="store_true", help="use plain stdin/stdout chat instead of the TUI")
    chat.add_argument("--debug-events", action="store_true", help="show debug runtime events in the TUI")
    return parser


async def async_main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "chat":
        if args.plain:
            return await _run_plain_chat(args)
        return await _run_tui_chat(args)
    return 1


async def _run_plain_chat(args: argparse.Namespace) -> int:
    from agent_cli.plain import run_plain_chat

    return await run_plain_chat(args)


async def _run_tui_chat(args: argparse.Namespace) -> int:
    try:
        from agent_cli.tui import run_tui_chat
    except ModuleNotFoundError as exc:
        if exc.name == "textual":
            print("error: TUI mode requires the textual package. Use --plain or install project dependencies.", file=sys.stderr)
            return 1
        raise

    return await run_tui_chat(args)


def main(argv: list[str] | None = None) -> int:
    try:
        return asyncio.run(async_main(argv))
    except KeyboardInterrupt:
        print()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
