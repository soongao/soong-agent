#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import time


def write(message: dict) -> None:
    sys.stdout.write(json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def read() -> dict:
    line = sys.stdin.readline()
    if not line:
        raise SystemExit(0)
    return json.loads(line)


def event(request_id: int, thread_id: str, msg: dict) -> None:
    write(
        {
            "jsonrpc": "2.0",
            "method": "codex/event",
            "params": {
                "_meta": {"requestId": request_id, "threadId": thread_id},
                "id": str(request_id),
                "msg": msg,
            },
        }
    )


def main() -> int:
    while True:
        message = read()
        method = message.get("method")
        request_id = message.get("id")
        if method == "initialize":
            write(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {"listChanged": True}},
                        "serverInfo": {"name": "fake-codex-mcp-server", "version": "0.0.0"},
                    },
                }
            )
        elif method == "notifications/initialized":
            continue
        elif method == "tools/call":
            params = message.get("params") if isinstance(message.get("params"), dict) else {}
            arguments = params.get("arguments") if isinstance(params.get("arguments"), dict) else {}
            prompt = str(arguments.get("prompt") or "")
            thread_id = str(arguments.get("threadId") or "thread_fake_1")
            event(request_id, thread_id, {"type": "task_started", "started_at": int(time.time())})
            event(
                request_id,
                thread_id,
                {
                    "type": "item_started",
                    "started_at_ms": 1000,
                    "item": {"type": "AgentMessage", "phase": "final_answer", "content": [{"type": "Text", "text": ""}]},
                },
            )
            if "approval" in prompt:
                command = ["/bin/bash", "-lc", "touch approval_probe.txt"]
                event(
                    request_id,
                    thread_id,
                    {
                        "type": "exec_approval_request",
                        "started_at_ms": 1100,
                        "call_id": "call_fake",
                        "command": command,
                        "cwd": str(arguments.get("cwd") or ""),
                        "reason": "fake approval request",
                    },
                )
                write(
                    {
                        "jsonrpc": "2.0",
                        "id": 99,
                        "method": "elicitation/create",
                        "params": {
                            "message": "Allow fake command?",
                            "threadId": thread_id,
                            "codex_elicitation": "exec-approval",
                            "codex_call_id": "call_fake",
                            "codex_command": command,
                            "codex_cwd": str(arguments.get("cwd") or ""),
                            "codex_parsed_cmd": [{"type": "unknown", "cmd": "touch approval_probe.txt"}],
                        },
                    }
                )
                approval = read()
                decision = ((approval.get("result") or {}).get("decision") if isinstance(approval.get("result"), dict) else None) or "denied"
                text = f"approval decision: {decision}"
            else:
                event(request_id, thread_id, {"type": "agent_message_content_delta", "delta": "codex-"})
                event(request_id, thread_id, {"type": "agent_message_content_delta", "delta": "mcp-ok"})
                text = "codex-mcp-ok"
            event(
                request_id,
                thread_id,
                {
                    "type": "item_completed",
                    "completed_at_ms": 1200,
                    "item": {
                        "type": "AgentMessage",
                        "phase": "final_answer",
                        "content": [{"type": "Text", "text": text}],
                    },
                },
            )
            event(request_id, thread_id, {"type": "task_complete"})
            write(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "content": [{"type": "text", "text": text}],
                        "structuredContent": {"threadId": thread_id, "content": text},
                    },
                }
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
