from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger(__name__)

ReverseRequestHandler = Callable[[str, dict[str, Any]], Awaitable[Any]]
NotificationHandler = Callable[[str, dict[str, Any]], Awaitable[None]]


class JsonRpcProcess:
    def __init__(
        self,
        *,
        command: list[str],
        reverse_request_handler: ReverseRequestHandler,
        notification_handler: NotificationHandler,
        log_name: str = "jsonrpc",
    ) -> None:
        self.command = command
        self._reverse_request_handler = reverse_request_handler
        self._notification_handler = notification_handler
        self._log_name = log_name
        self._proc: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._next_id = 1
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._write_lock = asyncio.Lock()
        self._stderr_lines: deque[str] = deque(maxlen=20)

    async def start(self) -> None:
        self._proc = await asyncio.create_subprocess_exec(
            *self.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._reader_task = asyncio.create_task(self._read_stdout())
        self._stderr_task = asyncio.create_task(self._read_stderr())

    async def close(self) -> None:
        proc = self._proc
        if proc is not None and proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=2)
            except TimeoutError:
                proc.kill()
                await proc.wait()
        for task in (self._reader_task, self._stderr_task):
            if task is not None:
                task.cancel()
        await asyncio.gather(*(task for task in (self._reader_task, self._stderr_task) if task is not None), return_exceptions=True)
        self._fail_pending(RuntimeError("JSON-RPC process closed"))

    async def request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        request_id = self._next_id
        self._next_id += 1
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        await self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}})
        return await future

    async def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        await self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    async def _send_result(self, request_id: Any, result: Any) -> None:
        await self._send({"jsonrpc": "2.0", "id": request_id, "result": result if result is not None else {}})

    async def _send_error(self, request_id: Any, code: int, message: str, data: Any | None = None) -> None:
        error: dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            error["data"] = data
        await self._send({"jsonrpc": "2.0", "id": request_id, "error": error})

    async def _send(self, payload: dict[str, Any]) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None or proc.returncode is not None:
            raise RuntimeError("JSON-RPC process is not running")
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
        async with self._write_lock:
            proc.stdin.write(line)
            await proc.stdin.drain()

    async def _read_stdout(self) -> None:
        assert self._proc and self._proc.stdout
        while True:
            line = await self._proc.stdout.readline()
            if not line:
                self._fail_pending(RuntimeError(self._closed_message("JSON-RPC stdout closed")))
                return
            try:
                message = json.loads(line.decode("utf-8"))
            except Exception:
                logger.warning("ignoring malformed JSON-RPC line from %s: %r", self._log_name, line[:500])
                continue
            await self._handle_message(message)

    async def _read_stderr(self) -> None:
        assert self._proc and self._proc.stderr
        while True:
            line = await self._proc.stderr.readline()
            if not line:
                return
            text = line.decode("utf-8", errors="replace").rstrip()
            self._stderr_lines.append(text)
            logger.debug("%s stderr: %s", self._log_name, text)

    async def _handle_message(self, message: dict[str, Any]) -> None:
        if "id" in message and ("result" in message or "error" in message) and "method" not in message:
            request_id = message.get("id")
            future = self._pending.pop(int(request_id), None) if isinstance(request_id, int) else None
            if future is None or future.done():
                return
            if "error" in message:
                error = message.get("error") if isinstance(message.get("error"), dict) else {}
                future.set_exception(RuntimeError(str(error.get("message") or "JSON-RPC request failed")))
            else:
                future.set_result(message.get("result"))
            return

        method = message.get("method")
        if not isinstance(method, str):
            return
        params = message.get("params")
        if not isinstance(params, dict):
            params = {}
        if "id" in message:
            request_id = message["id"]
            try:
                result = await self._reverse_request_handler(method, params)
            except Exception as exc:
                await self._send_error(request_id, -32000, str(exc))
            else:
                await self._send_result(request_id, result)
            return
        await self._notification_handler(method, params)

    def _fail_pending(self, exc: Exception) -> None:
        for future in self._pending.values():
            if not future.done():
                future.set_exception(exc)
        self._pending.clear()

    def _closed_message(self, message: str) -> str:
        proc = self._proc
        details: list[str] = []
        if proc is not None and proc.returncode is not None:
            details.append(f"exit_code={proc.returncode}")
        if self._stderr_lines:
            details.append("stderr=" + "\n".join(self._stderr_lines))
        if details:
            return f"{message} ({'; '.join(details)})"
        return message
