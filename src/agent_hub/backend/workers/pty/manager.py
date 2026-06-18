from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
import os
import pty
import re
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

OutputCallback = Callable[[str], Awaitable[None]]


@dataclasses.dataclass(frozen=True)
class PtySessionKey:
    core_session_id: str
    worker_id: str
    executor_type: str


@dataclasses.dataclass(frozen=True)
class PtyInputReceipt:
    core_session_id: str
    worker_id: str
    executor_type: str
    worker_run_id: str


@dataclasses.dataclass
class _ActiveTurn:
    worker_run_id: str
    completion_marker: str
    output_callback: OutputCallback
    result: asyncio.Future[str]
    active_input_suffix: str = "\n"
    startup_ready_pattern: re.Pattern[str] | None = None
    startup_ready: asyncio.Future[None] | None = None
    parts: list[str] = dataclasses.field(default_factory=list)
    pending: str = ""


class PtySessionManager:
    def __init__(self) -> None:
        self._sessions: dict[PtySessionKey, PtySession] = {}
        self._lock = asyncio.Lock()

    def session(self, key: PtySessionKey) -> "PtySession" | None:
        existing = self._sessions.get(key)
        if existing is not None and existing.running:
            return existing
        return None

    async def get_or_create(
        self,
        key: PtySessionKey,
        *,
        command: list[str],
        cwd: Path,
        env: dict[str, str] | None = None,
        output_transform: Callable[[str], str] | None = None,
    ) -> "PtySession":
        async with self._lock:
            existing = self._sessions.get(key)
            if existing is not None and existing.running:
                return existing
            if existing is not None:
                await existing.close()
            session = PtySession(
                key=key,
                command=command,
                cwd=cwd,
                env=env,
                output_transform=output_transform,
                on_exit=lambda: self._sessions.pop(key, None),
            )
            await session.start()
            self._sessions[key] = session
            return session

    async def write_to_active(self, *, core_session_id: str, worker_id: str, text: str) -> PtyInputReceipt | None:
        async with self._lock:
            sessions = list(self._sessions.items())
        for key, session in sessions:
            if key.core_session_id != core_session_id or key.worker_id != worker_id:
                continue
            receipt = await session.write_to_active(text)
            if receipt is not None:
                return receipt
        return None

    async def close(self) -> None:
        sessions = list(self._sessions.values())
        self._sessions.clear()
        await asyncio.gather(*(session.close() for session in sessions), return_exceptions=True)


class PtySession:
    _PRE_TURN_BUFFER_LIMIT = 65536

    def __init__(
        self,
        *,
        key: PtySessionKey,
        command: list[str],
        cwd: Path,
        env: dict[str, str] | None,
        output_transform: Callable[[str], str] | None,
        on_exit: Callable[[], None],
    ) -> None:
        self.key = key
        self.command = command
        self.cwd = cwd
        self.env = env
        self.output_transform = output_transform
        self.on_exit = on_exit
        self._process: asyncio.subprocess.Process | None = None
        self._master_fd: int | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._active_turn: _ActiveTurn | None = None
        self._pre_turn_output = ""
        self._turn_count = 0
        self._write_lock = asyncio.Lock()

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.returncode is None and self._master_fd is not None

    @property
    def turn_count(self) -> int:
        return self._turn_count

    async def start(self) -> None:
        if self.running:
            return
        self.cwd.mkdir(parents=True, exist_ok=True)
        master_fd, slave_fd = pty.openpty()
        try:
            self._process = await asyncio.create_subprocess_exec(
                *self.command,
                cwd=str(self.cwd),
                env=self.env,
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                start_new_session=True,
            )
        finally:
            os.close(slave_fd)
        self._master_fd = master_fd
        os.set_blocking(master_fd, False)
        self._reader_task = asyncio.create_task(self._read_loop())
        logger.info("pty session started executor=%s worker_id=%s pid=%s", self.key.executor_type, self.key.worker_id, self._process.pid if self._process else None)

    async def run_turn(
        self,
        *,
        worker_run_id: str,
        prompt: str,
        completion_marker: str,
        output_callback: OutputCallback,
        input_suffix: str = "\n",
        active_input_suffix: str = "\n",
        startup_delay_seconds: float = 0.0,
        startup_ready_pattern: str | None = None,
        startup_ready_timeout_seconds: float = 0.0,
    ) -> str:
        if not self.running:
            raise RuntimeError("pty session is not running")
        if self._active_turn is not None:
            raise RuntimeError("pty session already has an active turn")
        loop = asyncio.get_running_loop()
        turn = _ActiveTurn(
            worker_run_id=worker_run_id,
            completion_marker=completion_marker,
            output_callback=output_callback,
            result=loop.create_future(),
            active_input_suffix=active_input_suffix,
            startup_ready_pattern=re.compile(startup_ready_pattern) if startup_ready_pattern else None,
            startup_ready=loop.create_future() if startup_ready_pattern else None,
        )
        self._active_turn = turn
        self._turn_count += 1
        try:
            await self._flush_pre_turn_output(turn)
            if startup_delay_seconds > 0:
                await asyncio.sleep(startup_delay_seconds)
            if turn.startup_ready is not None and not turn.startup_ready.done():
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(turn.startup_ready, timeout=max(startup_ready_timeout_seconds, 0.0))
            await self.write(prompt + input_suffix)
            return await turn.result
        except asyncio.CancelledError:
            await self.interrupt()
            if not turn.result.done():
                turn.result.cancel()
            raise
        finally:
            if self._active_turn is turn:
                self._active_turn = None

    async def write_to_active(self, text: str) -> PtyInputReceipt | None:
        turn = self._active_turn
        if turn is None:
            return None
        await self.write(text + turn.active_input_suffix)
        return PtyInputReceipt(
            core_session_id=self.key.core_session_id,
            worker_id=self.key.worker_id,
            executor_type=self.key.executor_type,
            worker_run_id=turn.worker_run_id,
        )

    async def write(self, text: str) -> None:
        if self._master_fd is None:
            raise RuntimeError("pty session is closed")
        data = text.encode("utf-8", errors="replace")
        async with self._write_lock:
            await asyncio.to_thread(os.write, self._master_fd, data)

    async def interrupt(self) -> None:
        if self._master_fd is not None:
            with contextlib.suppress(OSError):
                await asyncio.to_thread(os.write, self._master_fd, b"\x03")

    async def close(self) -> None:
        reader_task = self._reader_task
        self._reader_task = None
        if reader_task is not None:
            reader_task.cancel()
            with contextlib.suppress(BaseException):
                await reader_task
        process = self._process
        self._process = None
        master_fd = self._master_fd
        self._master_fd = None
        if process is not None and process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.terminate()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(process.wait(), timeout=2)
            if process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
                await process.wait()
        if master_fd is not None:
            with contextlib.suppress(OSError):
                os.close(master_fd)

    async def _read_loop(self) -> None:
        try:
            while True:
                data = await self._read_once()
                if not data:
                    break
                text = data.decode("utf-8", errors="replace")
                if self.output_transform is not None:
                    text = self.output_transform(text)
                if text:
                    await self._handle_output(text)
        except asyncio.CancelledError:
            raise
        except OSError:
            logger.debug("pty read loop ended executor=%s worker_id=%s", self.key.executor_type, self.key.worker_id, exc_info=True)
        finally:
            self._finish_active_with_error(RuntimeError("pty process exited"))
            self.on_exit()

    async def _read_once(self) -> bytes:
        if self._master_fd is None:
            return b""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bytes] = loop.create_future()

        def on_readable() -> None:
            if future.done() or self._master_fd is None:
                return
            try:
                future.set_result(os.read(self._master_fd, 4096))
            except BlockingIOError:
                return
            except OSError as exc:
                future.set_exception(exc)

        loop.add_reader(self._master_fd, on_readable)
        try:
            return await future
        finally:
            if self._master_fd is not None:
                with contextlib.suppress(Exception):
                    loop.remove_reader(self._master_fd)

    async def _handle_output(self, text: str) -> None:
        turn = self._active_turn
        if turn is None:
            self._buffer_pre_turn_output(text)
            return
        marker = turn.completion_marker
        combined = turn.pending + text
        marker_span = _find_marker_span(combined, marker)
        if marker_span is not None:
            marker_start, _marker_end = marker_span
            before_marker = combined[:marker_start]
            if before_marker:
                await self._emit_turn_output(turn, before_marker)
            if not turn.result.done():
                turn.result.set_result("".join(turn.parts).strip())
            if self._active_turn is turn:
                self._active_turn = None
            return

        holdback = _partial_marker_holdback(combined, marker)
        emit_len = max(len(combined) - holdback, 0)
        if emit_len:
            await self._emit_turn_output(turn, combined[:emit_len])
        turn.pending = combined[emit_len:]

    async def _emit_turn_output(self, turn: _ActiveTurn, text: str) -> None:
        turn.parts.append(text)
        await turn.output_callback(text)
        if turn.startup_ready is not None and not turn.startup_ready.done() and turn.startup_ready_pattern is not None:
            if turn.startup_ready_pattern.search("".join(turn.parts)):
                turn.startup_ready.set_result(None)

    async def _flush_pre_turn_output(self, turn: _ActiveTurn) -> None:
        if not self._pre_turn_output:
            return
        text = self._pre_turn_output
        self._pre_turn_output = ""
        await self._emit_turn_output(turn, text)

    def _buffer_pre_turn_output(self, text: str) -> None:
        self._pre_turn_output = (self._pre_turn_output + text)[-self._PRE_TURN_BUFFER_LIMIT :]

    def _finish_active_with_error(self, exc: Exception) -> None:
        turn = self._active_turn
        self._active_turn = None
        if turn is not None and not turn.result.done():
            if turn.pending:
                turn.parts.append(turn.pending)
            turn.result.set_exception(exc)


def _find_marker_span(text: str, marker: str) -> tuple[int, int] | None:
    compact_chars: list[str] = []
    compact_to_original: list[int] = []
    for index, char in enumerate(text):
        if char.isspace():
            continue
        compact_chars.append(char)
        compact_to_original.append(index)
    compact = "".join(compact_chars)
    compact_index = compact.find(marker)
    if compact_index < 0:
        return None
    marker_start = compact_to_original[compact_index]
    marker_end = compact_to_original[compact_index + len(marker) - 1] + 1
    return marker_start, marker_end


def _partial_marker_holdback(text: str, marker: str) -> int:
    compact_chars: list[str] = []
    compact_to_original: list[int] = []
    for index, char in enumerate(text):
        if char.isspace():
            continue
        compact_chars.append(char)
        compact_to_original.append(index)
    compact = "".join(compact_chars)
    max_partial = min(len(compact), len(marker) - 1)
    for partial_len in range(max_partial, 0, -1):
        if compact.endswith(marker[:partial_len]):
            hold_from = compact_to_original[len(compact) - partial_len]
            return len(text) - hold_from
    return 0
