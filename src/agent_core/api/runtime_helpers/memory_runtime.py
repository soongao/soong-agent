from __future__ import annotations

import asyncio
import json
from typing import Any

from agent_core.api.handles import RunHandle
from agent_core.api.runtime_helpers.memory import (
    _estimate_memory_source_tokens,
    _memory_candidate_selector_line,
    _memory_extraction_source_text,
    _memory_frontmatter_candidates,
    _parse_json_object,
)
from agent_core.api.runtime_helpers.model import _ensure_provider_supports_request, _structured_json_request
from agent_core.artifacts.redaction import redact_value
from agent_core.config.loader import resolve_model_config
from agent_core.errors import AgentCoreError
from agent_core.errors.codes import ErrorCode
from agent_core.events import make_event
from agent_core.memory import MemoryExtractionJob, MemoryScanCursor, parse_memory_candidates, resolve_memory_dir
from agent_core.types import ErrorPayload, Node

_MEMORY_EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "memories": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "decision": {"type": "string", "enum": ["new"]},
                    "category": {"type": "string", "enum": ["user", "feedback", "reference"]},
                    "filename": {"type": "string", "pattern": r"^[a-z0-9][a-z0-9_-]*\.md$"},
                    "summary": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "source_node_ids": {"type": "array", "items": {"type": "string"}},
                    "content": {"type": "string"},
                },
                "required": ["decision", "category", "filename", "summary", "tags", "source_node_ids", "content"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["memories"],
    "additionalProperties": False,
}
_MEMORY_RECALL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "selected_paths": {
            "type": "array",
            "items": {"type": "string"},
        }
    },
    "required": ["selected_paths"],
    "additionalProperties": False,
}
_MEMORY_INTENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "explicit": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["explicit", "reason"],
    "additionalProperties": False,
}


async def select_memory(
    runtime: Any,
    *,
    session_id: str,
    query: str,
    top_k: int | None = None,
) -> dict[str, Any]:
    self = runtime
    await self._ensure_started()
    assert self.config and self.paths
    memory_root = resolve_memory_dir(
        self.config.memory.memory_dir,
        home_dir=self.paths.home_dir,
        project_dir=self.paths.project_dir,
    )
    candidates = _memory_frontmatter_candidates(memory_root)
    if not candidates:
        return {"selected_paths": [], "selected_by_model": False, "candidates": []}
    model_config = resolve_model_config(self.config, self.config.memory.recall_model_profile)
    result = await self._run_structured_json_model(
        model_config=model_config,
        schema=_MEMORY_RECALL_SCHEMA,
        purpose="memory_recall_selector",
        session_id=session_id,
        system_text=(
            "You are the Soong Agent memory recall selector. "
            "Select user-level memory files that may help answer the query. "
            "Match semantically using summaries, categories, tags, and filenames; do not require exact word overlap. "
            "For identity, profile, role, language, preference, or background questions, include memories describing the user. "
            "Return selected_paths with at most the requested top_k. Return [] only when no candidate is plausibly relevant. "
            "Do not invent paths."
        ),
        user_text=(
            f"Query: {query}\n"
            f"top_k: {top_k or self.config.memory.recall_top_k}\n"
            "Memory candidates:\n"
            + "\n".join(_memory_candidate_selector_line(item) for item in candidates)
        ),
        max_output_tokens=min(model_config.max_output_tokens, 1024),
    )
    selected = [str(path) for path in result.get("selected_paths") or []]
    allowed = {item["relative_path"]: item for item in candidates}
    max_items = top_k or self.config.memory.recall_top_k
    selected = [path for path in selected if path in allowed][:max_items]
    return {
        "selected_paths": selected,
        "selected_by_model": True,
        "candidates": candidates,
    }

async def maybe_run_memory_extraction(runtime: Any, handle: RunHandle, *, prompt_text: str) -> None:
    self = runtime
    assert self.config and self.paths and self.store
    if not self.config.memory.enabled:
        return
    metadata = await self.store.session_metadata(handle.session_id)
    cursor_seq = int(metadata.get("memory_scan_node_seq") or 0)
    sources = await self.store.memory_source_nodes_since(handle.session_id, cursor_seq)
    if not sources:
        return
    reason = await self._memory_extraction_trigger_reason(
        session_id=handle.session_id,
        sources=sources,
        latest_user_text=prompt_text,
        max_pending_messages=max(1, self.config.memory.extract_every_messages),
        token_threshold=max(1, self.config.memory.extract_every_tokens),
    )
    if reason is None:
        self._schedule_memory_idle_extraction(session_id=handle.session_id)
        return
    await self._run_memory_extraction_for_sources(session_id=handle.session_id, cursor_seq=cursor_seq, sources=sources, reason=reason)

async def memory_extraction_trigger_reason(
    runtime: Any,
    *,
    session_id: str,
    sources: list[tuple[int, Node]],
    latest_user_text: str,
    max_pending_messages: int,
    token_threshold: int,
) -> str | None:
    self = runtime
    if await self._has_explicit_memory_intent(session_id=session_id, latest_user_text=latest_user_text):
        return "explicit"
    if len(sources) >= max_pending_messages:
        return "message_backlog"
    if _estimate_memory_source_tokens([node for _seq, node in sources]) >= token_threshold:
        return "token_backlog"
    return None

async def has_explicit_memory_intent(runtime: Any, *, session_id: str, latest_user_text: str) -> bool:
    self = runtime
    text = latest_user_text.strip()
    if not text:
        return False
    assert self.config
    model_config = resolve_model_config(self.config, self.config.memory.extract_model_profile)
    try:
        decision = await self._run_structured_json_model(
            model_config=model_config,
            schema=_MEMORY_INTENT_SCHEMA,
            purpose="memory_intent_classifier",
            session_id=session_id,
            system_text=(
                "Decide whether the user's latest message explicitly asks the assistant to remember, "
                "store, keep for future conversations, or always apply a user preference/profile fact. "
                "Return JSON with explicit=true only for direct memory/storage intent. "
                "Return explicit=false for ordinary questions, facts, status updates, or tasks that do not ask to remember."
            ),
            user_text=f"Latest user message:\n{text}",
            max_output_tokens=min(model_config.max_output_tokens, 256),
        )
    except AgentCoreError:
        return False
    return bool(decision.get("explicit"))

async def run_structured_json_model(
    runtime: Any,
    *,
    model_config: Any,
    schema: dict[str, Any],
    purpose: str,
    session_id: str,
    system_text: str,
    user_text: str,
    max_output_tokens: int,
) -> dict[str, Any]:
    self = runtime
    request = _structured_json_request(
        model_config=model_config,
        schema=schema,
        purpose=purpose,
        session_id=session_id,
        system_text=system_text,
        user_text=user_text,
        max_output_tokens=max_output_tokens,
    )
    provider = self._provider_for_model(model_config)
    _ensure_provider_supports_request(provider, request)
    delta_parts: list[str] = []
    final_parts: list[str] = []
    tool_payload: dict[str, Any] | None = None
    async for model_event in provider.stream(request):
        if model_event.event_type == "model_text_delta" and model_event.text_delta:
            delta_parts.append(model_event.text_delta)
        elif model_event.event_type == "model_failed":
            error = model_event.error or ErrorPayload(code=ErrorCode.PROVIDER_ERROR, message=f"{purpose} provider failed")
            raise AgentCoreError(error.code, error.message, details=error.details)
        elif model_event.event_type == "model_completed":
            for call in model_event.tool_calls:
                if call.name == "internal.structured_json":
                    tool_payload = dict(call.arguments)
                    break
            for block in model_event.content:
                if getattr(block, "type", None) == "text":
                    final_parts.append(getattr(block, "text", ""))
            break
    if tool_payload is not None:
        return tool_payload
    text = ("".join(final_parts) if final_parts else "".join(delta_parts)).strip()
    parsed = _parse_json_object(text)
    if parsed is None:
        raise AgentCoreError(ErrorCode.PROVIDER_ERROR, f"{purpose} returned invalid JSON")
    return parsed

async def run_memory_extraction_for_sources(
    runtime: Any,
    *,
    session_id: str,
    cursor_seq: int,
    sources: list[tuple[int, Node]],
    reason: str,
) -> None:
    self = runtime
    assert self.config and self.paths and self.store
    source_node_ids = [node.node_id for _seq, node in sources]
    max_seq = max(seq for seq, _node in sources)
    await self.store.add_event(
        make_event(
            session_id=session_id,
            event_type="memory_extraction_started",
            payload={
                "reason": reason,
                "from_node_seq": cursor_seq + 1,
                "to_node_seq": max_seq,
                "source_node_ids": source_node_ids,
            },
        )
    )
    try:
        model_config = resolve_model_config(self.config, self.config.memory.extract_model_profile)
        text = await self._run_memory_extraction_model(
            session_id=session_id,
            model_config=model_config,
            source_nodes=[node for _seq, node in sources],
        )
        candidates = parse_memory_candidates(text)
        allowed_sources = set(source_node_ids)
        invalid_sources = sorted(
            {
                source_node_id
                for candidate in candidates
                for source_node_id in candidate.source_node_ids
                if source_node_id not in allowed_sources
            }
        )
        if invalid_sources:
            raise AgentCoreError(
                ErrorCode.MEMORY_WRITE_FAILED,
                f"memory candidate references source nodes outside extraction range: {invalid_sources}",
            )
        job = MemoryExtractionJob(
            home_dir=self.paths.home_dir,
            memory_dir=resolve_memory_dir(
                self.config.memory.memory_dir,
                home_dir=self.paths.home_dir,
                project_dir=self.paths.project_dir,
            ),
            cursor=MemoryScanCursor(node_seq=cursor_seq),
            source_session_id=session_id,
        )
        result = job.apply(candidates, source_node_seq=max_seq)
        await self.store.update_session_metadata(
            session_id,
            {"memory_scan_node_seq": result.scan_cursor.node_seq},
        )
        await self.store.add_event(
            make_event(
                session_id=session_id,
                event_type="memory_extraction_completed",
                payload={
                    "reason": reason,
                    "created_memory_ids": result.created,
                    "updated_memory_ids": result.updated,
                    "ignored_candidates": result.ignored,
                    "duplicate_decisions": result.duplicate,
                    "conflicts": [],
                    "source_node_ids": source_node_ids,
                    "files_changed": result.files_changed,
                    "scan_cursor": {"node_seq": result.scan_cursor.node_seq},
                },
            )
        )
    except Exception as exc:
        code = getattr(exc, "code", ErrorCode.MEMORY_WRITE_FAILED)
        message = getattr(exc, "message", str(exc))
        await self.store.add_event(
            make_event(
                session_id=session_id,
                event_type="memory_extraction_failed",
                level="error",
                payload={
                    "reason": reason,
                    "code": str(code),
                    "message": redact_value(message[:500]),
                    "from_node_seq": cursor_seq + 1,
                    "to_node_seq": max_seq,
                    "source_node_ids": source_node_ids,
                },
            )
        )

def schedule_memory_idle_extraction(runtime: Any, *, session_id: str) -> None:
    self = runtime
    assert self.config
    self._cancel_memory_idle_task(session_id)
    idle_seconds = max(float(self.config.memory.idle_seconds), 0.0)
    self._memory_idle_tasks[session_id] = asyncio.create_task(self._memory_idle_extraction_after_delay(session_id, idle_seconds))

def cancel_memory_idle_task(runtime: Any, session_id: str) -> None:
    self = runtime
    task = self._memory_idle_tasks.pop(session_id, None)
    if task is not None and not task.done():
        task.cancel()

async def memory_idle_extraction_after_delay(runtime: Any, session_id: str, idle_seconds: float) -> None:
    self = runtime
    try:
        await asyncio.sleep(idle_seconds)
        assert self.config and self.store
        metadata = await self.store.session_metadata(session_id)
        cursor_seq = int(metadata.get("memory_scan_node_seq") or 0)
        sources = await self.store.memory_source_nodes_since(session_id, cursor_seq)
        if sources:
            await self._run_memory_extraction_for_sources(
                session_id=session_id,
                cursor_seq=cursor_seq,
                sources=sources,
                reason="idle",
            )
    except asyncio.CancelledError:
        raise
    finally:
        current = self._memory_idle_tasks.get(session_id)
        if current is asyncio.current_task():
            self._memory_idle_tasks.pop(session_id, None)

async def run_memory_extraction_model(runtime: Any, *, session_id: str, model_config: Any, source_nodes: list[Node]) -> str:
    self = runtime
    assert self.paths
    source_text = _memory_extraction_source_text(source_nodes)
    assert self.config
    catalog = resolve_memory_dir(
        self.config.memory.memory_dir,
        home_dir=self.paths.home_dir,
        project_dir=self.paths.project_dir,
    ) / "MEMORY.md"
    catalog_text = catalog.read_text(encoding="utf-8", errors="replace") if catalog.exists() else "# Memory Catalog\n"
    payload = await self._run_structured_json_model(
        model_config=model_config,
        schema=_MEMORY_EXTRACTION_SCHEMA,
        purpose="memory_extraction",
        session_id=session_id,
        system_text=(
            "You are the Soong Agent memory extraction job. "
            "Decide whether new user-visible context should be written as long-term memory. "
            "For each memory, decision must be \"new\". "
            "Category must be exactly one of these strings: \"user\", \"feedback\", or \"reference\". "
            "Use category \"user\" for user profile, preferences, skills, and facts explicitly requested to remember. "
            "Do not output combined category strings such as \"user|reference\". "
            "Filename must be a local lowercase markdown filename ending in .md, for example backend_developer.md. "
            "source_node_ids must copy node IDs exactly from the source nodes. "
            "Use an empty memories array when nothing should be stored. "
            "Never store secrets, credentials, transient task state, plans, full transcripts, or command output."
        ),
        user_text=(
            f"Session: {session_id}\n\n"
            f"Existing MEMORY.md catalog:\n{catalog_text[:12000]}\n\n"
            f"New source nodes:\n{source_text}"
        ),
        max_output_tokens=model_config.max_output_tokens,
    )
    return json.dumps(payload, ensure_ascii=False)
