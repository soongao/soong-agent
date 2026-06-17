from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from agent_core.api.runtime_helpers.views import _text_sha256
from agent_core.context.skills import build_skill_catalog, find_skill_by_name, read_skill_body, skill_context_text
from agent_core.errors.codes import ErrorCode
from agent_core.types import ErrorPayload, LoadSkillResult, RunMode, SkillInfo, TextBlock


async def list_skills(runtime: Any) -> list[SkillInfo]:
    await runtime._ensure_started()
    assert runtime.paths
    return [SkillInfo(**entry) for entry in build_skill_catalog(runtime.paths.home_dir)]


async def load_skill(
    runtime: Any,
    session_id: str,
    name: str,
    *,
    mode: Literal["normal", "orchestrator"] = "normal",
) -> LoadSkillResult:
    await runtime._ensure_started()
    assert runtime.paths and runtime.store
    name = name.strip()
    if not name:
        return LoadSkillResult(
            session_id=session_id,
            name=name,
            loaded=False,
            error=ErrorPayload(code=ErrorCode.VALIDATION_ERROR, message="skill name is required"),
        )
    if session_id in runtime._session_active or runtime._session_queues.get(session_id) or await runtime.store.has_active_runs(session_id):
        return LoadSkillResult(
            session_id=session_id,
            name=name,
            loaded=False,
            error=ErrorPayload(code=ErrorCode.SESSION_ACTIVE, message="session has active or queued runs"),
        )
    entry = find_skill_by_name(runtime.paths.home_dir, name)
    if entry is None:
        return LoadSkillResult(
            session_id=session_id,
            name=name,
            loaded=False,
            error=ErrorPayload(code=ErrorCode.SKILL_NOT_FOUND, message=f"skill not found: {name}"),
        )
    if entry.get("error") == "duplicate":
        return LoadSkillResult(
            session_id=session_id,
            name=name,
            loaded=False,
            error=ErrorPayload(code=ErrorCode.SKILL_LOAD_FAILED, message=f"duplicate skill name: {name}"),
        )
    run_mode = RunMode(mode)
    session = await runtime.store.get_session(session_id)
    agent_id = str(session["root_agent_id"]) if session is not None else runtime._root_agent_id(session_id=session_id, mode=run_mode)
    session_created = await runtime.store.ensure_session(session_id=session_id, cwd=str(runtime.paths.project_dir), root_agent_id=agent_id)
    await runtime.store.ensure_agent(
        agent_id=agent_id,
        session_id=session_id,
        agent_type="orchestrator" if run_mode == RunMode.ORCHESTRATOR else "main",
    )
    if session_created:
        await runtime._run_observe_hook(
            event_type="session_started",
            session_id=session_id,
            agent_id=agent_id,
            run_id=None,
            payload={
                "event_type": "SessionStart",
                "session_id": session_id,
                "agent_id": agent_id,
                "run_id": None,
                "mode": run_mode.value,
                "cwd": str(runtime.paths.project_dir),
            },
        )
    path = Path(entry["path"]).resolve()
    body = read_skill_body(path)
    already_loaded = runtime._context_state_for_session(session_id).mark_skill(name, path, body)
    digest = _text_sha256(body)
    if already_loaded:
        return LoadSkillResult(
            session_id=session_id,
            name=name,
            path=str(path),
            hash=digest,
            loaded=True,
            already_loaded=True,
        )
    parent_id = await runtime.store.active_node_id(session_id)
    node = await runtime.store.add_node(
        session_id=session_id,
        parent_id=parent_id,
        agent_id=agent_id,
        run_id=None,
        role="user",
        node_type="skill_context",
        content=[TextBlock(text=skill_context_text(name=name, body=body))],
        metadata={"synthetic": True, "source": "runtime.load_skill", "name": name, "path": str(path), "hash": digest},
        make_active=True,
    )
    return LoadSkillResult(
        session_id=session_id,
        name=name,
        path=str(path),
        hash=digest,
        node_id=node.node_id,
        loaded=True,
        already_loaded=False,
    )
