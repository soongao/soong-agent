from __future__ import annotations

import logging

from agent_core.types import WorkerConfigCreate

logger = logging.getLogger(__name__)


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
)


async def seed_default_workers(runtime) -> list[str]:
    created: list[str] = []
    for worker in DEFAULT_HUB_WORKERS:
        existing = await runtime.get_worker_config(worker.worker_id)
        if existing is not None:
            continue
        try:
            await runtime.create_worker_config(worker)
        except Exception:
            logger.exception("failed to seed default Hub worker worker_id=%s", worker.worker_id)
            raise
        created.append(worker.worker_id)
    if created:
        logger.info("seeded default Hub workers worker_ids=%s", ",".join(created))
    return created
