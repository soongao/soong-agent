from __future__ import annotations

import logging

from agent_hub.backend.workers.definitions import DEFAULT_HUB_WORKERS

logger = logging.getLogger(__name__)


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
