from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agent_core.config.paths import resolve_home_dir
from agent_core.errors import AgentCoreError
from agent_core.providers import ProviderRegistry
from agent_hub.backend.config import HubConfigBootstrapError, HubConfigValidationError, bootstrap_default_config, hub_db_path, validate_config
from agent_hub.backend.database import HubDatabase
from agent_hub.backend.errors import HubApiError, agent_core_error_response, hub_error_response
from agent_hub.backend.events import HubEventHub
from agent_hub.backend.permissions import PermissionBridge
from agent_hub.backend.routes import conversations, events, health, permissions, tools, workers
from agent_hub.backend.runtime import HubRuntimeBridge
from agent_hub.backend.state import HubAppState
from agent_hub.backend.workers import seed_default_workers

logger = logging.getLogger(__name__)


def create_app(
    *,
    home_dir: str | Path | None = None,
    project_dir: str | Path | None = None,
    provider_registry: ProviderRegistry | None = None,
) -> FastAPI:
    resolved_home = resolve_home_dir(home_dir)
    resolved_project = Path(project_dir or Path.cwd()).expanduser().resolve()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("agenthub startup project_dir=%s home_dir=%s", resolved_project, resolved_home)
        db = HubDatabase(hub_db_path(home_dir=resolved_home))
        await db.open()
        event_hub = HubEventHub()
        permission_bridge = PermissionBridge(db, event_hub)
        runtime_bridge: HubRuntimeBridge | None = None
        startup_error: dict | None = None
        config_info: dict = {"config_path": str(resolved_home / "config.toml")}
        try:
            bootstrap_default_config(home_dir=resolved_home)
            config_info = validate_config(home_dir=resolved_home)
            runtime_bridge = HubRuntimeBridge(
                db=db,
                events=event_hub,
                permission_bridge=permission_bridge,
                project_dir=resolved_project,
                home_dir=resolved_home,
                provider_registry=provider_registry,
            )
            await runtime_bridge.start()
            await seed_default_workers(runtime_bridge.runtime)
            logger.info(
                "agenthub runtime initialized provider=%s model=%s",
                config_info.get("provider"),
                config_info.get("model"),
            )
        except HubConfigBootstrapError as exc:
            startup_error = {"code": "config_bootstrap_failed", "message": str(exc), "details": {}}
            logger.warning("agenthub config bootstrap failed: %s", exc)
        except HubConfigValidationError as exc:
            startup_error = {"code": "config_invalid", "message": str(exc), "details": {}}
            logger.warning("agenthub config invalid: %s", exc)
        except Exception as exc:
            startup_error = {"code": "core_start_failed", "message": str(exc), "details": {"type": exc.__class__.__name__}}
            logger.exception("agenthub core startup failed")
        app.state.hub = HubAppState(
            home_dir=resolved_home,
            project_dir=resolved_project,
            config_path=resolved_home / "config.toml",
            db=db,
            event_hub=event_hub,
            permission_bridge=permission_bridge,
            runtime_bridge=runtime_bridge,
            config_info=config_info,
            startup_error=startup_error,
        )
        try:
            yield
        finally:
            logger.info("agenthub shutdown")
            await permission_bridge.shutdown()
            if runtime_bridge is not None:
                await runtime_bridge.close()
            await db.close()

    app = FastAPI(title="Agent Hub", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^http://(127\.0\.0\.1|localhost):\d+$",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health.router)
    app.include_router(events.router)
    app.include_router(conversations.router)
    app.include_router(workers.router)
    app.include_router(tools.router)
    app.include_router(permissions.router)
    app.add_exception_handler(HubApiError, lambda _request, exc: hub_error_response(exc))
    app.add_exception_handler(AgentCoreError, lambda _request, exc: agent_core_error_response(exc))
    return app


app = create_app()
