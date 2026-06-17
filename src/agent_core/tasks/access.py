from __future__ import annotations

from agent_core.errors import AgentCoreError
from agent_core.errors.codes import ErrorCode
from agent_core.tasks.helpers import worker_scope
from agent_core.tasks.models import TaskStep
from agent_core.tools.execution import ToolExecutionContext


def ensure_worker_task_scope(context: ToolExecutionContext, task_id: str) -> None:
    if context.agent_role != "worker":
        return
    scope = worker_scope(context)
    if scope.get("task_id") != task_id:
        raise AgentCoreError(ErrorCode.PERMISSION_DENIED, "worker cannot access task outside dispatch scope")


def step_visible_to_worker(context: ToolExecutionContext, step: TaskStep) -> bool:
    if context.agent_role != "worker":
        return True
    scope = worker_scope(context)
    allowed_step_ids = scope.get("allowed_step_ids")
    if allowed_step_ids is not None and step.step_id not in allowed_step_ids:
        return False
    worker_pool_id = scope.get("worker_pool_id")
    if worker_pool_id and step.worker_pool_id and step.worker_pool_id != worker_pool_id:
        return False
    return True
