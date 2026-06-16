from __future__ import annotations

from typing import Any

from agent_core.assets.loader import read_asset
from agent_core.config.paths import expand_config_path
from agent_core.tasks.service import TaskService
from agent_core.tools.execution import ToolExecutionContext
from agent_core.tools.registry import ToolRegistry
from agent_core.types.tools import ToolDefinition


PLAN_TEMPLATE_ID = "template.plan.default"
TASK_TEMPLATE_ID = "template.task_dag.default"
TEMPLATE_VERSION = "1"

_STRING_LIST_SCHEMA = {"type": "array", "items": {"type": "string"}}
_TASK_STEP_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "step_id": {"type": "string"},
        "title": {"type": "string"},
        "summary": {"type": "string"},
        "status": {"type": "string"},
        "depends_on_step_ids": _STRING_LIST_SCHEMA,
        "required": {"type": "boolean"},
        "worker_pool_id": {"type": ["string", "null"]},
    },
    "required": ["step_id", "title"],
}
_TASK_OPERATION_SCHEMA = {
    "type": "object",
    "properties": {
        "op": {
            "type": "string",
            "enum": [
                "update_task",
                "add_step",
                "update_step",
                "delete_step",
                "add_dependency",
                "remove_dependency",
                "cancel_step",
                "reopen_step",
            ],
        },
        "title": {"type": "string"},
        "summary": {"type": "string"},
        "status": {"type": "string"},
        "reason": {"type": ["string", "null"]},
        "step": _TASK_STEP_INPUT_SCHEMA,
        "step_id": {"type": "string"},
        "depends_on_step_id": {"type": "string"},
        "depends_on_step_ids": _STRING_LIST_SCHEMA,
        "worker_pool_id": {"type": ["string", "null"]},
        "required": {"type": "boolean"},
    },
    "required": ["op"],
}


def register_task_tools(registry: ToolRegistry, service: TaskService) -> None:
    registry.register_tool(
        _definition("agent.plan_template", "Return the built-in plan writing template.", {"goal": {"type": ["string", "null"]}, "suggested_dir": {"type": ["string", "null"]}}),
        plan_template,
    )
    registry.register_tool(
        _definition("agent.task_template", "Return the built-in Task DAG template.", {"goal": {"type": ["string", "null"]}}),
        task_template,
    )
    registry.register_tool(
        _definition(
            "agent.task_create",
            "Create a Task DAG and project Task WAL.",
            {
                "task_id": {"type": "string"},
                "wal_name": {"type": "string"},
                "title": {"type": "string"},
                "summary": {"type": "string"},
                "steps": {"type": "array", "items": _TASK_STEP_INPUT_SCHEMA},
            },
            required=["task_id", "wal_name", "title", "summary", "steps"],
        ),
        lambda context, args: _call(service.create_task, context, args),
    )
    registry.register_tool(
        _definition("agent.task_get", "Get a Task DAG.", {"task_id": {"type": "string"}, "include_terminal_steps": {"type": "boolean"}}, required=["task_id"]),
        lambda context, args: _call(service.get_task, context, args),
    )
    registry.register_tool(
        _definition(
            "agent.task_list",
            "List tasks.",
            {"status": {"type": ["string", "null"]}, "include_terminal": {"type": "boolean"}, "limit": {"type": "integer"}, "offset": {"type": "integer"}},
        ),
        lambda context, args: _call(service.list_tasks, context, args),
    )
    registry.register_tool(
        _definition(
            "agent.task_update",
            "Apply structured Task DAG patch operations.",
            {"task_id": {"type": "string"}, "operations": {"type": "array", "items": _TASK_OPERATION_SCHEMA}},
            required=["task_id", "operations"],
        ),
        lambda context, args: _call(service.update_task, context, args),
    )
    registry.register_tool(
        _definition(
            "agent.task_query_steps",
            "Query task steps.",
            {
                "task_id": {"type": "string"},
                "statuses": {"type": ["array", "null"], "items": {"type": "string"}},
                "worker_pool_id": {"type": ["string", "null"]},
                "claimed_by_agent_id": {"type": ["string", "null"]},
                "include_terminal_steps": {"type": "boolean"},
                "limit": {"type": "integer"},
                "offset": {"type": "integer"},
            },
            required=["task_id"],
        ),
        lambda context, args: _call(service.query_steps, context, args),
    )
    registry.register_tool(
        _definition("agent.task_claim_step", "Claim a ready task step.", {"task_id": {"type": "string"}, "step_id": {"type": "string"}}, required=["task_id", "step_id"]),
        lambda context, args: _call(service.claim_step, context, args),
    )
    registry.register_tool(
        _definition(
            "agent.task_update_step",
            "Update execution state and result of a task step.",
            {
                "task_id": {"type": "string"},
                "step_id": {"type": "string"},
                "status": {"type": ["string", "null"]},
                "result_summary": {"type": ["string", "null"]},
                "artifact_ids": {"type": ["array", "null"], "items": {"type": "string"}},
                "reason": {"type": ["string", "null"]},
            },
            required=["task_id", "step_id"],
        ),
        lambda context, args: _call(service.update_step, context, args),
    )
    registry.register_tool(
        _definition("agent.task_complete", "Complete a task.", {"task_id": {"type": "string"}, "result_summary": {"type": ["string", "null"]}}, required=["task_id"]),
        lambda context, args: _call(service.complete_task, context, args),
    )
    registry.register_tool(
        _definition("agent.task_fail", "Fail a task and terminate unfinished steps.", {"task_id": {"type": "string"}, "reason": {"type": ["string", "null"]}}, required=["task_id"]),
        lambda context, args: _terminate_task_with_runtime(service.fail_task, context, args),
    )
    registry.register_tool(
        _definition("agent.task_cancel", "Cancel a task and terminate unfinished steps.", {"task_id": {"type": "string"}, "reason": {"type": ["string", "null"]}}, required=["task_id"]),
        lambda context, args: _terminate_task_with_runtime(service.cancel_task, context, args),
    )


async def plan_template(context: ToolExecutionContext, args: dict[str, Any]) -> dict[str, Any]:
    content = read_asset(PLAN_TEMPLATE_ID)
    suggested_dir = args.get("suggested_dir")
    if suggested_dir is None:
        suggested_dir = str(
            expand_config_path(
                context.config.plan.default_dir,
                home_dir=context.home_dir,
                project_dir=context.project_dir,
            ).resolve()
        )
    return {
        "node_type": "plan_instruction",
        "content": content,
        "goal": args.get("goal"),
        "suggested_dir": suggested_dir,
        "template_id": PLAN_TEMPLATE_ID,
        "template_version": TEMPLATE_VERSION,
        "metadata": {"template_id": PLAN_TEMPLATE_ID, "template_version": TEMPLATE_VERSION},
    }


async def task_template(context: ToolExecutionContext, args: dict[str, Any]) -> dict[str, Any]:
    content = read_asset(TASK_TEMPLATE_ID)
    return {
        "node_type": "task_instruction",
        "content": content,
        "goal": args.get("goal"),
        "template_id": TASK_TEMPLATE_ID,
        "template_version": TEMPLATE_VERSION,
        "metadata": {"template_id": TASK_TEMPLATE_ID, "template_version": TEMPLATE_VERSION},
    }


async def _call(func, context: ToolExecutionContext, args: dict[str, Any]) -> dict[str, Any]:
    return func(context, args)


async def _terminate_task_with_runtime(func, context: ToolExecutionContext, args: dict[str, Any]) -> dict[str, Any]:
    result = func(context, args)
    runtime = context.services.get("runtime") if context.services else None
    if runtime is not None:
        await runtime.cancel_worker_runs(
            session_id=context.session_id,
            task_id=str(args["task_id"]),
            worker_run_ids=result.get("terminated_worker_run_ids") or [],
            reason=str(args.get("reason") or "task_terminated"),
        )
    return result


def _definition(name: str, description: str, properties: dict[str, Any], *, required: list[str] | None = None) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=description,
        input_schema={"type": "object", "properties": properties, "required": required or []},
        permission="write" if name in _WRITE_TASK_TOOLS else "readonly",
        tags={"agent", "task"} if name.startswith("agent.task") else {"agent"},
    )


_WRITE_TASK_TOOLS = {
    "agent.task_create",
    "agent.task_update",
    "agent.task_claim_step",
    "agent.task_update_step",
    "agent.task_complete",
    "agent.task_fail",
    "agent.task_cancel",
}
