from __future__ import annotations

from agent_core.agents.registry import AgentDefinitionRegistry
from agent_core.agents.workers import WorkerPoolRuntime
from agent_core.errors import AgentCoreError
from agent_core.errors.codes import ErrorCode
from agent_core.tools.execution import ToolExecutionContext
from agent_core.tools.registry import ToolRegistry
from agent_core.types.tools import ToolDefinition


def register_agent_tools(
    registry: ToolRegistry,
    definitions: AgentDefinitionRegistry,
    workers: WorkerPoolRuntime | None = None,
) -> None:
    registry.register_tool(
        ToolDefinition(
            name="agent.list_agent_definitions",
            description="List available AgentDefinition catalog entries.",
            input_schema={"type": "object", "properties": {}, "required": []},
            permission="readonly",
            tags={"agent", "readonly"},
        ),
        lambda context, args: list_agent_definitions(context, args, definitions),
    )
    registry.register_tool(
        ToolDefinition(
            name="agent.list_workers",
            description="List configured worker pool runtime states.",
            input_schema={
                "type": "object",
                "properties": {"worker_pool_id": {"type": ["string", "null"]}},
                "required": [],
            },
            permission="readonly",
            tags={"agent", "readonly"},
        ),
        lambda context, args: list_workers(context, args, workers),
    )
    registry.register_tool(
        ToolDefinition(
            name="agent.create_sub_agent",
            description="Create a sub agent run from an AgentDefinition.",
            input_schema={
                "type": "object",
                "properties": {
                    "agent_definition_id": {"type": ["string", "null"]},
                    "task": {"type": "string"},
                    "context": {"type": ["string", "null"]},
                    "constraints": {"type": ["object", "null"]},
                    "allowed_tools": {"type": ["array", "null"]},
                    "expected_output_schema": {"type": ["object", "null"]},
                    "timeout_ms": {"type": ["integer", "null"]},
                },
                "required": ["task"],
            },
            permission="readonly",
            tags={"agent"},
        ),
        lambda context, args: create_sub_agent(context, args),
    )
    registry.register_tool(
        ToolDefinition(
            name="agent.fork_agent",
            description="Fork an agent run from an AgentDefinition.",
            input_schema={
                "type": "object",
                "properties": {
                    "agent_definition_id": {"type": ["string", "null"]},
                    "task": {"type": "string"},
                    "constraints": {"type": ["object", "null"]},
                    "allowed_tools": {"type": ["array", "null"]},
                    "expected_output_schema": {"type": ["object", "null"]},
                    "timeout_ms": {"type": ["integer", "null"]},
                },
                "required": ["task"],
            },
            permission="readonly",
            tags={"agent"},
        ),
        lambda context, args: fork_agent(context, args),
    )
    registry.register_tool(
        ToolDefinition(
            name="agent.dispatch_worker",
            description="Dispatch work to an idle configured worker.",
            input_schema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "worker_pool_id": {"type": ["string", "null"]},
                    "worker_agent_id": {"type": ["string", "null"]},
                    "allowed_step_ids": {"type": ["array", "null"]},
                    "instruction": {"type": "string"},
                    "context": {"type": ["string", "null"]},
                    "constraints": {"type": ["object", "null"]},
                    "allowed_tools": {"type": ["array", "null"]},
                    "expected_output_schema": {"type": ["object", "null"]},
                    "timeout_ms": {"type": ["integer", "null"]},
                },
                "required": ["task_id", "instruction"],
            },
            permission="readonly",
            tags={"agent", "worker"},
        ),
        lambda context, args: dispatch_worker(context, args, workers),
    )


async def list_agent_definitions(context: ToolExecutionContext, args: dict, definitions: AgentDefinitionRegistry) -> dict:
    return {"agent_definitions": [definition.model_dump(mode="json") for definition in definitions.list()]}


async def list_workers(context: ToolExecutionContext, args: dict, workers: WorkerPoolRuntime | None) -> dict:
    if workers is None:
        return {"workers": []}
    return {
        "workers": [
            {
                "worker_id": worker.worker_id,
                "worker_pool_id": worker.pool_id,
                "agent_definition_id": worker.agent_definition_id,
                "status": worker.status,
                "allowed_tools": worker.allowed_tools,
            }
            for worker in workers.list_workers(args.get("worker_pool_id"))
        ]
    }


async def create_sub_agent(context: ToolExecutionContext, args: dict) -> dict:
    runtime = context.service("runtime")
    agent_definition_id = args.get("agent_definition_id") or context.config.agents.default_sub_agent_definition
    task = str(args["task"])
    if args.get("context"):
        task = task + "\n\nContext:\n" + str(args["context"])
    return await runtime.run_child_agent(
        session_id=context.session_id,
        parent_run_id=context.run_id,
        parent_agent_id=context.agent_id,
        agent_definition_id=agent_definition_id,
        task=task,
        mode="sub",
        constraints=args.get("constraints"),
        allowed_tools=args.get("allowed_tools"),
        expected_output_schema=args.get("expected_output_schema"),
        timeout_ms=args.get("timeout_ms"),
        parent_handle=context.run_handle,
    )


async def fork_agent(context: ToolExecutionContext, args: dict) -> dict:
    runtime = context.service("runtime")
    agent_definition_id = args.get("agent_definition_id") or context.config.agents.default_fork_agent_definition
    return await runtime.run_child_agent(
        session_id=context.session_id,
        parent_run_id=context.run_id,
        parent_agent_id=context.agent_id,
        agent_definition_id=agent_definition_id,
        task=str(args["task"]),
        mode="fork",
        constraints=args.get("constraints"),
        allowed_tools=args.get("allowed_tools"),
        expected_output_schema=args.get("expected_output_schema"),
        timeout_ms=args.get("timeout_ms"),
        parent_handle=context.run_handle,
    )


async def dispatch_worker(context: ToolExecutionContext, args: dict, workers: WorkerPoolRuntime | None) -> dict:
    if workers is None:
        raise AgentCoreError(ErrorCode.WORKER_NOT_AVAILABLE, "worker runtime not configured")
    if context.services and "runtime" in context.services:
        runtime = context.service("runtime")
        return await runtime.run_worker_agent(
            session_id=context.session_id,
            parent_run_id=context.run_id,
            parent_agent_id=context.agent_id,
            task_id=str(args["task_id"]),
            instruction=str(args["instruction"]),
            worker_pool_id=args.get("worker_pool_id"),
            worker_agent_id=args.get("worker_agent_id"),
            allowed_step_ids=args.get("allowed_step_ids"),
            dispatch_context=args.get("context"),
            constraints=args.get("constraints"),
            allowed_tools=args.get("allowed_tools"),
            expected_output_schema=args.get("expected_output_schema"),
            timeout_ms=args.get("timeout_ms"),
            parent_handle=context.run_handle,
        )
    allowed_step_ids = args.get("allowed_step_ids")
    if allowed_step_ids == []:
        raise AgentCoreError(ErrorCode.VALIDATION_ERROR, "allowed_step_ids cannot be empty")
    allowed_set = set(str(item) for item in allowed_step_ids) if allowed_step_ids is not None else None
    worker = workers.select_worker(worker_pool_id=args.get("worker_pool_id"), worker_agent_id=args.get("worker_agent_id"))
    task_service = context.service("task_service")
    query = task_service.query_steps(
        context,
        {
            "task_id": args["task_id"],
            "statuses": ["ready"],
            "include_terminal_steps": False,
            "limit": 50,
            "offset": 0,
        },
    )
    candidates = query["steps"]
    if allowed_set is not None:
        candidates = [step for step in candidates if step["step_id"] in allowed_set]
    if not candidates:
        return {
            "worker_id": worker.worker_id,
            "claimed_step_id": None,
            "step_status": None,
            "step_result_summary": None,
            "no_step_claimed": True,
        }
    chosen = candidates[0]
    workers.mark_busy(worker, task_id=str(args["task_id"]))
    try:
        claimed = task_service.claim_step(context, {"task_id": args["task_id"], "step_id": chosen["step_id"]})
        return {
            "worker_id": worker.worker_id,
            "claimed_step_id": claimed["step"]["step_id"],
            "step_status": claimed["step"]["status"],
            "step_result_summary": claimed["step"].get("result_summary"),
            "no_step_claimed": False,
        }
    finally:
        workers.mark_idle(worker)
