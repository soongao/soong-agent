# Multi Agent

Sub agents, fork agents, and workers are separate runs with their own effective tool sets and context boundaries. Use them only when delegation materially advances the user's goal.

## Delegation

- Delegate bounded, self-contained work with clear expected output.
- Do not delegate a task when the next local step is blocked on the answer and direct local inspection would be faster.
- Pass enough context for the child or worker to act without guessing, but do not pass unrelated transcript history.
- `allowed_tools` can only narrow a child or worker tool set. It never expands permissions.

## Sub And Fork Agents

- `agent.create_sub_agent` starts a child from an AgentDefinition and explicit task/context.
- `agent.fork_agent` analyzes from the visible active path when available. It does not recover sibling branches or hidden provider prompts.
- Child and fork results should be concise and structured enough for the parent to continue without reading the full child transcript.

## Workers And Task DAG

- Orchestrator owns Task DAG structure, worker selection, dispatch scope, and task terminal decisions.
- Workers only operate inside the dispatched `task_id` and optional `allowed_step_ids`.
- Workers claim at most one ready step per run, update execution status through `agent.task_update_step`, and leave content/DAG topology changes to the Orchestrator.
- `no_step_claimed` is a successful worker result, not a fatal error.

## Failure And Cancellation

- A failed child or worker result does not automatically fail the parent run. Use the result to decide whether to retry, adjust scope, or update the Task.
- Task-level fail/cancel terminates unfinished steps and cancels relevant worker runs according to the runtime contract.
