# Default Task DAG Template

Create a Task DAG that can be written with `agent.task_create`. The DAG is the source of truth for Orchestrator/worker progress, not a prose checklist.

Required shape:

1. `task_id`: stable, lowercase, and safe for filenames.
2. `wal_name`: a matching `.wal.jsonl` file name.
3. `title`: short human-readable task title.
4. `summary`: one paragraph describing the work.
5. `steps`: step objects with stable `step_id`, `title`, optional `summary`, optional `depends_on_step_ids`, optional `worker_pool_id`, and optional `required`.

Step design:

- Prefer small steps with clear completion criteria.
- Express ordering with `depends_on_step_ids`, not list position.
- Use `required=false` only for optional exploration that should not block task completion.
- Assign `worker_pool_id` only when the work can be dispatched independently to a configured worker pool.
- Keep content fields editable by the Orchestrator; workers should only claim and update execution status/results.

Before creating the task, check that dependencies are acyclic and every referenced step exists.
