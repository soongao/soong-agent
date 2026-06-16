# Default Task DAG Template

Create a Task DAG that can be written with `agent.task_create`.

Required shape:

1. `task_id`: stable, lowercase, and safe for filenames.
2. `wal_name`: a matching `.wal.jsonl` file name.
3. `title`: short human-readable task title.
4. `summary`: one paragraph describing the work.
5. `steps`: ordered step objects with stable `step_id`, `title`, optional `summary`, optional `depends_on_step_ids`, optional `worker_pool_id`, and optional `required`.

Prefer small steps with explicit dependencies. Only assign worker pools when the work can be dispatched independently.
