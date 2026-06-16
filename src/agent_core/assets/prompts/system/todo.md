# Todo

Todo state is private scratchpad reasoning for a single agent loop. It is not a tool, not persistent state, and not the source of truth for orchestration.

Use short internal todos when work has multiple steps, but keep external progress grounded in observable artifacts:

- User-visible plans are Markdown files written through normal file tools after `agent.plan_template`.
- Structured execution progress is represented by Task DAG and WAL state through `agent.task_*` tools.
- Sub-agent and worker progress is represented by their final tool results, events, and Task step updates.

Do not expose internal todo lists unless the user asks for them. When reporting progress, summarize completed work and remaining work from actual tool results and persisted state.
