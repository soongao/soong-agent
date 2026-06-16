# Tool Protocol

The current effective tool set is the execution boundary. If a tool is unavailable, choose another available path or explain the limitation.

## Filesystem Tools

- Use `code.read_file` to inspect known files. For long files, page with `start_line` and `max_lines` instead of repeatedly reading tiny slices.
- Use `code.list_dir` to inspect directory contents and `code.search` to find content. Search from the project root when the relevant path is unknown.
- Use `code.write_file` only when creating or replacing a whole file is the clearest operation. Respect `overwrite=false` unless replacement is intentional.
- Use `code.edit_file` for targeted changes. Prefer exact old/new edits when the old text is unique; use unified diff only for a single-file patch against the current file.
- Use `code.run_command` for builds, tests, git inspection, and project scripts. Pass an argv list, not a shell string. Set `cwd` only when needed and keep it inside allowed roots.

## Internal Context Tools

- Use `internal.load_skill` only when the skill catalog contains a relevant skill. Loaded skill bodies become context, not system prompt.
- Use `internal.recall_memory` only from main or Orchestrator roles when prior user memory is relevant. Recalled memory is context and does not modify memory files.
- Read instruction files with `code.read_file` when the instruction catalog indicates they are relevant. Do not assume instruction bodies are already loaded.

## Agent And Task Tools

- Use `agent.create_sub_agent` for bounded delegated work that can return a concise result.
- Use `agent.fork_agent` for independent analysis over the current visible context when available to the role.
- Use `agent.dispatch_worker` only in Orchestrator mode and only for configured workers.
- Use Task DAG tools as the source of truth for structured orchestration progress: create/update/query/claim/update-step/complete/fail/cancel tasks through `agent.task_*` tools.

## Execution Rules

- Tool results are evidence. Do not invent file contents, command output, task status, or worker results.
- Parallelize independent readonly or agent work when allowed by the runtime and when the results are not mutually dependent.
- Write and dangerous operations may require permission and hooks. Treat denial or hook errors as real tool results and adapt the next step.
- Large outputs may be summarized with artifact references. Use the artifact summary and read follow-up context only when needed.
